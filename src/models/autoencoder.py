# src/models/autoencoder.py
import math
import random
from math import radians
import numpy as np
import logging
from contextlib import contextmanager

import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.normal import Normal
from torchvision.ops import batched_nms, nms
import torchvision.transforms as T
import pytorch_lightning as pl
from torchmetrics.functional.image import total_variation

from ldm.models.autoencoder import AutoencoderKL
from ldm.util import instantiate_from_config
from ldm.modules.ema import LitEma

from src.modules.autoencodermodules.feat_encoder import FeatEncoder
from src.modules.autoencodermodules.feat_decoder import FeatDecoder, AdaptiveFeatDecoder
from src.modules.autoencodermodules.pose_decoder import PoseDecoderSpatialVAE as PoseDecoder
from src.util.distributions import DiagonalGaussianDistribution
from src.util.misc import flip_tensor    
# from src.util.misc import ReflectPadCenterCrop
from src.data.specs import LABEL_NAME2ID, LABEL_ID2NAME, CAM_NAMESPACE,  POSE_DIM, LHW_DIM, BBOX_3D_DIM, BACKGROUND_CLASS_IDX, BBOX_DIM, POSE_6D_DIM, FILL_FACTOR_DIM, FINAL_PERTURB_RAD

try:
    import wandb
except ImportWarning:
    print("WandB not installed")
    wandb = None


class Autoencoder(AutoencoderKL):
    """Autoencoder model with KL divergence loss."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class PoseAutoencoder(AutoencoderKL):
    """
    Autoencoder model for pose encoding and decoding.
    """

    def __init__(self,
                ddconfig,
                lossconfig,
                embed_dim,
                ckpt_path=None,
                ignore_keys=[],
                image_mask_key=None,
                image_rgb_key="patch",
                pose_key="pose_6d",
                fill_factor_key="fill_factor",
                pose_perturbed_key="pose_6d_perturbed",
                class_key="class_id",
                bbox_key="bbox_sizes",
                colorize_nlabels=None,
                monitor=None,
                pose_decoder_config=None,
                dropout_prob_init=1.0,
                dropout_prob_final=0.7,
                dropout_warmup_steps=5000,
                pose_conditioned_generation_steps=10000,
                perturb_rad_warmup_steps=100000,
                intermediate_img_feature_leak_steps=0,
                add_noise_to_z_obj=False,
                train_on_yaw=True,
                ema_decay=0.999,
                apply_conv_crop_img_space=False,
                apply_conv_crop_latent_space=False,
                quantconfig=None,
                quantize_obj=False,
                quantize_pose=False,
                # latent_manual_crop_reflect_pad=False,
                **kwargs
                ):
        pl.LightningModule.__init__(self)
        
        # Inputs
        self.image_rgb_key = image_rgb_key
        self.pose_key = pose_key
        self.pose_perturbed_key = pose_perturbed_key
        self.class_key = class_key
        self.bbox_key = bbox_key
        self.fill_factor_key = fill_factor_key
        self.image_mask_key = image_mask_key
        self.train_on_yaw = train_on_yaw
        
        # Encoder Setup
        self.encoder = FeatEncoder(**ddconfig)
        self.obj_quantization = quantize_obj
        self.pose_quantization = quantize_pose
        
        quant_conv_obj_embed_dim = embed_dim if quantize_obj else 2*embed_dim
        
        if quantize_obj:
            assert quantconfig is not None, "quantconfig is not defined but quantize_obj is set to True."
            quantconfig["params"]["e_dim"] = embed_dim
            self.quantize_obj = instantiate_from_config(quantconfig)
        
        if quantize_pose:
            assert quantconfig is not None, "quantconfig is not defined but quantize_pose is set to True."
            quantconfig["params"]["e_dim"] = embed_dim
            self.quantize_pose = instantiate_from_config(quantconfig)

        mult_z = 2 if ddconfig["double_z"] else 1
            
        self.quant_conv_obj = torch.nn.Conv2d(mult_z*ddconfig["z_channels"], quant_conv_obj_embed_dim, 1)
        self.quant_conv_pose = torch.nn.Conv2d(mult_z*ddconfig["z_channels"], embed_dim, 1) # TODO: Need to fix the dimensions
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if quantconfig is not None:
            self.quantization = True
            self.quantize = instantiate_from_config(quantconfig)
        else:
            self.quantization = False
            
        self.encoder_pretrain_steps = lossconfig["params"]["encoder_pretrain_steps"]
        
        # Object latent
        spatial_dim = (np.array([1/k for k in ddconfig["ch_mult"]]).prod()  * 256).astype(int)
        self.add_noise_to_z_obj = add_noise_to_z_obj
        self.feat_dims = [embed_dim, spatial_dim, spatial_dim]
        
        self.z_channels = ddconfig["z_channels"]
        
        pose_decoder_config["params"]["num_channels"] = ddconfig["z_channels"]
        
        # Pose prediction and latent
        self.pose_decoder = instantiate_from_config(pose_decoder_config)
        
        # Decoder Setup
        # self.apply_conv_crop_img_space = apply_conv_crop_img_space
        # self.apply_conv_crop_latent_space = apply_conv_crop_latent_space
        # self.latent_manual_crop_reflect_pad = latent_manual_crop_reflect_pad
        # assert not (self.apply_conv_crop_img_space and self.apply_conv_crop_latent_space), "Only one of the crop types (image or latent space) can be applied"
        self.decoder = FeatDecoder(**ddconfig)
        
        # Dropout Setup
        self.dropout_prob_final = dropout_prob_final # 0.7
        self.dropout_prob_init = dropout_prob_init # 1.0
        self.dropout_prob = self.dropout_prob_init
        self.dropout_warmup_steps = dropout_warmup_steps # 10000 (after stage 1: encoder pretraining)
        self.pose_conditioned_generation_steps = pose_conditioned_generation_steps # 10000
        self.intermediate_img_feature_leak_steps = intermediate_img_feature_leak_steps

        # Loss setup
        lossconfig["params"]["pose_conditioned_generation_steps"] = pose_conditioned_generation_steps
        lossconfig["params"]["train_on_yaw"] = self.train_on_yaw
        if not quantize_obj and not quantize_pose:
            lossconfig["params"]["codebook_weight"] = 0.0
        self.loss = instantiate_from_config(lossconfig)
        self.num_classes = lossconfig["params"]["num_classes"]
        
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
            
        self.perturb_rad_warmup_steps = perturb_rad_warmup_steps
        
        # Checkpointing
        self.use_ema = ema_decay is not None
        if self.use_ema:
            self.ema_decay = ema_decay
            assert 0. < ema_decay < 1.
            self.model_ema = LitEma(self, decay=ema_decay)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")
        
        if ckpt_path is not None:
            try:
                self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
            except Exception:
                # add optimizer to ignore_keys list
                ignore_keys = ignore_keys + ["optimizer"]
                self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.iter_counter = 0

        ### Placeholders
        self.patch_center_rad = None
        self.patch_center_rad_init = None


        ## Inverse refinement cfgs
        self.chunk_size = 128
        self.class_thresh = 0.3 # TODO: Set to 0.5 or other value that works on val set
        self.fill_factor_thresh = 0.5 # TODO: Set to 0.5 or other value that works on val set
        self.num_refinement_steps = 10
        self.ref_lr=1.0e0

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def on_train_batch_end(self, *args, **kwargs):
        # pass
        if self.use_ema:
            self.model_ema(self)
                
    def _decode_pose_to_distribution(self, z, sample_posterior=True):
        # no mu and std dev for class prediction, this is just a class confidence score list of len num_classes
        c_pred = z[..., -self.num_classes:] # torch.Size([8, num_classes])
        # the rest are the moments. first POSE_6D_DIM + LHW_DIM are mu and last POSE_6D_DIM + LHW_DIM are std dev
        bbox_mu = z[..., :POSE_6D_DIM + LHW_DIM + FILL_FACTOR_DIM] # torch.Size([8, 9])
        bbox_std = z[..., POSE_6D_DIM + LHW_DIM + FILL_FACTOR_DIM:2*(POSE_6D_DIM + LHW_DIM + FILL_FACTOR_DIM)] # torch.Size([8, 9])
        bbox_moments = torch.cat([bbox_mu, bbox_std], dim=-1) # torch.Size([8, 18])
        bbox_moments = bbox_moments.to(self.device)
        bbox_posterior = DiagonalGaussianDistribution(bbox_moments)
        return bbox_posterior, c_pred
    
    def decode_pose(self, x, sample_posterior=True):
        """
        Decode the pose from the given image feature map.
        
        Args:
            x: Input image feature map tensor.
        
        Returns:
            Decoded pose tensor.
        """
        x = x.view(x.size(0), -1)  # flatten the input tensor

        z = self.pose_decoder(x) # torch.Size([8, 19])
        
        bbox_posterior, c_pred = self._decode_pose_to_distribution(z)
        
        if sample_posterior and not self.pose_quantization:
            bbox_pred = bbox_posterior.sample()
        else:
            bbox_pred = bbox_posterior.mode()
        
        c = c_pred
        dec_pose = torch.cat([bbox_pred, c], dim=-1) # torch.Size([8, 19])
        return dec_pose, bbox_posterior
    
    def encode(self, x):
        h = self.encoder(x) # torch.Size([B, F, 16, 16])
        moments_obj = self.quant_conv_obj(h) # torch.Size([B, 32, 16, 16])
        pose_feat = self.quant_conv_pose(h) # torch.Size([B, 16, 16, 16])
        
        q_loss = torch.tensor([0.0]).to(self.device)
        info_obj = (None, None, None)
        info_pose = (None, None, None)
        if self.obj_quantization:
            posterior_obj, q_loss_obj, info_obj = self.quantize_obj(moments_obj) # torch.Size([B, 8, 16, 16])
            q_loss += q_loss_obj
        else:
            posterior_obj = DiagonalGaussianDistribution(moments_obj) # torch.Size([B, 16, 16, 16]) sample
        
        if self.pose_quantization:
            pose_feat, q_loss_pose, info_pose = self.quantize_pose(pose_feat) # torch.Size([B, 8, 16, 16])
            q_loss += q_loss_pose

        return posterior_obj, pose_feat, q_loss, info_obj, info_pose
    
    def _get_dropout_prob(self):
        """
        Get the current dropout probability.
        
        Returns:
            Dropout probability.
        """
        if self.iter_counter < self.encoder_pretrain_steps: # encoder pretraining phase
            # set dropout probability to 1.0
            return self.dropout_prob_init
        elif self.iter_counter < self.intermediate_img_feature_leak_steps + self.encoder_pretrain_steps:
            return 0.95
        elif self.iter_counter < self.intermediate_img_feature_leak_steps + self.encoder_pretrain_steps + self.pose_conditioned_generation_steps: # pose conditioned generation phase
            return self.dropout_prob_init
        elif self.iter_counter < self.intermediate_img_feature_leak_steps + self.dropout_warmup_steps + self.encoder_pretrain_steps + self.pose_conditioned_generation_steps: # pose conditioned generation phase
            # linearly decrease dropout probability from initial to final value
            if self.dropout_prob_init == self.dropout_prob_final or self.dropout_warmup_steps == 0:
                return self.dropout_prob_final
            else:
                return self.dropout_prob_init - (self.dropout_prob_init - self.dropout_prob_final) * (self.iter_counter - self.encoder_pretrain_steps) / self.dropout_warmup_steps # 1.0 - (1.0 - 0.7) * (10000 - 10000) / 10000 = 1.0
        else: # VAE phase
            # set dropout probability to dropout_prob_final
            return self.dropout_prob_final

    def forward(self, input_im, pose_gt, sample_posterior=True, second_pose=None, zoom_mult=None):
        """
        Forward pass of the autoencoder model.
        
        Args:
            input (Tensor): Input tensor to the autoencoder.
            sample_posterior (bool, optional): Whether to sample from the posterior distribution or use the mode.
        
        Returns:
            pred_objdec_obj (Tensor): Decoded object tensor.
            dec_pose (Tensor): Decoded pose tensor.
            posterior_obj (Distribution): Posterior distribution of the object latent space.
            posterior_pose (Distribution): Posterior distribution of the pose latent space.
        """
        # apply_manual_crop = self.apply_conv_crop_img_space or self.apply_conv_crop_latent_space
        # assert apply_manual_crop, "manual crop experiment only in vanilla autoencoder"
        # reshape input_im to (batch_size, 3, 256, 256)
        input_im = input_im.to(memory_format=torch.contiguous_format).float().to(self.device) # torch.Size([4, 3, 256, 256])
        # Encode Image
        posterior_obj, pose_feat, q_loss, (_,_,ind_obj), (_,_,ind_pose) = self.encode(input_im) # Distribution: torch.Size([4, 16, 16, 16]), torch.Size([4, 16, 16, 16])
        
        if self.obj_quantization:
            z_obj = posterior_obj
        else:
            # Sample from posterior distribution or use mode
            if sample_posterior: # True
                z_obj = posterior_obj.sample() # torch.Size([4, 16, 16, 16])
            else:
                z_obj = posterior_obj.mode()
                
        # Extract pose information from pose features
        pred_pose, bbox_posterior = self.decode_pose(pose_feat, sample_posterior=sample_posterior if not self.obj_quantization else False) # torch.Size([4, 8]), torch.Size([4, 7])
        
        # Dropout object feature latents
        self.dropout_prob = self._get_dropout_prob()
        if self.dropout_prob > 0:
            dropout = nn.Dropout(p=self.dropout_prob)
            z_obj = dropout(z_obj)
            
        if self.iter_counter < self.encoder_pretrain_steps: # no reconstruction loss in this phase
            pred_obj = torch.zeros_like(input_im).to(self.device) # torch.Size([4, 3, 256, 256])
                
        # if apply_manual_crop:
        #     assert zoom_mult is not None, "zoom_mult is not specified, aka None"

        # Apply crop in latent space
        # if self.apply_conv_crop_latent_space:
        #     latent_crop_pad_mode = "reflect" if self.latent_manual_crop_reflect_pad else "zero"
        #     z_obj = self.manual_crop(z_obj, zoom_mult, crop_pad_mode=latent_crop_pad_mode)
        
        # Predict images from object and pose latents
        pred_obj = self.decode(z_obj) # torch.Size([4, 3, 256, 256])
        
        # Apply crop in image space
        # if self.apply_conv_crop_img_space:
        #     pred_obj = self.manual_crop(pred_obj, zoom_mult, crop_pad_mode="zero")
            
        return pred_obj, pred_pose, posterior_obj, bbox_posterior, q_loss, ind_obj, ind_pose

    # def batch_center_crop_resize(self, images, H_crops, W_crops, crop_pad_mode="zero"):
    #     assert crop_pad_mode in ["zero", "reflect"], f"crop_pad_mode should be either 'zero' or 'reflect', but got {crop_pad_mode}"
    #     batch_size, _, H, W = images.shape # torch.Size([4, 3, 256, 256])
    #     cropped_resized_images = torch.zeros_like(images) # torch.Size([4, 3, 256, 256])
    #     resizer = T.Resize(size=(H, W),interpolation=T.InterpolationMode.BILINEAR)
    #     for i in range(batch_size):
    #         H_crop = int(H_crops[i])
    #         W_crop = int(W_crops[i])
    #         if crop_pad_mode == "zero":
    #             center_cropper = T.CenterCrop(size=(H_crop, W_crop))
    #         elif crop_pad_mode == "reflect":
    #             center_cropper = ReflectPadCenterCrop(size=(H_crop, W_crop))
    #         cropped_image = center_cropper(images[i].unsqueeze(0))
    #         cropped_resized_images[i] = resizer(cropped_image)

    #     return cropped_resized_images
    
    # def manual_crop(self, images, zoom_mult, crop_pad_mode="zero"):
    #     _, _, H, W = images.shape
        
    #     H_crops = (H * zoom_mult).long()
    #     W_crops = (W * zoom_mult).long()
        
    #     original_crop_recropped_resized = self.batch_center_crop_resize(images, H_crops, W_crops, crop_pad_mode=crop_pad_mode)

    #     return original_crop_recropped_resized

    def get_pose_input(self, batch, k, postfix=""):
        x = batch[k+postfix] 

        if self.train_on_yaw:
            yaw = batch["yaw"+postfix]
            # place yaw at index 3
            x[:, 3] = yaw
        
        x = x.to(memory_format=torch.contiguous_format).float()
        return x
    
    def get_mask_input(self, batch, k):
        if k is None:
            return None
        x = batch[k]
        return x
    
    def get_class_input(self, batch, k):
        x = batch[k]
        return x
    
    def get_bbox_input(self, batch, k):
        x = batch[k]
        return x
        
    def get_fill_factor_input(self, batch, k):
        x = batch[k]
        return x
    
    def get_all_inputs(self, batch, postfix=""):
        # Get RGB GT
        rgb_gt = self.get_input(batch, self.image_rgb_key).permute(0, 2, 3, 1).to(self.device) # torch.Size([4, 3, 256, 256]) 
        rgb_gt = self._rescale(rgb_gt)
        rgb_in = rgb_gt.clone()
        # Get 2D Mask
        mask_2d_bbox = batch["mask_2d_bbox"]
        # Get Pose GT
        pose_gt = self.get_pose_input(batch, self.pose_key).to(self.device) # torch.Size([4, 4]) #
        # Get Segmentation Mask
        segm_mask_gt = self.get_mask_input(batch, self.image_mask_key) # None
        segm_mask_gt = segm_mask_gt.to(self.device) if segm_mask_gt is not None else None
        # Get BBOX GT values
        bbox_gt = self.get_bbox_input(batch, self.bbox_key).to(self.device) # torch.Size([4, 3])
        # Get fill factor in patch
        fill_factor_gt = self.get_fill_factor_input(batch, self.fill_factor_key).to(self.device).float()
        # Get Class GT
        class_gt = self.get_class_input(batch, self.class_key).to(self.device) # torch.Size([4])
        class_gt_label = batch["class_name"]
    
        # torch.Size([4, 3, 256, 256]), torch.Size([4, 8]), torch.Size([4, 16, 16, 16]), torch.Size([4, 7])
        if "pose_6d_2" in batch:
            # Replace RGB and Mask with second patch
            rgb_gt = self.get_input(batch, self.image_rgb_key+"_2").permute(0, 2, 3, 1).to(self.device) # torch.Size([4, 3, 256, 256])
            rgb_gt = self._rescale(rgb_gt)
            mask_2d_bbox = batch["mask_2d_bbox_2"]
            
            # Get respective pose for forward pass
            snd_pose = self.get_pose_input(batch, self.pose_key, postfix="_2").to(self.device) # torch.Size([4, 4]) #
            snd_bbox = self.get_bbox_input(batch, self.bbox_key).to(self.device) # torch.Size([4, 3]) 
            snd_fill = self.get_fill_factor_input(batch, self.fill_factor_key+"_2").to(self.device).float().unsqueeze(1)

            # snd_pose = snd_pose.unsqueeze(0) if snd_pose.dim() == 1 else snd_pose
            # snd_bbox = snd_bbox.unsqueeze(0) if snd_bbox.dim() == 1 else snd_bbox
            # get one hot encoding for class_id - all classes in self.label_id2class_id.values
            num_classes = self.num_classes
            class_probs = torch.nn.functional.one_hot(class_gt, num_classes=num_classes).float()
            second_pose = torch.cat((snd_pose, snd_bbox, snd_fill, class_probs), dim=1)
            # Replace Segmentation Mask
            segm_mask_gt = self.get_mask_input(batch, self.image_mask_key) # None
            segm_mask_gt = segm_mask_gt.to(self.device) if segm_mask_gt is not None else None
        else:
            second_pose = None
            
        return rgb_in, rgb_gt, pose_gt, segm_mask_gt, mask_2d_bbox, class_gt, class_gt_label, bbox_gt, fill_factor_gt, second_pose
    
    def _update_patch_center_rad(self):
        if self.iter_counter >= self.perturb_rad_warmup_steps:
            self.perturb_rad.data = torch.tensor(FINAL_PERTURB_RAD, requires_grad=False)
        else: # linearly decrease perturb_rad from initial value to 0.5 over 1 epoch
            step_size = (FINAL_PERTURB_RAD - self.patch_center_rad_init) / self.perturb_rad_warmup_steps 
            current_rad = self.patch_center_rad_init + (self.iter_counter * step_size)
            self.patch_center_rad.data = torch.tensor(min(current_rad, FINAL_PERTURB_RAD), requires_grad=False)

    def training_step(self, batch, batch_idx, optimizer_idx):
        
        if self.patch_center_rad is not None:
            self._update_patch_center_rad()
            self.log("perturb_rad", self.patch_center_rad, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        
        # Get inputs in right shape
        rgb_in, rgb_gt, pose_gt, segm_mask_gt, mask_2d_bbox, class_gt, class_gt_label, bbox_gt, fill_factor_gt, second_pose \
            = self.get_all_inputs(batch)
        
        # Run full forward pass
        pred_obj, dec_pose, posterior_obj, bbox_posterior, q_loss, ind_obj, ind_pose = self.forward(rgb_in, pose_gt, second_pose=second_pose, zoom_mult=batch["zoom_multiplier_2"])
        
        self.log("dropout_prob", self.dropout_prob, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.iter_counter += 1
        # Train the autoencoder
        if optimizer_idx == 0:
            # train encoder+decoder+logvar # last layer: torch.Size([3, 128, 3, 3])
            aeloss, log_dict_ae = self.loss(rgb_gt, segm_mask_gt, pose_gt,
                                            pred_obj, dec_pose,
                                            class_gt, class_gt_label, bbox_gt, fill_factor_gt,
                                            posterior_obj, bbox_posterior, optimizer_idx, self.iter_counter, mask_2d_bbox,
                                            last_layer=self.get_last_layer(), split="train",
                                            q_loss=q_loss, predicted_indices_obj=ind_obj, predicted_indices_pose=ind_pose)
                                                    
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            
            if torch.isnan(aeloss):
                return None

            return aeloss
        # Train the discriminator
        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(rgb_gt, segm_mask_gt, pose_gt,
                                                pred_obj, dec_pose,
                                                class_gt, class_gt_label, bbox_gt, fill_factor_gt,
                                                posterior_obj, bbox_posterior, optimizer_idx, self.iter_counter, mask_2d_bbox,
                                                last_layer=self.get_last_layer(), split="train", q_loss=q_loss)
            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            
            if torch.isnan(discloss):
                return None

            return discloss
    
    def validation_step(self, batch, batch_idx):
        
        # Get inputs in right shape
        rgb_in, rgb_gt, pose_gt, segm_mask_gt, mask_2d_bbox, class_gt, class_gt_label, bbox_gt, fill_factor_gt, second_pose \
            = self.get_all_inputs(batch)
        
        # Run full forward pass
        pred_obj, dec_pose, posterior_obj, bbox_posterior, q_loss, ind_obj, ind_pose = self.forward(rgb_in, pose_gt, second_pose=second_pose, zoom_mult=batch["zoom_multiplier_2"])
        
        _, log_dict_ae = self.loss(rgb_gt, segm_mask_gt, pose_gt,
                                    pred_obj, dec_pose,
                                    class_gt, class_gt_label, bbox_gt,fill_factor_gt,
                                    posterior_obj, bbox_posterior, 0, 
                                    global_step=self.iter_counter, mask_2d_bbox=mask_2d_bbox,
                                    last_layer=self.get_last_layer(), split="val",
                                    q_loss=q_loss, predicted_indices_obj=ind_obj, predicted_indices_pose=ind_pose)

        _, log_dict_disc = self.loss(rgb_gt, segm_mask_gt, pose_gt,
                                        pred_obj, dec_pose,
                                        class_gt, class_gt_label, bbox_gt,fill_factor_gt,
                                        posterior_obj, bbox_posterior, 1, 
                                        global_step=self.iter_counter, mask_2d_bbox=mask_2d_bbox,
                                        last_layer=self.get_last_layer(), split="val", 
                                        q_loss=q_loss, predicted_indices_obj=ind_obj, predicted_indices_pose=ind_pose)

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"], sync_dist=True)
        del log_dict_ae["val/rec_loss"]
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict
    
    def test_step(self, batch, batch_idx):        
        # Prepare Input
        input_patches = batch[self.image_rgb_key].to(self.device) # torch.Size([B, 3, 256, 256])
        assert input_patches.dim() == 4 or (input_patches.dim() == 5 and input_patches.shape[0] == 1), f"Only supporting batch size 1. Input_patches shape: {input_patches.shape}"
        if input_patches.dim() == 5:
            input_patches = input_patches.squeeze(0) # torch.Size([B, 3, 256, 256])
        input_patches = self._rescale(input_patches) # torch.Size([B, 3, 256, 256])
        
        # Chunked dec_pose[..., :POSE_6D_DIM + LHW_DIM + FILL_FACTOR_DIM]
        all_objects = []
        all_poses = []
        all_patch_indeces = []
        all_scores = []
        all_classes = []
        chunk_size = self.chunk_size
        with torch.no_grad():
            for i in range(0, len(input_patches), chunk_size):
                selected_patches = input_patches[i:i+chunk_size]
                global_patch_index = i + torch.arange(chunk_size)[:len(selected_patches)]
                selected_patch_indeces, z_obj, dec_pose, score, class_idx = self._get_valid_patches(selected_patches, global_patch_index)
                all_patch_indeces.append(selected_patch_indeces)
                all_objects.append(z_obj)
                all_poses.append(dec_pose)
                all_scores.append(score)
                all_classes.append(class_idx)
                
        scores = torch.cat(all_scores)
                    
        # Inference refinement
        all_patch_indeces = torch.cat(all_patch_indeces)
        if not len(all_patch_indeces):
            return torch.empty(0)
        all_z_objects = torch.cat(all_objects)
        all_z_poses = torch.cat(all_poses)
        patches_w_objs = input_patches[all_patch_indeces]
        # TODO: save everything to an output file!
        return self._refinement_step(patches_w_objs, all_z_objects, all_z_poses)
    
    def _get_valid_patches(self, input_patches, global_patch_index):
        local_patch_idx = torch.arange(len(global_patch_index))
        # Run encoder
        posterior_obj, pose_feat, _, (_,_,_), (_,_,_) = self.encode(input_patches)
        z_obj = posterior_obj.mode()
        dec_pose, _  = self.decode_pose(pose_feat, sample_posterior=False)
        
        # Threshold by class probabilities
        # TODO: Check class probabilities after sofmax
        class_pred = dec_pose[:, -self.num_classes:]
        class_prob = torch.softmax(class_pred, dim=-1)
        # obj_prob = class_prob[:, -self.num_classes:]
        score, class_idx = torch.max(class_prob, -1)
        class_thresh_mask = ((score > self.class_thresh) # Exclude unlikely patches
                                # & (bckg_prob < self.class_thresh).squeeze(-1) 
                                & ~(class_idx == self.num_classes-1) # Exclude background class
                                )
        if not class_thresh_mask.sum():
            return self.return_empty(global_patch_index, z_obj, dec_pose, score, class_idx)
        local_patch_idx = local_patch_idx[class_thresh_mask]
        
        # Threshold by fill factor
        fill_factor = dec_pose[:, POSE_6D_DIM + LHW_DIM : POSE_6D_DIM + LHW_DIM + FILL_FACTOR_DIM][local_patch_idx].squeeze(-1)
        fill_factor_mask = fill_factor > self.fill_factor_thresh # TODO: Check negative or positive fill factor
        if not fill_factor_mask.sum():
            return self.return_empty(global_patch_index, z_obj, dec_pose, score, class_idx)
        local_patch_idx = local_patch_idx[fill_factor_mask]
        
        # TODO: Finish when transformations are available
        # # NMS on the BEV plane
        # bbox_enc = dec_pose[:, POSE_6D_DIM + LHW_DIM][local_patch_idx]
        # # TODO: Transform the box to 3D
        # bbox3D = transform_to_3D(bbox_enc)
        # # TODO: Only WL and XY on BEV + reshape to (N, 4) box
        # bbox_BEV = transform_to_BEV(bbox3D)
        # class_scores = class_prob[local_patch_idx]
        # class_prediction = torch.argmax(class_scores, dim=-1)
        
        # nms_box_mask = batched_nms(boxes=bbox_BEV, scores=class_scores, idxs=class_prediction, iou_threshold=0.5)
        # local_patch_idx = local_patch_idx[nms_box_mask]       
        
        
        return global_patch_index[local_patch_idx], z_obj[local_patch_idx], dec_pose[local_patch_idx], score[local_patch_idx], class_idx[local_patch_idx]
    
    def return_empty(self, global_patch_index, z_obj, dec_pose, score, class_idx):
        return torch.empty_like(global_patch_index[:0]), torch.empty_like(z_obj[:0]), torch.empty_like(dec_pose[:0]), torch.empty_like(score[:0]), torch.empty_like(class_idx[:0])

    @torch.enable_grad()
    def _refinement_step(self, input_patches, z_obj, z_pose, gt_x=None, gt_y=None):
        # Initialize optimizer and parameters
        refined_pose = z_pose[:, :-self.num_classes]
        obj_class = z_pose[:, -self.num_classes:]
        if True: # TODO: for debug only
            if gt_x is not None and gt_y is not None:
                refined_pose_chosen = torch.zeros_like(refined_pose[:, :2])
                refined_pose_chosen[:, 0] = gt_x
                refined_pose_chosen[:, 1] = gt_y
            else:
                refined_pose_chosen = refined_pose[:, :2]
            
            refined_pose_not_chosen = refined_pose[:, 2:]
            refined_pose_param = nn.Parameter(refined_pose_chosen, requires_grad=True)
        else:
            refined_pose_chosen = refined_pose
            refined_pose_param = nn.Parameter(refined_pose, requires_grad=True)
        optim_refined = self._init_refinement_optimizer(refined_pose_param, lr=self.ref_lr)
        # Run K iter refinement steps
        for k in range(self.num_refinement_steps):
            dec_pose = torch.cat([refined_pose_param, obj_class], dim=-1)
            optim_refined.zero_grad()
            gen_image = self.decode(z_obj)
            rec_loss = self.loss._get_rec_loss(input_patches, gen_image, use_pixel_loss=True).mean()
            rec_loss.backward()
            optim_refined.step()
        if True: # TODO: for debug only 
            refined_pose = torch.cat([refined_pose_chosen, refined_pose_not_chosen], dim=-1)
        dec_pose = torch.cat([refined_pose, obj_class], dim=-1)   
        return dec_pose.data
    
    def _init_refinement_optimizer(self, pose, lr=1e-3):
        return torch.optim.AdamW([pose], lr=lr)
    
    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae_params = list(self.encoder.parameters())+ \
                        list(self.decoder.parameters())+ \
                        list(self.quant_conv_obj.parameters())+ \
                        list(self.quant_conv_pose.parameters())+ \
                        list(self.post_quant_conv.parameters())+ \
                        list(self.pose_decoder.parameters())
        if hasattr(self, 'quantize_obj'):
            opt_ae_params = opt_ae_params + list(self.quantize_obj.parameters())
        
        if hasattr(self, 'quantize_pose'):
            opt_ae_params = opt_ae_params + list(self.quantize_pose.parameters())

        opt_ae = torch.optim.Adam(opt_ae_params,
                                lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))

        # if False: # TODO: make optional
        #     opt_ae.param_groups[0]['capturable'] = True
        #     opt_disc.param_groups[0]['capturable'] = True
        return [opt_ae, opt_disc], []
    
    def _log_reconstructions(self, rgb_in, pose_gt, rgb_in_viz, second_pose, log, namespace, zoom_mult_1, zoom_mult_2):
        # torch.Size([4, 3, 256, 256]) torch.Size([4, 8]) torch.Size([4, 16, 16, 16]) torch.Size([4, 7])
        # Run full forward pass

        xrec_2, _, _, _, _, _, _ = self.forward(rgb_in, pose_gt, second_pose=second_pose, zoom_mult=zoom_mult_2)
        xrec, _, _, _, _, _, _ = self.forward(rgb_in, pose_gt, zoom_mult=zoom_mult_1)
        
        xrec_rgb = xrec[:, :3, :, :] # torch.Size([8, 3, 64, 64])
        xrec_rgb_2 = xrec_2[:, :3, :, :] # torch.Size([8, 3, 64, 64])
        
        if rgb_in_viz.shape[1] > 3:
            # colorize with random projection
            assert xrec_rgb.shape[1] > 3
            rgb_in_viz = self.to_rgb(rgb_in_viz)
            rgb_gt_viz = self.to_rgb(rgb_gt_viz)
            xrec_rgb = self.to_rgb(xrec_rgb)
            xrec_rgb_2 = self.to_rgb(xrec_rgb_2)
            
        # scale is 0, 1. scale to -1, 1
        log[f"reconstructions_rgb{namespace}"] = xrec_rgb.clone().detach()
        log[f"reconstructions_rgb_2{namespace}"] = xrec_rgb_2.clone().detach()


        return log
    
    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()

        rgb_in, rgb_gt, pose_gt, segm_mask_gt, mask_2d_bbox, class_gt, class_gt_label, bbox_gt, fill_factor_gt, second_pose = self.get_all_inputs(batch)
        
        rgb_in_viz = self._rescale(rgb_in)
        rgb_gt_viz = self._rescale(rgb_gt)
        
        log["inputs_rgb_in"] = rgb_in_viz.clone().detach()
        log["inputs_rgb_gt"] = rgb_gt_viz.clone().detach()

        if not only_inputs:
            log = self._log_reconstructions(rgb_in, pose_gt, rgb_in_viz, second_pose, log, namespace="", zoom_mult_1=batch["zoom_multiplier"], zoom_mult_2=batch["zoom_multiplier_2"])

            # EMA weights visualization
            with self.ema_scope():
                log = self._log_reconstructions(rgb_in, pose_gt, rgb_in_viz, second_pose, log, namespace="_ema", zoom_mult_1=batch["zoom_multiplier"], zoom_mult_2=batch["zoom_multiplier_2"])
        return log
    
    def _rescale(self, x):
        # scale is -1
        return 2. * (x - x.min()) / (x.max() - x.min()) - 1.
    
    def to_rgb(self, x):
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x

    def _get_tv_loss(self, x, reduction='sum', weight=1e0):
        return total_variation(x, reduction=reduction) * weight

class AdaptivePoseAutoencoder(PoseAutoencoder):
    def __init__(self,
                ddconfig,
                lossconfig,
                embed_dim,
                ckpt_path=None,
                ignore_keys=[],
                image_mask_key=None,
                image_rgb_key="patch",
                pose_key="pose_6d",
                fill_factor_key="fill_factor",
                pose_perturbed_key="pose_6d_perturbed",
                class_key="class_id",
                bbox_key="bbox_sizes",
                colorize_nlabels=None,
                monitor=None,
                pose_decoder_config=None,
                dropout_prob_init=1.0,
                dropout_prob_final=0.7,
                dropout_warmup_steps=5000,
                pose_conditioned_generation_steps=10000,
                perturb_rad_warmup_steps=100000,
                intermediate_img_feature_leak_steps=0,
                add_noise_to_z_obj=False,
                train_on_yaw=True,
                ema_decay=0.999,
                quantconfig=None,
                quantize_obj=False,
                quantize_pose=False,
                decoder_mid_adaptive=True,
                **kwargs
                ):
        pl.LightningModule.__init__(self)
        
        # Inputs
        self.image_rgb_key = image_rgb_key
        self.pose_key = pose_key
        self.pose_perturbed_key = pose_perturbed_key
        self.class_key = class_key
        self.bbox_key = bbox_key
        self.fill_factor_key = fill_factor_key
        self.image_mask_key = image_mask_key
        self.train_on_yaw = train_on_yaw
        
        # Encoder Setup
        self.encoder = FeatEncoder(**ddconfig)
        self.obj_quantization = quantize_obj
        self.pose_quantization = quantize_pose
        
        quant_conv_obj_embed_dim = embed_dim if quantize_obj else 2*embed_dim
        
        if quantize_obj:
            assert quantconfig is not None, "quantconfig is not defined but quantize_obj is set to True."
            quantconfig["params"]["e_dim"] = embed_dim
            self.quantize_obj = instantiate_from_config(quantconfig)
        
        if quantize_pose:
            assert quantconfig is not None, "quantconfig is not defined but quantize_pose is set to True."
            quantconfig["params"]["e_dim"] = embed_dim
            self.quantize_pose = instantiate_from_config(quantconfig)

        mult_z = 2 if ddconfig["double_z"] else 1
            
        self.quant_conv_obj = torch.nn.Conv2d(mult_z*ddconfig["z_channels"], quant_conv_obj_embed_dim, 1)
        self.quant_conv_pose = torch.nn.Conv2d(mult_z*ddconfig["z_channels"], embed_dim, 1) # TODO: Need to fix the dimensions
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if quantconfig is not None:
            self.quantization = True
            self.quantize = instantiate_from_config(quantconfig)
        else:
            self.quantization = False
            
        self.encoder_pretrain_steps = lossconfig["params"]["encoder_pretrain_steps"]
        
        # Object latent
        spatial_dim = (np.array([1/k for k in ddconfig["ch_mult"]]).prod() * 256).astype(int)
        self.add_noise_to_z_obj = add_noise_to_z_obj
        self.feat_dims = [embed_dim, spatial_dim, spatial_dim]
        
        self.z_channels = ddconfig["z_channels"]
        
        pose_decoder_config["params"]["num_channels"] = ddconfig["z_channels"]
        
        # Pose prediction and latent
        self.pose_decoder = instantiate_from_config(pose_decoder_config)
        
        # Decoder Setup
        decoder_cfg = ddconfig.copy()
        decoder_cfg["mid_adaptive"] = decoder_mid_adaptive
        decoder_cfg["num_classes"] = lossconfig["params"]["num_classes"]
        self.decoder = AdaptiveFeatDecoder(**decoder_cfg)
        
        # Dropout Setup
        self.dropout_prob_final = dropout_prob_final # 0.7
        self.dropout_prob_init = dropout_prob_init # 1.0
        self.dropout_prob = self.dropout_prob_init
        self.dropout_warmup_steps = dropout_warmup_steps # 10000 (after stage 1: encoder pretraining)
        self.pose_conditioned_generation_steps = pose_conditioned_generation_steps # 10000
        self.intermediate_img_feature_leak_steps = intermediate_img_feature_leak_steps

        # Loss setup
        lossconfig["params"]["pose_conditioned_generation_steps"] = pose_conditioned_generation_steps
        lossconfig["params"]["train_on_yaw"] = self.train_on_yaw
        if not quantize_obj and not quantize_pose:
            lossconfig["params"]["codebook_weight"] = 0.0
        self.loss = instantiate_from_config(lossconfig)
        self.num_classes = lossconfig["params"]["num_classes"]
        
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
            
        self.perturb_rad_warmup_steps = perturb_rad_warmup_steps
        
        # Checkpointing
        self.use_ema = ema_decay is not None
        if self.use_ema:
            self.ema_decay = ema_decay
            assert 0. < ema_decay < 1.
            self.model_ema = LitEma(self, decay=ema_decay)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")
        
        if ckpt_path is not None:
            try:
                self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
            except Exception as e:
                # add optimizer to ignore_keys list
                ignore_keys = ignore_keys + ["optimizer"]
                self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.iter_counter = 0

        ### Placeholders
        self.patch_center_rad = None
        self.patch_center_rad_init = None
        
    
    def decode(self, z, pose):
        z = self.post_quant_conv(z)
        return self.decoder(z, pose)

    def forward(self, input_im, pose_gt, sample_posterior=True, second_pose=None, zoom_mult=None):
        """
        Forward pass of the autoencoder model.
        
        Args:
            input (Tensor): Input tensor to the autoencoder.
            sample_posterior (bool, optional): Whether to sample from the posterior distribution or use the mode.
        
        Returns:
            pred_objdec_obj (Tensor): Decoded object tensor.
            dec_pose (Tensor): Decoded pose tensor.
            posterior_obj (Distribution): Posterior distribution of the object latent space.
            posterior_pose (Distribution): Posterior distribution of the pose latent space.
        """
        # reshape input_im to (batch_size, 3, 256, 256)
        input_im = input_im.to(memory_format=torch.contiguous_format).float().to(self.device) # torch.Size([4, 3, 256, 256])
        # Encode Image
        posterior_obj, pose_feat, q_loss, (_,_,ind_obj), (_,_,ind_pose) = self.encode(input_im) # Distribution: torch.Size([4, 16, 16, 16]), torch.Size([4, 16, 16, 16])
        
        if self.obj_quantization:
            z_obj = posterior_obj
        else:
            # Sample from posterior distribution or use mode
            z_obj = posterior_obj.sample() if sample_posterior else posterior_obj.mode()
                
        # Extract pose information from pose features
        pred_pose, bbox_posterior = self.decode_pose(pose_feat, sample_posterior=False if self.obj_quantization else sample_posterior) # torch.Size([4, 8]), torch.Size([4, 7])
        
        # Dropout object feature latents
        self.dropout_prob = self._get_dropout_prob()
        if self.dropout_prob > 0:
            dropout = nn.Dropout(p=self.dropout_prob)
            z_obj = dropout(z_obj)
            
        if self.iter_counter < self.encoder_pretrain_steps: # no reconstruction loss in this phase
            pred_obj = torch.zeros_like(input_im).to(self.device) # torch.Size([4, 3, 256, 256])
            
        # Replace pose with other pose if supervised with other patch
        gen_pose = pred_pose if second_pose is None else second_pose.to(pred_pose)

        # Predict images from object and pose latents
        pred_obj = self.decode(z_obj, gen_pose) # torch.Size([4, 3, 256, 256])             
        return pred_obj, pred_pose, posterior_obj, bbox_posterior, q_loss, ind_obj, ind_pose

    @torch.enable_grad()
    def _refinement_step(self, input_patches, z_obj, z_pose, gt_x=None, gt_y=None):
        # Initialize optimizer and parameters
        refined_pose = z_pose[:, :-self.num_classes]
        obj_class = z_pose[:, -self.num_classes:]
        if True: # TODO: for debug only
            if gt_x is not None and gt_y is not None:
                refined_pose_chosen = torch.zeros_like(refined_pose[:, :2])
                refined_pose_chosen[:, 0] = min(gt_x + 0.3, 1.0).reshape_as(refined_pose[:, 0])
                refined_pose_chosen[:, 1] = min(gt_y + 0.3, 1.0).reshape_as(refined_pose[:, 1])
            else:
                refined_pose_chosen = refined_pose[:, :2]
            
            refined_pose_not_chosen = refined_pose[:, 2:]
            refined_pose_param = nn.Parameter(refined_pose_chosen, requires_grad=True)
        else:
            refined_pose_chosen = refined_pose
            refined_pose_param = nn.Parameter(refined_pose_chosen, requires_grad=True)
        optim_refined = self._init_refinement_optimizer(refined_pose_param, lr=self.ref_lr)

        x_list = torch.zeros(self.num_refinement_steps)
        y_list = torch.zeros(self.num_refinement_steps)
        grad_x_list = torch.zeros(self.num_refinement_steps)
        grad_y_list = torch.zeros(self.num_refinement_steps)
        loss_list = torch.zeros(self.num_refinement_steps)
        tv_loss_list = torch.zeros(self.num_refinement_steps)

        # Run K iter refinement steps
        for k in range(self.num_refinement_steps):
            if True: # TODO: for debug only
                refined_pose_param_with_rest = torch.cat([refined_pose_param, refined_pose_not_chosen], dim=-1)
                dec_pose = torch.cat([refined_pose_param_with_rest, obj_class], dim=-1)
            else:
                dec_pose = torch.cat([refined_pose_param, obj_class], dim=-1)
            x_list[k] = refined_pose_param[:, 0].clone().squeeze()
            y_list[k] = refined_pose_param[:, 1].clone().squeeze()
            optim_refined.zero_grad()
            gen_pose = dec_pose
            gen_image = self.decode(z_obj, gen_pose)
            rec_loss = self.loss._get_rec_loss(input_patches, gen_image, use_pixel_loss=True).mean()
            tv_loss = self._get_tv_loss(gen_image, weight=self.tv_loss_weight)
            refinement_loss = rec_loss + tv_loss
            refinement_loss.backward()
            optim_refined.step()
            
            grad_x_list[k] = refined_pose_param.grad[:, 0].clone().squeeze()
            grad_y_list[k] = refined_pose_param.grad[:, 1].clone().squeeze()
            loss_list[k] = refinement_loss.clone().squeeze()
            tv_loss_list[k] = tv_loss.clone().squeeze().detach()
        
        if True: # TODO: for debug only 
            refined_pose = torch.cat([refined_pose_param, refined_pose_not_chosen], dim=-1)
        dec_pose = torch.cat([refined_pose, obj_class], dim=-1)   
        return dec_pose.data, gen_image, x_list, y_list, grad_x_list, grad_y_list, loss_list, tv_loss_list
