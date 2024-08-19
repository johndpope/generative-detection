# src/models/autoencoder.py
import math
import random
from math import radians
import numpy as np
import logging
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.normal import Normal
from torchvision.ops import batched_nms, nms
import pytorch_lightning as pl
import torchvision.transforms as T

from ldm.models.autoencoder import AutoencoderKL
from ldm.util import instantiate_from_config
from ldm.modules.ema import LitEma

from src.modules.autoencodermodules.feat_encoder import FeatEncoder
from src.modules.autoencodermodules.feat_decoder import FeatDecoder, AdaptiveFeatDecoder
from src.modules.autoencodermodules.pose_encoder import PoseEncoderSpatialVAE as PoseEncoder
from src.modules.autoencodermodules.pose_decoder import PoseDecoderSpatialVAE as PoseDecoder
from src.util.distributions import DiagonalGaussianDistribution
from src.util.misc import flip_tensor    
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
                 pose_encoder_config=None,
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
                 apply_convolutional_shift_img_space=False,
                 apply_convolutional_shift_latent_space=False,
                 quantconfig=None,
                 quantize_obj=False,
                 quantize_pose=False,
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

        if ddconfig["double_z"]:
            mult_z = 2
        else:
            mult_z = 1
            
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
        
        pose_encoder_config["params"]["num_channels"] = ddconfig["z_channels"]
        pose_decoder_config["params"]["num_channels"] = ddconfig["z_channels"]
        
        # Pose prediction and latent
        self.pose_decoder = instantiate_from_config(pose_decoder_config)
        self.pose_encoder = instantiate_from_config(pose_encoder_config)
        
        # Decoder Setup
        assert ~(apply_convolutional_shift_img_space and apply_convolutional_shift_latent_space), "Only one of the shift types (image or latent space) can be applied"
        self.apply_conv_shift_img_space = apply_convolutional_shift_img_space
        self.apply_conv_shift_latent_space = apply_convolutional_shift_latent_space
        self.apply_conv_crop_img_space = apply_conv_crop_img_space
        self.apply_conv_crop_latent_space = apply_conv_crop_latent_space
        assert not (self.apply_conv_crop_img_space and self.apply_conv_crop_latent_space), "Only one of the crop types (image or latent space) can be applied"
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
            except Exception as e:
                # add optimizer to ignore_keys list
                ignore_keys = ignore_keys + ["optimizer"]
                self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.iter_counter = 0

        ### Placeholders
        self.patch_center_rad = None
        self.patch_center_rad_init = None

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
        
    def _get_enc_feat_dims(self, ddconfig):
        """ pass in dummy input of size from config to get the output size of encoder and quant_conv """
        # multiply all feat dims
        return self.feat_dims[0] * self.feat_dims[1] * self.feat_dims[2]
    
    def _draw_samples(self, z_mu, z_logstd):
        
        z_std = torch.exp(z_logstd)
        z_dim = z_mu.size(1)
        
        # draw samples from variational posterior to calculate
        # E[p(x|z)]
        r = Variable(x.data.new(b,z_dim).normal_())
        z = z_std*r + z_mu
        
        return z
    
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
    
    def encode_pose(self, x):
        """
        Encode the pose to get the pose feature map from the given pose tensor.
        
        Args:
            x: Input pose tensor.
        
        Returns:
            Encoded pose feature map tensor.
        """
        # x: torch.Size([4, 8])
        flattened_encoded_pose_feat_map = self.pose_encoder(x) # torch.Size([4, 4096]) = 4, 16*16*16     
        return flattened_encoded_pose_feat_map.view(flattened_encoded_pose_feat_map.size(0), self.z_channels, self.feat_dims[1], self.feat_dims[2])
    
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
            dropout_prob = self.dropout_prob_init
        elif self.iter_counter < self.intermediate_img_feature_leak_steps + self.encoder_pretrain_steps:
            dropout_prob = 0.95
        
        elif self.iter_counter < self.intermediate_img_feature_leak_steps + self.encoder_pretrain_steps + self.pose_conditioned_generation_steps: # pose conditioned generation phase
            dropout_prob = self.dropout_prob_init
        
        elif self.iter_counter < self.intermediate_img_feature_leak_steps + self.dropout_warmup_steps + self.encoder_pretrain_steps + self.pose_conditioned_generation_steps: # pose conditioned generation phase
            # linearly decrease dropout probability from initial to final value
            if self.dropout_prob_init == self.dropout_prob_final or self.dropout_warmup_steps == 0:
                dropout_prob = self.dropout_prob_final
            else:
                dropout_prob = self.dropout_prob_init - (self.dropout_prob_init - self.dropout_prob_final) * (self.iter_counter - self.encoder_pretrain_steps) / self.dropout_warmup_steps # 1.0 - (1.0 - 0.7) * (10000 - 10000) / 10000 = 1.0
        
        else: # VAE phase
            # set dropout probability to dropout_prob_final
            dropout_prob = self.dropout_prob_final
        
        return dropout_prob

    def apply_manual_shift(self, images, shifts_x, shifts_y):
        batch_size, channels, height, width = images.shape
        
        shifts_x_pixels = (shifts_x * (width // 2)).int()
        shifts_y_pixels = (shifts_y * (height // 2)).int()

        # Create a mesh grid
        grid_y, grid_x = torch.meshgrid(torch.arange(height), torch.arange(width))
        grid_x = grid_x.unsqueeze(0).expand(batch_size, -1, -1).float().to(shifts_x_pixels.device)  # [batch_size, height, width]
        grid_y = grid_y.unsqueeze(0).expand(batch_size, -1, -1).float().to(shifts_y_pixels.device)  # [batch_size, height, width]

        # Normalize the grid to [-1, 1]
        grid_x = 2.0 * grid_x / (width - 1) - 1.0
        grid_y = 2.0 * grid_y / (height - 1) - 1.0

        # Apply the shifts
        grid_x = grid_x - (shifts_x_pixels.view(-1, 1, 1) * 2.0 / (width - 1))
        grid_y = grid_y - (shifts_y_pixels.view(-1, 1, 1) * 2.0 / (height - 1))

        # Stack grids and reshape to [batch_size, height, width, 2]
        grid = torch.stack((grid_x, grid_y), dim=-1)

        # Sample the images using the computed grid
        shifted_images = F.grid_sample(images, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

        return shifted_images

    
    def apply_convolutional_shift(self, images, shifts_x, shifts_y):
        batch_size, channels, height, width = images.shape
        
        shifts_x_pixels = (shifts_x * (width // 2)).int()
        shifts_y_pixels = (shifts_y * (height // 2)).int()

        # Create the kernels
        kernels = torch.zeros((batch_size, channels, height, width)).to(images.device)

        for i in range(batch_size):
            shift_x = shifts_x_pixels[i].item()
            shift_y = shifts_y_pixels[i].item()
            
            kernels[i, :, (height // 2) + shift_y, (width // 2) + shift_x] = 1

        # flip kernel
        kernels_flipped = flip_tensor(kernels)
        # Perform the convolution - apply convolution to each channel separately
        images_expanded = images.reshape(1, batch_size * channels, height, width)
        kernels_expanded = kernels_flipped.view(batch_size * channels, 1, height, width)
        shifted_images = F.conv2d(
            input=images_expanded, 
            weight=kernels_expanded, 
            stride=1, 
            padding='same', 
            groups=batch_size * channels).view(batch_size, channels, height, width)

        return shifted_images

    def forward(self, input_im, pose_gt, sample_posterior=True, second_pose=None, return_pred_indices=False):
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
        apply_manual_shift = self.apply_conv_shift_img_space or self.apply_conv_shift_latent_space
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
        
        # Add gaussian noise to object feature latents
        # if self.add_noise_to_z_obj:
        #     # draw from standard normal distribution
        #     std_normal = Normal(0, 1)
        #     z_obj_noise = std_normal.sample(posterior_obj.mean.shape).to(self.device) # torch.Size([4, 16, 16, 16])
        #     z_obj = z_obj + z_obj_noise
            
        if not apply_manual_shift:
             # Replace pose with other pose if supervised with other patch
            if second_pose is not None:
                gen_pose = second_pose.to(pred_pose)
            else:
                gen_pose = pred_pose
            
            # Run pose encoder layers  
            z_pose = self.encode_pose(gen_pose) # torch.Size([B, 16, 16, 16])

            assert z_obj.shape == z_pose.shape, f"z_obj shape: {z_obj.shape}, z_pose shape: {z_pose.shape}"
            
            # Add object and pose latents
            z_obj_pose = z_obj + z_pose # torch.Size([B, 16, 16, 16])
        else:
            z_obj_pose = z_obj
            # Compute shift if shift between both patches
            if second_pose is not None:
                shift_x = second_pose[:, 0] - pose_gt[:, 0]
                shift_y = second_pose[:, 1] - pose_gt[:, 1]
                d_shift = shift_x.abs().sum() + shift_y.abs().sum() # Do not apply shift layer if shift is 0
            else:
                shift_x, shift_y = torch.zeros_like(pose_gt[:, 0]), torch.zeros_like(pose_gt[:, 0])
                d_shift = torch.tensor([0.0])
        
        # Apply shift in latent space
        if self.apply_conv_shift_latent_space and d_shift:
            z_obj_pose = self.apply_manual_shift(z_obj_pose, shift_x, shift_y)  
        
        # Predict images from object and pose latents
        pred_obj = self.decode(z_obj_pose) # torch.Size([4, 3, 256, 256])
        
        # Apply shift in image space
        if self.apply_conv_shift_img_space and not self.apply_conv_shift_latent_space and d_shift:
            pred_obj = self.apply_manual_shift(pred_obj, shift_x, shift_y)
            
        return pred_obj, pred_pose, posterior_obj, bbox_posterior, q_loss, ind_obj, ind_pose

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
    
    def _get_perturbed_pose(self, batch, k):
        x = batch[k].squeeze(1) # torch.Size([1, 4])
        if self.train_on_yaw:
            x = torch.zeros_like(x) # torch.Size([4, 6])
            x[:, -1] = batch["yaw_perturbed"] # torch.Size([4])
        return x # torch.Size([1, 4])
    
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

        zoom_mult = batch["zoom_multiplier"]

            
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
            zoom_mult = batch["zoom_multiplier_2"]
        else:
            second_pose = None
            
        return rgb_in, rgb_gt, pose_gt, segm_mask_gt, mask_2d_bbox, class_gt, class_gt_label, bbox_gt, fill_factor_gt, second_pose, zoom_mult
    
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
        rgb_in, rgb_gt, pose_gt, segm_mask_gt, mask_2d_bbox, class_gt, class_gt_label, bbox_gt, fill_factor_gt, second_pose, zoom_mult \
            = self.get_all_inputs(batch)
        
        # Run full forward pass
        pred_obj, dec_pose, posterior_obj, bbox_posterior, q_loss, ind_obj, ind_pose = self.forward(rgb_in, pose_gt, second_pose=second_pose, return_pred_indices=True, zoom_mult=zoom_mult)
        
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
        rgb_in, rgb_gt, pose_gt, segm_mask_gt, mask_2d_bbox, class_gt, class_gt_label, bbox_gt, fill_factor_gt, second_pose, zoom_mult \
            = self.get_all_inputs(batch)
        
        # Run full forward pass
        pred_obj, dec_pose, posterior_obj, bbox_posterior, q_loss, ind_obj, ind_pose = self.forward(rgb_in, pose_gt, second_pose=second_pose, return_pred_indices=True, zoom_mult=zoom_mult)
        
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
        # TODO: Move to init
        self.chunk_size = 128
        self.class_thresh = 0.3 # TODO: Set to 0.5 or other value that works on val set
        self.fill_factor_thresh = 0.5 # TODO: Set to 0.5 or other value that works on val set
        self.num_refinement_steps = 10
        self.ref_lr=1.0e0
        
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
        dec_pose_refined = self._refinement_step(patches_w_objs, all_z_objects, all_z_poses)
        
        # TODO: save everything to an output file!
        return dec_pose_refined
    
    def _get_valid_patches(self, input_patches, global_patch_index):
        local_patch_idx = torch.arange(len(global_patch_index))
        # Run encoder
        posterior_obj, pose_feat, q_loss, (_,_,ind_obj), (_,_,ind_pose) = self.encode(input_patches)
        z_obj = posterior_obj.mode()
        dec_pose, bbox_posterior  = self.decode_pose(pose_feat, sample_posterior=False)
        
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
    def _refinement_step(self, input_patches, z_obj, z_pose):
        # Initialize optimizer and parameters
        refined_pose = z_pose[:, :-self.num_classes]
        obj_class = z_pose[:, -self.num_classes:]
        refined_pose_param = nn.Parameter(refined_pose, requires_grad=True)
        optim_refined = self._init_refinement_optimizer(refined_pose_param, lr=self.ref_lr)
        # Run K iter refinement steps
        for k in range(self.num_refinement_steps):
            dec_pose = torch.cat([refined_pose_param, obj_class], dim=-1)
            optim_refined.zero_grad()
            enc_pose = self.encode_pose(dec_pose)
            z_obj_pose = z_obj + enc_pose
            gen_image = self.decode(z_obj_pose)
            rec_loss = self.loss._get_rec_loss(input_patches, gen_image, use_pixel_loss=True).mean()
            rec_loss.backward()
            optim_refined.step()
            print("Gradient: ", refined_pose_param.grad.mean(), refined_pose_param.grad.std())
            print("Poses: ", refined_pose_param.mean(), refined_pose_param.std())
        
        dec_pose = torch.cat([refined_pose, obj_class], dim=-1)   
        return dec_pose.data
    
    def _init_refinement_optimizer(self, pose, lr=1e-3):
        return torch.optim.Adam([pose], lr=lr)
    
    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae_params = list(self.encoder.parameters())+ \
                                  list(self.decoder.parameters())+ \
                                  list(self.quant_conv_obj.parameters())+ \
                                  list(self.quant_conv_pose.parameters())+ \
                                  list(self.post_quant_conv.parameters())+ \
                                  list(self.pose_encoder.parameters())+ \
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
    
    def _perturb_poses(self, batch, dec_pose):
        pose_6d_perturbed_v3 = self._get_perturbed_pose(batch, self.pose_perturbed_key)[:, -1]
        pose_6d_perturbed_ret = dec_pose.clone() # torch.Size([4, 8])
        pose_6d_perturbed_ret[:, 3] = pose_6d_perturbed_v3 # torch.Size([4, 8])
        assert pose_6d_perturbed_ret.shape == dec_pose.shape, f"pose_6d_perturbed_ret shape: {pose_6d_perturbed_ret.shape}"
        return pose_6d_perturbed_ret.to(self.device) # torch.Size([4, 8])
 
    def _log_reconstructions(self, rgb_in, pose_gt, rgb_in_viz, second_pose, log, namespace, zoom_mult):
        # torch.Size([4, 3, 256, 256]) torch.Size([4, 8]) torch.Size([4, 16, 16, 16]) torch.Size([4, 7])
        # Run full forward pass

        xrec_2, poserec_2, posterior_obj_2, bbox_posterior_2, q_loss, _, _ = self.forward(rgb_in, pose_gt, second_pose=second_pose, zoom_mult=zoom_mult)
        xrec, poserec, posterior_obj, bbox_posterior, q_loss, _, _ = self.forward(rgb_in, pose_gt, zoom_mult=zoom_mult)
        
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
        log["reconstructions_rgb" + namespace] = xrec_rgb.clone().detach()
        log["reconstructions_rgb_2" + namespace] = xrec_rgb_2.clone().detach()

        # log = self.log_perturbed_poses(second_pose, rgb_in, pose_gt, rgb_in_viz, log)

        return log
    
    def log_perturbed_poses(self, second_pose, rgb_in, pose_gt, rgb_in_viz, log):
        # test different x and y values in range -1, 1 (along a diagonal)
        x = torch.linspace(-1, 1, steps=5) # torch.Size([5])
        y = torch.linspace(-1, 1, steps=5) # torch.Size([5])

        second_pose_xy_list = []
        for i in range(5):
            for j in range(5):
                second_pose_xy_list.append(torch.tensor([x[i], y[j]]))
        second_pose_xy = torch.stack(second_pose_xy_list).to(self.device) # torch.Size([25, 2])
        # replace the x and y values in the second pose with the new values
        second_pose = second_pose.clone() # torch.Size([3, 13])
        second_pose = second_pose.unsqueeze(0) if second_pose.dim() == 1 else second_pose
        batch_size = len(second_pose)
        num_poses = second_pose_xy.shape[0]
        # create tensor of dim 0 = 25 containing the second pose with the xy values in second_pose_xy
        second_pose = second_pose.repeat(num_poses, 1, 1).permute(1, 0, 2) # torch.Size([3, 25, 13])
        
        # torch.Size([3, 25, 13])
        second_pose[:, :, :2] = second_pose_xy # torch.Size([25, 13])
        # second_pose = second_pose.unsqueeze(0) if second_pose.dim() == 1 else second_pose
        second_pose = second_pose.reshape(num_poses, batch_size, -1)
        for idx, snd_pose in enumerate(second_pose):
            # torch.Size([1, 4])
            snd_pose = snd_pose.unsqueeze(0) # torch.Size([1, 13])
            xrec_2, poserec_2, posterior_obj_2, bbox_posterior_2, q_loss, _, _ = self.forward(rgb_in, pose_gt, second_pose=snd_pose, zoom_mult=zoom_mult)
            xrec_rgb_2 = xrec_2[:, :3, :, :]    

            if rgb_in_viz.shape[1] > 3:
                # colorize with random projection
                assert xrec_rgb_2.shape[1] > 3
                xrec_rgb_2 = self.to_rgb(xrec_rgb_2)

            log[f"reconstructions_rgb_2_{idx}"] = xrec_rgb_2.clone().detach()
        return log
        

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()

        rgb_in, rgb_gt, pose_gt, segm_mask_gt, mask_2d_bbox, class_gt, class_gt_label, bbox_gt, fill_factor_gt, second_pose, zoom_mult \
            = self.get_all_inputs(batch)
        
        rgb_in_viz = self._rescale(rgb_in)
        rgb_gt_viz = self._rescale(rgb_gt)
        
        log["inputs_rgb_in"] = rgb_in_viz.clone().detach()
        log["inputs_rgb_gt"] = rgb_gt_viz.clone().detach()

        if not only_inputs:
            log = self._log_reconstructions(rgb_in, pose_gt, rgb_in_viz, second_pose, log, namespace="", zoom_mult=zoom_mult)

            # EMA weights visualization
            with self.ema_scope():
                log = self._log_reconstructions(rgb_in, pose_gt, rgb_in_viz, second_pose, log, namespace="_ema", zoom_mult=zoom_mult)
        return log
    
    def _rescale(self, x):
        # scale is -1
        x_min = 0.
        x_max = 1.
        return 2. * (x - x.min()) / (x.max() - x.min()) - 1.
    
    def to_rgb(self, x):
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        # clip to [0, 1]
        # x = torch.clamp(x, 0., 1.)
        # x = (2. * x) - 1.
        return x



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
                 pose_encoder_config=None,
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
                 apply_convolutional_shift_img_space=False,
                 apply_convolutional_shift_latent_space=False,
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

        if ddconfig["double_z"]:
            mult_z = 2
        else:
            mult_z = 1
            
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
        # enc_feat_dims = self._get_enc_feat_dims(ddconfig)
        
        pose_encoder_config["params"]["num_channels"] = ddconfig["z_channels"]
        pose_decoder_config["params"]["num_channels"] = ddconfig["z_channels"]
        
        # Pose prediction and latent
        self.pose_decoder = instantiate_from_config(pose_decoder_config)
        self.pose_encoder = instantiate_from_config(pose_encoder_config)
        
        # Decoder Setup
        assert ~(apply_convolutional_shift_img_space and apply_convolutional_shift_latent_space), "Only one of the shift types (image or latent space) can be applied"
        self.apply_conv_shift_img_space = apply_convolutional_shift_img_space
        self.apply_conv_shift_latent_space = apply_convolutional_shift_latent_space

        self.apply_conv_crop_img_space = apply_conv_crop_img_space
        self.apply_conv_crop_latent_space = apply_conv_crop_latent_space
        assert not (self.apply_conv_crop_img_space and self.apply_conv_crop_latent_space), "Only one of the crop types (image or latent space) can be applied"

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
        
        assert not self.apply_conv_shift_img_space, "Not supporting shift in image space for adaptive pose autoencoder"
        assert not self.apply_conv_shift_latent_space, "Not supporting shift in latent space for adaptive pose autoencoder"
    
    def decode(self, z, pose):
        z = self.post_quant_conv(z)
        dec = self.decoder(z, pose)
        return dec

    def forward(self, input_im, pose_gt, sample_posterior=True, second_pose=None, return_pred_indices=False, zoom_mult=None):
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
        # apply_manual_shift = self.apply_conv_shift_img_space or self.apply_conv_shift_latent_space
        apply_manual_crop = self.apply_conv_crop_img_space or self.apply_conv_crop_latent_space
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
        
        # Add gaussian noise to object feature latents
        # if self.add_noise_to_z_obj:
        #     # draw from standard normal distribution
        #     std_normal = Normal(0, 1)
        #     z_obj_noise = std_normal.sample(posterior_obj.mean.shape).to(self.device) # torch.Size([4, 16, 16, 16])
        #     z_obj = z_obj + z_obj_noise
        
        # Replace pose with other pose if supervised with other patch
        if second_pose is not None:
            gen_pose = second_pose.to(pred_pose)
        else:
            gen_pose = pred_pose

        if not apply_manual_crop:

            # Run pose encoder layers  
            z_pose = self.encode_pose(gen_pose) # torch.Size([B, 16, 16, 16])

            assert z_obj.shape == z_pose.shape, f"z_obj shape: {z_obj.shape}, z_pose shape: {z_pose.shape}"
            
            # Add object and pose latents
            z_obj_pose = z_obj + z_pose # torch.Size([B, 16, 16, 16])
        else:
            z_obj_pose = z_obj
            assert zoom_mult is not None, "zoom_mult is not specified, aka None"
            # # Compute shift if shift between both patches
            # if second_pose is not None:
            #     shift_x = second_pose[:, 0] - pose_gt[:, 0]
            #     shift_y = second_pose[:, 1] - pose_gt[:, 1]
            #     d_shift = shift_x.abs().sum() + shift_y.abs().sum() # Do not apply shift layer if shift is 0
            # else:
            #     shift_x, shift_y = torch.zeros_like(pose_gt[:, 0]), torch.zeros_like(pose_gt[:, 0])
            #     d_shift = torch.tensor([0.0])
        
        # # Apply shift in latent space
        # if self.apply_conv_shift_latent_space and d_shift:
        #     z_obj_pose = self.apply_manual_shift(z_obj_pose, shift_x, shift_y)  
        
        # Apply crop in latent space
        if self.apply_conv_crop_latent_space:
            z_obj_pose = self.manual_crop(z_obj_pose, zoom_mult)

        # Predict images from object and pose latents
        pred_obj = self.decode(z_obj_pose, gen_pose) # torch.Size([4, 3, 256, 256])
        
        # Apply crop in image space
        if self.apply_conv_crop_img_space:
            pred_obj = self.manual_crop(pred_obj, zoom_mult)

        # # Apply shift in image space
        # if self.apply_conv_shift_img_space and not self.apply_conv_shift_latent_space and d_shift:
        #     pred_obj = self.apply_manual_shift(pred_obj, shift_x, shift_y)
            
        return pred_obj, pred_pose, posterior_obj, bbox_posterior, q_loss, ind_obj, ind_pose

    def batch_center_crop_resize(self, images, H_crops, W_crops):
        batch_size, channels, H, W = images.shape # torch.Size([4, 3, 256, 256])
        cropped_resized_images = torch.zeros_like(images) # torch.Size([4, 3, 256, 256])
        resizer = T.Resize(size=(H, W),interpolation=T.InterpolationMode.BILINEAR)
        for i in range(batch_size):
            H_crop = int(H_crops[i])
            W_crop = int(W_crops[i])

            center_cropper = T.CenterCrop(size=(H_crop, W_crop))
            cropped_image = center_cropper(images[i].unsqueeze(0))
            cropped_resized_images[i] = resizer(cropped_image)

        return cropped_resized_images
        
    def manual_crop(self, images, zoom_mult):
        batch_size, channels, H, W = images.shape
        
        H_crops = (H * zoom_mult).long()
        W_crops = (W * zoom_mult).long()
        
        original_crop_recropped_resized = self.batch_center_crop_resize(images, H_crops, W_crops)

        return original_crop_recropped_resized