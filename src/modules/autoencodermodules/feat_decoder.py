"""
src/modules/autoencodermodules/feat_decoder.py
=======================================================================
Code adapted from https://github.com/NVlabs/stylegan2-ada-pytorch and 
https://github.com/CompVis/latent-diffusion. Licenses provided below.
=======================================================================
NVIDIA Source Code License for StyleGAN2 with Adaptive Discriminator Augmentation (ADA)
Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
=======================================================================
MIT License
Copyright (c) 2022 Machine Vision and Learning Group, LMU Munich
"""

import torch
import torch.nn as nn
import numpy as np
from ldm.modules.diffusionmodules.model import Normalize, make_attn, Upsample, nonlinearity, ResnetBlock
from ldm.modules.diffusionmodules.model import Decoder as LDMDecoder
from src.modules.autoencodermodules.adaptiveconv import SynthesisLayer as AdpativeConv2dLayer
from src.util.misc import PositionalEncoding

class FeatDecoder(LDMDecoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class AdaptiveFeatDecoder(nn.Module):
    def __init__(self, *, ch, out_ch, num_classes, pose_dim=13, w_dim=512, ch_mult=(1,2,4,8), num_res_blocks,
                attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                attn_type="vanilla", pose_embedding_layers=1, 
                pose_embedding_map_hidden_dim=512, pe_num_frequencies=6, 
                mid_adaptive=True, upsample_adaptive=False, 
                **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.num_classes = num_classes

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        self.mid_adaptive=mid_adaptive
        self.upsample_adaptive=upsample_adaptive

        # z to block_in
        self.conv_in = AdpativeConv2dLayer(z_channels, # 16
                                            block_in,
                                            w_dim=w_dim,
                                            resolution=curr_res,
                                            kernel_size=3)

        # middle
        self.mid = nn.Module()
        if self.mid_adaptive:
            self.mid.block_1 = AdaptiveResnetBlock(in_channels=block_in,
                                        resolution=curr_res,
                                        out_channels=block_in,
                                        temb_channels=self.temb_ch,
                                        dropout=dropout)
        else:
            self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                        out_channels=block_in,
                                        temb_channels=self.temb_ch,
                                        dropout=dropout)

        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)

        if self.mid_adaptive:
            self.mid.block_2 = AdaptiveResnetBlock(in_channels=block_in,
                                            resolution=curr_res,
                                        out_channels=block_in,
                                        temb_channels=self.temb_ch,
                                        dropout=dropout)
        else:
            self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                        out_channels=block_in,
                                        temb_channels=self.temb_ch,
                                        dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if self.upsample_adaptive:
                    block.append(AdaptiveResnetBlock(in_channels=block_in,
                                            resolution=curr_res,
                                            out_channels=block_out,
                                            temb_channels=self.temb_ch,
                                            dropout=dropout))

                else:
                    block.append(ResnetBlock(in_channels=block_in,
                                            out_channels=block_out,
                                            temb_channels=self.temb_ch,
                                            dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
                                        
        # add PositionalEncoding with k channels
        num_channels = pose_dim - num_classes
        self.pose_pe = PositionalEncoding(num_channels=num_channels, num_frequencies=pe_num_frequencies)

        # pose embedding map: maps pose_pe output to w
        curr_in_dim = self.pose_pe.output_dim + num_classes
        curr_out_dim = pose_embedding_map_hidden_dim
        pose_embedding_map = nn.Sequential()
        for N in range(pose_embedding_layers):
            if N == (pose_embedding_layers - 1):
                curr_out_dim = w_dim
            else:
                curr_out_dim = pose_embedding_map_hidden_dim
            
            pose_embedding_map.add_module(f'layer_{N}', nn.Linear(curr_in_dim, curr_out_dim))

            curr_in_dim = curr_out_dim

        self.pose_embedding_map = pose_embedding_map
        
        
    def forward(self, z, pose):
        # pose --> pose_pe --> w
        # pose: torch.Size([4, 13])
        pose_no_class = pose[:, :-self.num_classes] # torch.Size([4, 10])
        class_label = pose[:, -self.num_classes:] # torch.Size([4, 3])
        pose_pe_no_class = self.pose_pe(pose_no_class) # torch.Size([4, 156])
        pose_pe = torch.cat((pose_pe_no_class, class_label), dim=1) # torch.Size([4, 159])

        w = self.pose_embedding_map(pose_pe) # torch.Size([4, 512])

        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        # torch.Size([4, 16, 16, 16]), torch.Size([4, 512])
        h = self.conv_in(z, w) #[4, 512, 16, 16])


        # middle
        h = self.mid.block_1(h, w, temb) if self.mid_adaptive else self.mid.block_1(h, temb)
        # torch.Size([4, 512, 16, 16])
        
        h = self.mid.attn_1(h)# torch.Size([4, 512, 16, 16])
        
        h = self.mid.block_2(h, w, temb) if self.mid_adaptive else self.mid.block_2(h, temb) 
        # torch.Size([4, 512, 16, 16])

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, w, temb) if self.upsample_adaptive else self.up[i_level].block[i_block](h, temb) # torch.Size([4, 512, 16, 16])
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h) # torch.Size([4, 128, 256, 256])
        if self.tanh_out:
            h = torch.tanh(h)
        return h

class SplitAdaptiveFeatDecoder(nn.Module):
    def __init__(self, *, ch, out_ch, num_classes, pose_dim=13, w_dim=512, ch_mult=(1,2,4,8), num_res_blocks,
                attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                attn_type="vanilla", pose_embedding_layers=1, 
                pose_embedding_map_hidden_dim=512, pe_num_frequencies=6, 
                mid_adaptive=True, upsample_adaptive=False, 
                **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.num_classes = num_classes

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        self.mid_adaptive=mid_adaptive
        self.upsample_adaptive=upsample_adaptive

        # z to block_in
        self.conv_in = AdpativeConv2dLayer(z_channels, # 16
                                            block_in,
                                            w_dim=w_dim,
                                            resolution=curr_res,
                                            kernel_size=3)

        # middle
        self.mid = nn.Module()
        if self.mid_adaptive:
            self.mid.block_1 = AdaptiveResnetBlock(in_channels=block_in,
                                        resolution=curr_res,
                                        out_channels=block_in,
                                        temb_channels=self.temb_ch,
                                        dropout=dropout)
        else:
            self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                        out_channels=block_in,
                                        temb_channels=self.temb_ch,
                                        dropout=dropout)

        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)

        if self.mid_adaptive:
            self.mid.block_2 = AdaptiveResnetBlock(in_channels=block_in,
                                            resolution=curr_res,
                                        out_channels=block_in,
                                        temb_channels=self.temb_ch,
                                        dropout=dropout)
        else:
            self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                        out_channels=block_in,
                                        temb_channels=self.temb_ch,
                                        dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if self.upsample_adaptive:
                    block.append(AdaptiveResnetBlock(in_channels=block_in,
                                            resolution=curr_res,
                                            out_channels=block_out,
                                            temb_channels=self.temb_ch,
                                            dropout=dropout))

                else:
                    block.append(ResnetBlock(in_channels=block_in,
                                            out_channels=block_out,
                                            temb_channels=self.temb_ch,
                                            dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
                                        
        # add PositionalEncoding with k channels
        num_channels = pose_dim - num_classes
        self.pose_pe = PositionalEncoding(num_channels=num_channels, num_frequencies=pe_num_frequencies)

        # pose embedding map: maps pose_pe output to w
        curr_in_dim = self.pose_pe.output_dim + num_classes
        curr_out_dim = pose_embedding_map_hidden_dim
        pose_embedding_map = nn.Sequential()
        for N in range(pose_embedding_layers):
            if N == (pose_embedding_layers - 1):
                curr_out_dim = w_dim
            else:
                curr_out_dim = pose_embedding_map_hidden_dim
            
            pose_embedding_map.add_module(f'layer_{N}', nn.Linear(curr_in_dim, curr_out_dim))

            curr_in_dim = curr_out_dim

        self.pose_embedding_map = pose_embedding_map
        
    def _create_masked_pose(self, pose, mask_idx):
        pose_masked = pose.clone()
        mask = torch.ones_like(pose_masked)
        for idx in mask_idx:
            mask[:, idx] = 0.0
        pose_masked = pose_masked * mask
        return pose_masked

    def forward(self, z, pose):
        # pose --> pose_pe --> w
        # pose: torch.Size([4, 13])
        pose_no_class = pose[:, :-self.num_classes] # torch.Size([4, 10])

        pose_no_zoom = self._create_masked_pose(pose_no_class, [2, 7])
        pose_no_shift = self._create_masked_pose(pose_no_class, [0, 1])
        
        class_label = pose[:, -self.num_classes:] # torch.Size([4, 3])
        pose_pe_no_class = self.pose_pe(pose_no_class) # torch.Size([4, 156])
        pose_pe_no_zoom = self.pose_pe(pose_no_zoom)
        pose_pe_no_shift = self.pose_pe(pose_no_shift)
        
        pose_pe = torch.cat((pose_pe_no_class, class_label), dim=1) # torch.Size([4, 159])
        pose_pe_no_zoom = torch.cat((pose_pe_no_zoom, class_label), dim=1)
        pose_pe_no_shift = torch.cat((pose_pe_no_shift, class_label), dim=1)

        w = self.pose_embedding_map(pose_pe) # torch.Size([4, 512])
        w_no_zoom = self.pose_embedding_map(pose_pe_no_zoom)
        w_no_shift = self.pose_embedding_map(pose_pe_no_shift)
        
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        # torch.Size([4, 16, 16, 16]), torch.Size([4, 512])
        h = self.conv_in(z, w) #[4, 512, 16, 16])


        # middle

        # block 1: no zoom
        h = self.mid.block_1(h, w_no_zoom, temb) if self.mid_adaptive else self.mid.block_1(h, temb) # torch.Size([4, 512, 16, 16])
        
        h = self.mid.attn_1(h)# torch.Size([4, 512, 16, 16])
        
        # block 2: no shift
        h = self.mid.block_2(h, w_no_shift, temb) if self.mid_adaptive else self.mid.block_2(h, temb)  # torch.Size([4, 512, 16, 16])

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, w, temb) if self.upsample_adaptive else self.up[i_level].block[i_block](h, temb) # torch.Size([4, 512, 16, 16])
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h) # torch.Size([4, 128, 256, 256])
        if self.tanh_out:
            h = torch.tanh(h)
        return h

class SplitAdaptiveFeatDecoder4Block(nn.Module):
    def __init__(self, *, ch, out_ch, num_classes, pose_dim=13, w_dim=512, ch_mult=(1,2,4,8), num_res_blocks,
                attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                attn_type="vanilla", pose_embedding_layers=1, 
                pose_embedding_map_hidden_dim=512, pe_num_frequencies=6, 
                mid_adaptive=True, upsample_adaptive=False, 
                **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.num_classes = num_classes

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        self.mid_adaptive=mid_adaptive
        self.upsample_adaptive=upsample_adaptive

        # z to block_in
        self.conv_in = AdpativeConv2dLayer(z_channels, # 16
                                            block_in,
                                            w_dim=w_dim,
                                            resolution=curr_res,
                                            kernel_size=3)

        # middle
        self.mid = nn.Module()
        if self.mid_adaptive:
            self.mid.block_1 = AdaptiveResnetBlock(in_channels=block_in,
                                        resolution=curr_res,
                                        out_channels=block_in,
                                        temb_channels=self.temb_ch,
                                        dropout=dropout)
        else:
            self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                        out_channels=block_in,
                                        temb_channels=self.temb_ch,
                                        dropout=dropout)

        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)

        if self.mid_adaptive:
            self.mid.block_2 = AdaptiveResnetBlock(in_channels=block_in,
                                            resolution=curr_res,
                                        out_channels=block_in,
                                        temb_channels=self.temb_ch,
                                        dropout=dropout)
        else:
            self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                        out_channels=block_in,
                                        temb_channels=self.temb_ch,
                                        dropout=dropout)

        self.mid.attn_2 = make_attn(block_in, attn_type=attn_type)

        if self.mid_adaptive:
            self.mid.block_3 = AdaptiveResnetBlock(in_channels=block_in,
                                            resolution=curr_res,
                                        out_channels=block_in,
                                        temb_channels=self.temb_ch,
                                        dropout=dropout)
        else:
            self.mid.block_3 = ResnetBlock(in_channels=block_in,
                                        out_channels=block_in,
                                        temb_channels=self.temb_ch,
                                        dropout=dropout)
        
        self.mid.attn_3 = make_attn(block_in, attn_type=attn_type)

        if self.mid_adaptive:
            self.mid.block_4 = AdaptiveResnetBlock(in_channels=block_in,
                                            resolution=curr_res,
                                        out_channels=block_in,
                                        temb_channels=self.temb_ch,
                                        dropout=dropout)
        else:
            self.mid.block_4 = ResnetBlock(in_channels=block_in,
                                        out_channels=block_in,
                                        temb_channels=self.temb_ch,
                                        dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if self.upsample_adaptive:
                    assert not self.upsample_adaptive, "do not use adaptive upsampling!"
                    block.append(AdaptiveResnetBlock(in_channels=block_in,
                                            resolution=curr_res,
                                            out_channels=block_out,
                                            temb_channels=self.temb_ch,
                                            dropout=dropout))

                else:
                    block.append(ResnetBlock(in_channels=block_in,
                                            out_channels=block_out,
                                            temb_channels=self.temb_ch,
                                            dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
                                        
        # add PositionalEncoding with k channels
        num_channels = pose_dim - num_classes
        self.pose_pe = PositionalEncoding(num_channels=num_channels, num_frequencies=pe_num_frequencies)

        # pose embedding map: maps pose_pe output to w
        curr_in_dim = self.pose_pe.output_dim + num_classes
        curr_out_dim = pose_embedding_map_hidden_dim
        pose_embedding_map = nn.Sequential()
        for N in range(pose_embedding_layers):
            if N == (pose_embedding_layers - 1):
                curr_out_dim = w_dim
            else:
                curr_out_dim = pose_embedding_map_hidden_dim
            
            pose_embedding_map.add_module(f'layer_{N}', nn.Linear(curr_in_dim, curr_out_dim))

            curr_in_dim = curr_out_dim

        self.pose_embedding_map = pose_embedding_map
        
    def _create_masked_pose(self, pose, mask_idx):
        pose_masked = pose.clone()
        mask = torch.ones_like(pose_masked)
        for idx in mask_idx:
            mask[:, idx] = 0.0
        pose_masked = pose_masked * mask
        return pose_masked

    def forward(self, z, pose):
        # pose --> pose_pe --> w
        # pose: torch.Size([4, 13])
        pose_no_class = pose[:, :-self.num_classes] # torch.Size([4, 10])

        pose_no_zoom = self._create_masked_pose(pose_no_class, [2, 7])
        pose_no_shift = self._create_masked_pose(pose_no_class, [0, 1])
        
        class_label = pose[:, -self.num_classes:] # torch.Size([4, 3])
        pose_pe_no_class = self.pose_pe(pose_no_class) # torch.Size([4, 156])
        pose_pe_no_zoom = self.pose_pe(pose_no_zoom)
        pose_pe_no_shift = self.pose_pe(pose_no_shift)
        
        pose_pe = torch.cat((pose_pe_no_class, class_label), dim=1) # torch.Size([4, 159])
        pose_pe_no_zoom = torch.cat((pose_pe_no_zoom, class_label), dim=1)
        pose_pe_no_shift = torch.cat((pose_pe_no_shift, class_label), dim=1)

        w = self.pose_embedding_map(pose_pe) # torch.Size([4, 512])
        w_no_zoom = self.pose_embedding_map(pose_pe_no_zoom)
        w_no_shift = self.pose_embedding_map(pose_pe_no_shift)
        
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        # torch.Size([4, 16, 16, 16]), torch.Size([4, 512])
        h = self.conv_in(z, w) #[4, 512, 16, 16])


        # middle

        # block 1: no zoom
        h = self.mid.block_1(h, w_no_zoom, temb) if self.mid_adaptive else self.mid.block_1(h, temb) # torch.Size([4, 512, 16, 16])
        
        h = self.mid.attn_1(h)# torch.Size([4, 512, 16, 16])
        
        # block 2: no zoom
        h = self.mid.block_2(h, w_no_zoom, temb) if self.mid_adaptive else self.mid.block_2(h, temb)  # torch.Size([4, 512, 16, 16])

        h = self.mid.attn_2(h)# torch.Size([4, 512, 16, 16])

        # block 3: no shift
        h = self.mid.block_3(h, w_no_shift, temb) if self.mid_adaptive else self.mid.block_3(h, temb) # torch.Size([4, 512, 16, 16])
        
        h = self.mid.attn_3(h)# torch.Size([4, 512, 16, 16])
        
        # block 4: no shift
        h = self.mid.block_4(h, w_no_shift, temb) if self.mid_adaptive else self.mid.block_4(h, temb)  # torch.Size([4, 512, 16, 16])

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, w, temb) if self.upsample_adaptive else self.up[i_level].block[i_block](h, temb) # torch.Size([4, 512, 16, 16])
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h) # torch.Size([4, 128, 256, 256])
        if self.tanh_out:
            h = torch.tanh(h)
        return h

class AdaptiveResnetBlock(nn.Module):
    def __init__(self, *, in_channels, resolution, w_dim=512, out_channels=None, conv_shortcut=False,
                dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = AdpativeConv2dLayer(in_channels,
                                        out_channels,
                                        w_dim=w_dim,
                                        resolution=resolution,
                                        kernel_size=3,)

        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                            out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)

        self.conv2 = AdpativeConv2dLayer(out_channels,
                                        out_channels,
                                        w_dim=w_dim,
                                        resolution=resolution,
                                        kernel_size=3,)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = AdpativeConv2dLayer(in_channels,
                                                    out_channels,
                                                    w_dim=w_dim,
                                                    resolution=resolution,
                                                    kernel_size=3,)

            else:
                self.nin_shortcut = AdpativeConv2dLayer(in_channels,
                                                    out_channels,
                                                    w_dim=w_dim,
                                                    resolution=resolution,
                                                    kernel_size=1)

    def forward(self, x, w, temb):
        
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h, w)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h, w)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x, w)
            else:
                x = self.nin_shortcut(x, w)

        return x+h
