# src/modules/autoencodermodules/feat_decoder.py
from ldm.modules.diffusionmodules.model import Decoder as LDMDecoder
import torch.nn as nn
from src.modules.autoencodermodules.adaptiveconv import SynthesisLayer as AdpativeConv2dLayer
import numpy as np
import torch
from ldm.modules.diffusionmodules.model import Normalize, make_attn, Upsample, nonlinearity

class FeatDecoder(LDMDecoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class PositionalEncoding(nn.Module):
    def __init__(self, num_channels=13, num_frequencies=6):
        super(PositionalEncoding, self).__init__()
        self.num_channels = num_channels
        self.num_frequencies = num_frequencies
        
        self.output_dim = num_channels * 2 * self.num_frequencies

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.unsqueeze(-1)  # Shape: (batch_size, 13, 1)
        
        # Prepare the frequency bands
        frequencies = torch.arange(self.num_frequencies, dtype=torch.float32).to(x.device)
        frequencies = 2 ** frequencies * torch.pi  # Shape: (6,)
        
        # Compute the positional encodings
        sinusoids = torch.einsum('bix,f->bixf', x, frequencies)  # Shape: (batch_size, 13, 1, 6)
        sinusoids = torch.cat([torch.sin(sinusoids), torch.cos(sinusoids)], dim=-1)  # Shape: (batch_size, 13, 1, 12)
        
        # Flatten the last two dimensions
        sinusoids = sinusoids.view(batch_size, self.output_dim)  # Shape: (batch_size, 156)
        
        return sinusoids

class AdaptiveFeatDecoder(nn.Module):
    def __init__(self, *, ch, out_ch, pose_dim=13, w_dim=512, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", pose_embedding_layers=1, 
                 pose_embedding_map_hidden_dim=512, pe_num_frequencies=6, **ignorekwargs):
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

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = AdpativeConv2dLayer(z_channels,
                                            block_in,
                                            w_dim=w_dim,
                                            resolution=16,
                                            kernel_size=3)
            #                            stride=1,
            #                            padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = AdaptiveResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = AdaptiveResnetBlock(in_channels=block_in,
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
                block.append(AdaptiveResnetBlock(in_channels=block_in,
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
        self.conv_out = AdpativeConv2dLayer(block_in,
                                        out_ch,
                                        w_dim=w_dim,
                                        resolution=16,
                                        kernel_size=3,)
        #                                 stride=1,
        #                                 padding=1)

        # add PositionalEncoding with k channels
        self.pose_pe = PositionalEncoding(num_channels=pose_dim, num_frequencies=pe_num_frequencies)

        # pose embedding map: maps pose_pe output to w
        curr_in_dim = self.pose_pe.output_dim
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
        pose_pe = self.pose_pe(pose)
        w = self.pose_embedding_map(pose_pe)

        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z, w) # torch.Size([8, 16, 16, 16]), torch.Size([8, 512])

        # middle
        h = self.mid.block_1(h, w, temb) # torch.Size([8, 512, 16, 16])
        h = self.mid.attn_1(h)# torch.Size([8, 512, 16, 16])
        h = self.mid.block_2(h, w, temb) # torch.Size([8, 512, 16, 16])

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, w, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h, w)
        if self.tanh_out:
            h = torch.tanh(h)
        return h

class AdaptiveResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = AdpativeConv2dLayer(in_channels,
                                        out_channels,
                                        w_dim=512,
                                        resolution=16,
                                        kernel_size=3,)
        #                              stride=1,
        #                              padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)

        self.conv2 = AdpativeConv2dLayer(out_channels,
                                        out_channels,
                                        w_dim=512,
                                        resolution=16,
                                        kernel_size=3,)
        #                              stride=1,
        #                              padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = AdpativeConv2dLayer(in_channels,
                                                     out_channels,
                                                     w_dim=512,
                                                    resolution=16,
                                                     kernel_size=3,)
                                                    #  stride=1,
                                                    #  padding=1)
            else:
                self.nin_shortcut = AdpativeConv2dLayer(in_channels,
                                                    out_channels,
                                                    w_dim=512,
                                                    resolution=16,
                                                    kernel_size=1,)
                                                    # stride=1,
                                                    # padding=0)

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
