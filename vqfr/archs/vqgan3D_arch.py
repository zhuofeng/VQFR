import pdb
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import numpy as np

from vqfr.archs.quantizer_arch import build_quantizer
from vqfr.utils.registry import ARCH_REGISTRY
from vqfr.archs.attention import MultiHeadAttention
from vqfr.archs.utils import shift_dim

class Downsample(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode='constant', value=0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        x = self.conv(x)
        return x


class ResnetBlock(nn.Module):

    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=channels_in, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=(3, 3), stride=(1, 1), padding=1)

        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=channels_out, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(channels_out, channels_out, kernel_size=(3, 3), stride=(1, 1), padding=1)

        self.act = nn.SiLU(inplace=True)
        if channels_in != channels_out:
            self.residual_func = nn.Conv2d(channels_in, channels_out, kernel_size=1)
        else:
            self.residual_func = nn.Identity()

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)
        return x + self.residual_func(residual)


class AttnBlock(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_

class VQGANEncoder_features(nn.Module):

    def __init__(self, base_channels, channel_multipliers, num_blocks, use_enc_attention, code_dim):
        super(VQGANEncoder_features, self).__init__()

        self.num_levels = len(channel_multipliers)
        self.num_blocks = num_blocks

        self.conv_in = nn.Conv2d(
            3, base_channels * channel_multipliers[0], kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.blocks = nn.ModuleList()

        for i in range(self.num_levels):
            blocks = []
            if i == 0:
                channels_prev = base_channels * channel_multipliers[i]
            else:
                channels_prev = base_channels * channel_multipliers[i - 1]

            if i != 0:
                blocks.append(Downsample(channels_prev))

            channels = base_channels * channel_multipliers[i]
            blocks.append(ResnetBlock(channels_prev, channels))
            if i == self.num_levels - 1 and use_enc_attention:
                blocks.append(AttnBlock(channels))

            for j in range(self.num_blocks - 1):
                blocks.append(ResnetBlock(channels, channels))
                if i == self.num_levels - 1 and use_enc_attention:
                    blocks.append(AttnBlock(channels))

            self.blocks.append(nn.Sequential(*blocks))

        channels = base_channels * channel_multipliers[-1]
        if use_enc_attention:
            self.mid_blocks = nn.Sequential(
                ResnetBlock(channels, channels), AttnBlock(channels), ResnetBlock(channels, channels))
        else:
            self.mid_blocks = nn.Sequential(ResnetBlock(channels, channels), ResnetBlock(channels, channels))

        self.conv_out = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True), nn.SiLU(inplace=True),
            nn.Conv2d(channels, code_dim, kernel_size=3, padding=1))

    def forward(self, x):
        features = []
        x = self.conv_in(x)
        for i in range(self.num_levels):
            x = self.blocks[i](x)
            features.append(x)

        x = self.mid_blocks(x)
        x = self.conv_out(x)
        return x, features


class AxialBlock(nn.Module):
    def __init__(self, n_hiddens, n_head):
        super().__init__()
        kwargs = dict(shape=(0,) * 3, dim_q=n_hiddens,
                      dim_kv=n_hiddens, n_head=n_head,
                      n_layer=1, causal=False, attn_type='axial')
        self.attn_w = MultiHeadAttention(attn_kwargs=dict(axial_dim=-2),
                                         **kwargs)
        self.attn_h = MultiHeadAttention(attn_kwargs=dict(axial_dim=-3),
                                         **kwargs)
        self.attn_t = MultiHeadAttention(attn_kwargs=dict(axial_dim=-4),
                                         **kwargs)

    def forward(self, x):
        x = shift_dim(x, 1, -1)
        x = self.attn_w(x, x, x) + self.attn_h(x, x, x) + self.attn_t(x, x, x)
        x = shift_dim(x, -1, 1)
        return x

class AttentionResidualBlock(nn.Module):
    def __init__(self, n_hiddens):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm3d(n_hiddens),
            nn.ReLU(),
            SamePadConv3d(n_hiddens, n_hiddens // 2, 3, bias=False),
            nn.BatchNorm3d(n_hiddens // 2),
            nn.ReLU(),
            SamePadConv3d(n_hiddens // 2, n_hiddens, 1, bias=False),
            nn.BatchNorm3d(n_hiddens),
            nn.ReLU(),
            AxialBlock(n_hiddens, 2)
        )

    def forward(self, x):
        return x + self.block(x)

# Does not support dilation
class SamePadConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        # assumes that the input shape is divisible by stride
        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]: # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(F.pad(x, self.pad_input))

class SamePadConvTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]: # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input

        self.convt = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                        stride=stride, bias=bias,
                                        padding=tuple([k - 1 for k in kernel_size]))

    def forward(self, x):
        return self.convt(F.pad(x, self.pad_input))

class Encoder3D(nn.Module):
    def __init__(self, n_hiddens, n_res_layers, downsample):
        super().__init__()
        n_times_downsample = np.array([int(math.log2(d)) for d in downsample])
        self.convs = nn.ModuleList()
        max_ds = n_times_downsample.max()
        for i in range(max_ds):
            in_channels = 1 if i == 0 else n_hiddens
            stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
            conv = SamePadConv3d(in_channels, n_hiddens, 4, stride=stride)
            self.convs.append(conv)
            n_times_downsample -= 1
        self.conv_last = SamePadConv3d(in_channels, n_hiddens, kernel_size=3)

        self.res_stack = nn.Sequential(
            *[AttentionResidualBlock(n_hiddens)
              for _ in range(n_res_layers)],
            nn.BatchNorm3d(n_hiddens),
            nn.ReLU()
        )

    def forward(self, x):
        h = x
        for conv in self.convs:
            h = F.relu(conv(h))
        h = self.conv_last(h)
        h = self.res_stack(h)
        return h

class Decoder3D(nn.Module):
    def __init__(self, n_hiddens, n_res_layers, upsample):
        super().__init__()
        self.res_stack = nn.Sequential(
            *[AttentionResidualBlock(n_hiddens)
              for _ in range(n_res_layers)],
            nn.BatchNorm3d(n_hiddens),
            nn.ReLU()
        )

        n_times_upsample = np.array([int(math.log2(d)) for d in upsample])
        max_us = n_times_upsample.max()
        self.convts = nn.ModuleList()
        for i in range(max_us):
            out_channels = 1 if i == max_us - 1 else n_hiddens
            us = tuple([2 if d > 0 else 1 for d in n_times_upsample])
            convt = SamePadConvTranspose3d(n_hiddens, out_channels, 4,
                                           stride=us)
            self.convts.append(convt)
            n_times_upsample -= 1

    def forward(self, x):
        h = self.res_stack(x)
        for i, convt in enumerate(self.convts):
            h = convt(h)
            if i < len(self.convts) - 1:
                h = F.relu(h)
        return h

@ARCH_REGISTRY.register()
class VQGAN3D(nn.Module):

    def __init__(self,
                 base_channels,
                 channel_multipliers,
                 num_enc_blocks,
                 use_enc_attention,
                 num_dec_blocks,
                 use_dec_attention,
                 code_dim,
                 quantizer_opt,
                 fix_keys=[]):
        super().__init__()

        n_hiddens = 240
        n_res_layers = 4
        downsample = [4, 4, 4]
        self.encoder = Encoder3D(n_hiddens, n_res_layers, downsample)
        self.decoder = Decoder3D(n_hiddens, n_res_layers, downsample)

        self.pre_vq_conv = SamePadConv3d(n_hiddens, code_dim, 1)
        self.post_vq_conv = SamePadConv3d(code_dim, n_hiddens, 1)

        self.quantizer = build_quantizer(quantizer_opt)

        self.apply(self._init_weights)

        for k, v in self.named_parameters():
            for fix_k in fix_keys:
                if fix_k in k:
                    v.requires_grad = False

    @torch.no_grad()
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
    def get_last_layer(self):
        return self.decoder.convts[-1].weight

    def forward(self, x, iters=-1, return_keys=('dec')):
        res = {}
        z = self.pre_vq_conv(self.encoder(x))
        quant_feat, emb_loss, quant_index = self.quantizer(z)
        res['quant_feat'] = quant_feat
        res['quant_index'] = quant_index
        if 'dec' in return_keys:
            dec = self.decoder(self.post_vq_conv(quant_feat))
            res['dec'] = dec
        
        return res, emb_loss