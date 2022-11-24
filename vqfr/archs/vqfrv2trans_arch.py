import pdb
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from distutils.version import LooseVersion
from einops import rearrange
from timm.models.layers import trunc_normal_

from vqfr.archs.vqganv2_arch import ResnetBlock, VQGANDecoder, VQGANEncoder, build_quantizer, VQGANEncoder_features
from vqfr.ops.dcn import ModulatedDeformConvPack, modulated_deform_conv
from vqfr.utils import get_root_logger
from vqfr.utils.registry import ARCH_REGISTRY
from vqfr.archs.ref_map_util import feature_match_index
from vqfr.archs.arch_util import tensor_shift
from vqfr.archs.dcn_v2 import DCN_sep_pre_multi_offset_flow_similarity as DynAgg
from vqfr.archs.flow_similarity_corres_generation_arch import FlowSimCorrespondenceGenerationArch as FlowSim
from vqfr.archs.swin_unetv3_ref_restoration_arch import DynamicAggregationRestoration as aggmodel

class DCNv2Pack(ModulatedDeformConvPack):
    """Modulated deformable conv for deformable alignment.
    Different from the official DCNv2Pack, which generates offsets and masks
    from the preceding features, this DCNv2Pack takes another different
    features to generate offsets and masks.
    Ref:
        Delving Deep into Deformable Alignment in Video Super-Resolution.
    """

    def forward(self, x, feat):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        offset_absmean = torch.mean(torch.abs(offset))
        if offset_absmean > 50:
            logger = get_root_logger()
            logger.warning(f'Offset abs mean is {offset_absmean}, larger than 50.')

        if LooseVersion(torchvision.__version__) >= LooseVersion('0.9.0'):
            return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
                                                 self.dilation, mask)
        else:
            return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding,
                                         self.dilation, self.groups, self.deformable_groups)


class TextureWarpingModule(nn.Module):

    def __init__(self, channel, cond_channels, cond_downscale_rate, deformable_groups, previous_offset_channel=0):
        super(TextureWarpingModule, self).__init__()
        self.cond_downscale_rate = cond_downscale_rate
        self.offset_conv1 = nn.Sequential(
            nn.Conv2d(channel + cond_channels, channel, kernel_size=1),
            nn.GroupNorm(num_groups=32, num_channels=channel, eps=1e-6, affine=True), nn.SiLU(inplace=True),
            nn.Conv2d(channel, channel, groups=channel, kernel_size=7, padding=3),
            nn.GroupNorm(num_groups=32, num_channels=channel, eps=1e-6, affine=True), nn.SiLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=1))

        self.offset_conv2 = nn.Sequential(
            nn.Conv2d(channel + previous_offset_channel, channel, 3, 1, 1),
            nn.GroupNorm(num_groups=32, num_channels=channel, eps=1e-6, affine=True), nn.SiLU(inplace=True))
        self.dcn = DCNv2Pack(channel, channel, 3, padding=1, deformable_groups=deformable_groups)
        
    def forward(self, x_main, inpfeat, previous_offset=None):
        _, _, h, w = inpfeat.shape
        inpfeat = F.interpolate(
            inpfeat,
            size=(h // self.cond_downscale_rate, w // self.cond_downscale_rate),
            mode='bilinear',
            align_corners=False)
        offset = self.offset_conv1(torch.cat([inpfeat, x_main], dim=1))
        if previous_offset is None:
            offset = self.offset_conv2(offset)
        else:
            offset = self.offset_conv2(torch.cat([offset, previous_offset], dim=1))
        warp_feat = self.dcn(x_main, offset)
        return warp_feat, offset


class MainDecoder(nn.Module):

    def __init__(self, base_channels, channel_multipliers, align_opt):
        super(MainDecoder, self).__init__()
        self.num_levels = len(channel_multipliers)

        self.decoder_dict = nn.ModuleDict()
        self.pre_upsample_dict = nn.ModuleDict()
        self.align_func_dict = nn.ModuleDict()
        self.toq_dict = nn.ModuleDict()
        self.tok_dict = nn.ModuleDict()
        self.tov_dict = nn.ModuleDict()
        self.flowsim = nn.ModuleDict()
        self.warpmodel = nn.ModuleDict()

        for i in reversed(range(self.num_levels)):
            if i == self.num_levels - 1:
                channels_prev = base_channels * channel_multipliers[i]
            else:
                channels_prev = base_channels * channel_multipliers[i + 1]
            channels = base_channels * channel_multipliers[i]

            if i != self.num_levels - 1:
                self.pre_upsample_dict['Level_%d' % 2**i] = \
                    nn.Sequential(
                        nn.UpsamplingNearest2d(scale_factor=2),
                        nn.Conv2d(channels_prev, channels, kernel_size=3, padding=1))

            previous_offset_channel = 0 if i == self.num_levels - 1 else channels_prev

            self.align_func_dict['Level_%d' % (2**i)] = \
                TextureWarpingModule(
                    channel=channels,
                    cond_channels=align_opt['cond_channels'],
                    cond_downscale_rate=2**i,
                    deformable_groups=align_opt['deformable_groups'],
                    previous_offset_channel=previous_offset_channel)

            if i != self.num_levels - 1:
                self.decoder_dict['Level_%d' % 2**i] = ResnetBlock(2 * channels, channels)

            self.toq_dict['Level_%d' % 2**i] = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.tok_dict['Level_%d' % 2**i] = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.tov_dict['Level_%d' % 2**i] = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.flowsim['Level_%d' % 2**i] = FlowSim(patch_size=3, stride=1)   
            self.warpmodel['Level_%d' % 2**i] = aggmodel(channels*3, channels)

    def index_to_flow(self, max_idx):
        device = max_idx.device
        # max_idx to flow
        h, w = max_idx.size()
        flow_w = max_idx % w
        flow_h = max_idx // w

        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, h).to(device),
            torch.arange(0, w).to(device))
        grid = torch.stack((grid_x, grid_y), 2).unsqueeze(0).float().to(device)
        grid.requires_grad = False
        flow = torch.stack((flow_w, flow_h),
                           dim=2).unsqueeze(0).float().to(device)
        flow = flow - grid  # shape:(1, w, h, 2)
        flow = torch.nn.functional.pad(flow, (0, 0, 0, 2, 0, 2))

        return flow

    def flow_warp(self,
                  x,
                  flow,
                  interp_mode='bilinear',
                  padding_mode='zeros',
                  align_corners=True):
        """Warp an image or feature map with optical flow.
        Args:
            x (Tensor): Tensor with size (n, c, h, w).
            flow (Tensor): Tensor with size (n, h, w, 2), normal value.
            interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
            padding_mode (str): 'zeros' or 'border' or 'reflection'.
                Default: 'zeros'.
            align_corners (bool): Before pytorch 1.3, the default value is
                align_corners=True. After pytorch 1.3, the default value is
                align_corners=False. Here, we use the True as default.
        Returns:
            Tensor: Warped image or feature map.
        """

        assert x.size()[-2:] == flow.size()[1:3]
        _, _, h, w = x.size()
        # create mesh grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, h).type_as(x),
            torch.arange(0, w).type_as(x))
        grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
        grid.requires_grad = False

        vgrid = grid + flow
        # scale grid to [-1,1]
        vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
        vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
        vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
        output = F.grid_sample(x,
                               vgrid_scaled,
                               mode=interp_mode,
                               padding_mode=padding_mode,
                               align_corners=align_corners)

        return output

    def forward(self, dec_res_dict, inpfeat, encoder_features, fidelity_ratio=1.0):
        # dec_res_dict is multi-level feature from VQ-VAE decoder (Texture branch in the paper)
        x, offset = self.align_func_dict['Level_%d' % 2**(self.num_levels - 1)]( # TWM model
            dec_res_dict['Level_%d' % 2**(self.num_levels - 1)], inpfeat)
        
        x_2q = {}
        warp_feat_2k = {}
        warp_feat_2v = {}
        pre_flow = {}
        pre_offset = {}
        pre_similarity = {}
        Again_warppedfeat = {}
        warp_feat = {}
        upsample_offset = {}
        decode_x = {}
        upsample_x = {}

        upsample_x['4'] = self.pre_upsample_dict['Level_%d' % 2**4](x)
        for scale in reversed(range(self.num_levels - 1)):
            
            if scale < int(self.num_levels - 2):
                upsample_x[str(scale)] = self.pre_upsample_dict['Level_%d' % 2**scale](decode_x[str(scale+1)])
            upsample_offset['Level_%d' % 2**scale] = F.interpolate(offset, scale_factor=2, align_corners=False, mode='bilinear') * 2
            warp_feat['Level_%d' % 2**scale], offset = self.align_func_dict['Level_%d' % 2**scale](
                dec_res_dict['Level_%d' % 2**scale], inpfeat, previous_offset=upsample_offset['Level_%d' % 2**scale]) # offset is not used anymore
            # warp_feat = dec_res_dict['Level_%d' % 2**scale]
            
            # transdform warp_feat to k and v
            x_2q[str(scale)] = self.toq_dict['Level_%d' % 2**scale](upsample_x[str(scale)])
            warp_feat_2k[str(scale)] = self.tok_dict['Level_%d' % 2**scale](warp_feat['Level_%d' % 2**scale])
            warp_feat_2v[str(scale)] = self.tov_dict['Level_%d' % 2**scale](warp_feat['Level_%d' % 2**scale])
            
            pre_flow[str(scale)], pre_offset[str(scale)], pre_similarity[str(scale)] = self.flowsim['Level_%d' % 2**scale](x_2q[str(scale)], warp_feat_2k[str(scale)]) # estimate the image deformation field
            Again_warppedfeat[str(scale)] =  self.warpmodel['Level_%d' % 2**scale](pre_offset[str(scale)], pre_similarity[str(scale)], encoder_features[scale], pre_flow[str(scale)], warp_feat_2v[str(scale)])
            decode_x[str(scale)] = self.decoder_dict['Level_%d' % 2**scale](Again_warppedfeat[str(scale)])
            ''' 
            pre_relu_swapped_feat = self.flow_warp(warp_feat_2v, pre_flow)   
            down_relu_offset = torch.cat([encoder_features[scale], pre_relu_swapped_feat, warp_feat_2v], 1)
            down_relu_offset = self.lrelu(self.down_offset_conv1['Level_%d' % 2**scale](down_relu_offset))
            down_relu_offset = self.lrelu(self.down_offset_conv2['Level_%d' % 2**scale](down_relu_offset))
            
            down_relu_swapped_feat = self.lrelu(self.dyn_agg['Level_%d' % 2**scale]([warp_feat_2v,down_relu_offset],
                                    pre_offset, pre_similarity))
            
            x = self.decoder_dict['Level_%d' % 2**scale](torch.cat([x, down_relu_swapped_feat], dim=1))
            # x = self.decoder_dict['Level_%d' % 2**scale](torch.cat([x, warp_feat_2v], dim=1))
            '''
        
        return dec_res_dict['Level_1'] + fidelity_ratio * decode_x[str(scale)]


@ARCH_REGISTRY.register()
class VQFRv2trans(nn.Module):

    def __init__(self, base_channels, channel_multipliers, num_enc_blocks, use_enc_attention, num_dec_blocks,
                 use_dec_attention, code_dim, inpfeat_dim, code_selection_mode, align_opt, quantizer_opt):
        super().__init__()

        self.encoder = VQGANEncoder_features(
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_blocks=num_enc_blocks,
            use_enc_attention=use_enc_attention,
            code_dim=code_dim)

        if code_selection_mode == 'Nearest':
            self.feat2index = None
        elif code_selection_mode == 'Predict':
            self.feat2index = nn.Sequential(
                nn.LayerNorm(quantizer_opt['code_dim']), nn.Linear(quantizer_opt['code_dim'],
                                                                   quantizer_opt['num_code']))

        self.decoder = VQGANDecoder(
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_blocks=num_dec_blocks,
            use_dec_attention=use_dec_attention,
            code_dim=code_dim)

        self.main_branch = MainDecoder(
            base_channels=base_channels, channel_multipliers=channel_multipliers, align_opt=align_opt)
        self.inpfeat_extraction = nn.Conv2d(3, inpfeat_dim, 3, padding=1)

        self.quantizer = build_quantizer(quantizer_opt)

        self.apply(self._init_weights)

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

    def get_last_layer(self):
        return self.decoder.conv_out[-1].weight

    def forward(self, x_lq, fidelity_ratio=1.0):
        inp_feat = self.inpfeat_extraction(x_lq)
        res = {}
        enc_feat, features = self.encoder(x_lq)
        
        res['enc_feat'] = enc_feat
        
        if self.feat2index is not None:
            # cross-entropy predict token
            enc_feat = rearrange(enc_feat, 'b c h w -> b (h w) c')
            quant_logit = self.feat2index(enc_feat)
            res['quant_logit'] = quant_logit
            quant_index = quant_logit.argmax(-1) # torch.Size([2, 64])
            quant_feat = self.quantizer.get_feature(quant_index)
            
        else:
            # nearest predict token
            quant_feat, _, _ = self.quantizer(enc_feat)
        with torch.no_grad():
            texture_dec, texture_feat_dict = self.decoder(quant_feat, return_feat=True) # VQ-VAE decoder
            res['texture_dec'] = texture_dec
        
        main_feature = self.main_branch(texture_feat_dict, inp_feat, encoder_features=features, fidelity_ratio=fidelity_ratio) # main decoder for image restoration
        main_dec = self.decoder.conv_out(main_feature)
        
        res['main_dec'] = main_dec
        return res
