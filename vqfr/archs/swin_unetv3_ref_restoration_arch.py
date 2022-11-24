import vqfr.archs.arch_util as arch_util
import torch
import torch.nn as nn
import torch.nn.functional as F
# from datsr.models.archs.DCNv2.dcn_v2 import DCN_sep_pre_multi_offset_flow_similarity as DynAgg
from vqfr.archs.dcn_v2 import DCN_sep_pre_multi_offset_flow_similarity as DynAgg
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import pdb


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

class ContentExtractor(nn.Module):

    def __init__(self, in_nc=3, out_nc=3, nf=64, n_blocks=16):
        super(ContentExtractor, self).__init__()

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.body = arch_util.make_layer(
            arch_util.ResidualBlockNoBN, n_blocks, nf=nf)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        arch_util.default_init_weights([self.conv_first], 0.1)

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        feat = self.body(feat)

        return feat

class DynamicAggregationRestoration(nn.Module):

    def __init__(self,
                 in_channel,
                 out_channel,
                 groups=8,
                 ):
        super(DynamicAggregationRestoration, self).__init__()
        
        # dynamic aggregation module for relu1_1 reference feature
        self.down_large_offset_conv1 = nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=True)
        self.down_large_offset_conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=True)
        self.down_large_dyn_agg = DynAgg(out_channel, out_channel, 3, stride=1, padding=1, dilation=1,
                                    deformable_groups=groups, extra_offset_mask=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
    def flow_warp(self,
                  x,
                  flow,
                  interp_mode='bilinear',
                  padding_mode='zeros',
                  align_corners=True):
        
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

    def forward(self, pre_offset, pre_similarity, encoder_features, pre_flow, warp_feat_2v):
        
        pre_relu_swapped_feat = self.flow_warp(warp_feat_2v, pre_flow)
        
        down_relu_offset = torch.cat([encoder_features, pre_relu_swapped_feat, warp_feat_2v], 1)
        down_relu_offset = self.lrelu(self.down_large_offset_conv1(down_relu_offset))
        down_relu_offset = self.lrelu(self.down_large_offset_conv2(down_relu_offset))
        
        down_relu_swapped_feat = self.lrelu( # the "A" in the paper
            self.down_large_dyn_agg([warp_feat_2v, down_relu_offset],
                               pre_offset, pre_similarity))

        x = torch.cat([encoder_features, down_relu_swapped_feat], 1) # "rfa" in the paper
        
        return x