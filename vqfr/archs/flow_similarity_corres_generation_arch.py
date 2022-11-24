import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from vqfr.archs.arch_util import tensor_shift
from vqfr.archs.ref_map_util import feature_match_index
import pdb
logger = logging.getLogger('base')


class FlowSimCorrespondenceGenerationArch(nn.Module):

    def __init__(self,
                 patch_size=3,
                 stride=1):
        super(FlowSimCorrespondenceGenerationArch, self).__init__()
        self.patch_size = patch_size
        self.stride = stride

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

    def forward(self, features1, features2):
        batch_offset_relu = []
        flows_relu = []
        similarity_relu = []
        
        for ind in range(features1.size(0)):  # [9, 3, 160, 160] # traverse a batch. when test img_ref_hr.size(0)=1
            feat_in = features1[ind]    # [256, 40, 40]
            feat_ref = features2[ind]   # [256, 40, 40]
            c, h, w = feat_in.size()
            feat_in = F.normalize(feat_in.reshape(c, -1), dim=0).view(c, h, w) # normlize for multiplation, make l2-norm = 1
            feat_ref = F.normalize(
                feat_ref.reshape(c, -1), dim=0).view(c, h, w)

            _max_idx, _max_val = feature_match_index(   # [38, 38], [38, 38] most corresponding patches and distance ([78, 118])
                feat_in,
                feat_ref,
                patch_size=self.patch_size,
                input_stride=self.stride,
                ref_stride=self.stride,
                is_norm=True,
                norm_input=True)
            
            # similarity for relu3_1
            sim_relu = F.pad(_max_val.clone(), (1,1,1,1)).unsqueeze(0)
            similarity_relu.append(sim_relu)
            # offset map for relu3_1
            offset_relu = self.index_to_flow(_max_idx.clone())   # [1, 40, 40, 2] how long should each pixel move through x and y axis
            flows_relu.append(offset_relu)
            # shift offset relu3
            shifted_offset_relu = []
            for i in range(0, 3):
                for j in range(0, 3):
                    # flow_shift = tensor_shift(offset_relu, (i, j))  # [1, 40, 40, 2]
                    # shifted_offset_relu.append(flow_shift)
                    shifted_offset_relu.append(tensor_shift(offset_relu, (i, j)))
            shifted_offset_relu = torch.cat(shifted_offset_relu, dim=0)  # [9, 40, 40, 2]
            batch_offset_relu.append(shifted_offset_relu)

        # size: [b, 9, h, w, 2], the order of the last dim: [x, y]
        batch_offset_relu = torch.stack(batch_offset_relu, dim=0) # [1, 9, 80, 120, 2]
        
        # flows
        pre_flow = torch.cat(flows_relu, dim=0) # [1, 80, 120, 2]
        
        pre_offset = batch_offset_relu  # [9, 9, 160, 160, 2]
        
        # similarity
        pre_similarity = torch.stack(similarity_relu, dim=0)
        
        return pre_flow, pre_offset, pre_similarity
