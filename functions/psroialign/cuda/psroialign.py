import math
from torch import nn
from torch.autograd import Function
import torch

import psroialign_cuda

class PSRoIAlign(nn.Module):
    def __init__(self, spatial_scale, roi_size, sampling_ratio, pooled_dim):
        super(PSRoIAlign, self).__init__()
        self.spatial_scale = spatial_scale
        self.roi_size = roi_size
        self.sampling_ratio = sampling_ratio
        self.pooled_dim = pooled_dim

    def forward(self, bottom_data, bottom_rois):
        return PSRoIAlignFunction.apply(bottom_data, 
                                        bottom_rois, 
                                        self.spatial_scale, 
                                        self.roi_size, 
                                        self.sampling_ratio, 
                                        self.pooled_dim)

class PSRoIAlignFunction(Function):
    @staticmethod
    def forward(ctx, bottom_data, bottom_rois, spatial_scale, roi_size, sampling_ratio, pooled_dim):
        ctx.spatial_scale = spatial_scale
        ctx.roi_size = roi_size
        ctx.sampling_ratio = sampling_ratio
        ctx.pooled_dim = pooled_dim
        ctx.feature_size = bottom_data.size()
        num_rois = bottom_rois.size(0)
        top_data = torch.zeros([num_rois, pooled_dim, roi_size, roi_size], dtype=torch.float32).to(bottom_data.device)
        argmax_data = torch.zeros([num_rois, pooled_dim, roi_size, roi_size], dtype=torch.int32).to(bottom_data.device)
        if bottom_data.is_cuda:
            psroialign_cuda.forward(bottom_data, bottom_rois, top_data, argmax_data, spatial_scale, roi_size, sampling_ratio)
            ctx.save_for_backward(bottom_rois, argmax_data)
        else:
            raise NotImplementedError

        return top_data

    @staticmethod
    def backward(ctx, top_diff):
        spatial_scale = ctx.spatial_scale
        roi_size = ctx.roi_size
        sampling_ratio = ctx.sampling_ratio
        pooled_dim = ctx.pooled_dim                
        batch_size, channels, height, width = ctx.feature_size
        [bottom_rois, argmax_data] = ctx.saved_tensors
        bottom_diff = None
        if ctx.needs_input_grad[0]:
            bottom_diff = torch.zeros([batch_size, channels, height, width], dtype=torch.float32).to(top_diff.device)
            psroialign_cuda.backward(top_diff, argmax_data, bottom_rois, bottom_diff, spatial_scale, roi_size, sampling_ratio)

        return bottom_diff, None, None, None, None, None