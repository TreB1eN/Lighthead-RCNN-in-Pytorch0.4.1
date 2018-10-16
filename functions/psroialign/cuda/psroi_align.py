import math
from torch import nn
from torch.autograd import Function
import torch

import psroi_align_cuda

class PS_roi_align(nn.Module):
    def __init__(self, spatial_scale, roi_size, sampling_ratio, pooled_dim):
        super(PS_roi_align, self).__init__()
        self.spatial_scale = spatial_scale
        self.roi_size = roi_size
        self.sampling_ratio = sampling_ratio
        self.pooled_dim = pooled_dim

    def forward(self, bottom_data, bottom_rois):
        return PSRoIAlignFunction(self.spatial_scale, self.roi_size, self.sampling_ratio, self.pooled_dim)(bottom_data, bottom_rois)

class PSRoIAlignFunction(Function):
    def __init__(self, spatial_scale, roi_size, sampling_ratio, pooled_dim):
        self.roi_size = int(roi_size)
        self.pooled_dim = int(pooled_dim)
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)
        self.bottom_rois = None
        self.feature_size = None
        self.argmax_data = None

    def forward(self, bottom_data, bottom_rois):
        self.bottom_rois = bottom_rois
        self.feature_size = bottom_data.size()

        num_rois = bottom_rois.size(0)

        top_data = torch.zeros([num_rois, self.pooled_dim, self.roi_size, self.roi_size], dtype=torch.float).to(bottom_data.device)
        argmax_data = torch.zeros([num_rois, self.pooled_dim, self.roi_size, self.roi_size], dtype=torch.int32).to(bottom_data.device)

        if bottom_data.is_cuda:
            psroi_align_cuda.forward(bottom_data, bottom_rois, top_data, argmax_data, self.spatial_scale, self.roi_size, self.sampling_ratio)
            self.argmax_data = argmax_data
        else:
            raise NotImplementedError

        return top_data

    def backward(self, top_diff):
        assert(self.feature_size is not None and top_diff.is_cuda)

        batch_size, channels, height, width = self.feature_size

        bottom_diff = torch.zeros([batch_size, channels, height, width], dtype=torch.float).to(top_diff.device)

        psroi_align_cuda.backward(top_diff, self.argmax_data, self.bottom_rois, bottom_diff, self.spatial_scale, self.roi_size, self.sampling_ratio)

        return bottom_diff, None