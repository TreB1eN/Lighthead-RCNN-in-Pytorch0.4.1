import torch
from torch.nn import Conv2d, Linear, Module
from torch.nn import functional as F
from functions.psroi_pooling.modules.psroi_pool import PSRoIPool
from utils.utils import normal_init

class GlobalContextModule(Module):
    def __init__(self, in_channels, mid_channels, out_channels, ksize):
        super(GlobalContextModule, self).__init__()
        padsize = int((ksize - 1) / 2)
        self.col_max = Conv2d(in_channels, mid_channels, (ksize, 1), 1, (padsize, 0))
        self.col = Conv2d(mid_channels, out_channels, (1, ksize), 1, (0, padsize))
        self.row_max = Conv2d(in_channels, mid_channels, (1, ksize), 1, (0, padsize))
        self.row = Conv2d(mid_channels, out_channels, (ksize, 1), 1, (padsize, 0))

    def __call__(self, x):
        h_col = self.col(self.col_max(x))
        h_row = self.row(self.row_max(x))
        return F.relu(h_col + h_row)

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class LightHeadRCNNResNet101_Head(Module):
    def __init__(self,
                 n_class = 81,
                 roi_size = 7,
                 spatial_scale = 1/16.):
        super(LightHeadRCNNResNet101_Head, self).__init__()
        self.n_class = n_class
        self.spatial_scale = spatial_scale
        self.roi_size = roi_size
        self.global_context_module = GlobalContextModule(2048,256,490,15)
        self.psroi_max_align_pooling = PSRoIPool(roi_size, roi_size, spatial_scale, group_size=roi_size, output_dim=10)
        self.flatten = Flatten()
        self.fc1 = Linear(self.roi_size * self.roi_size * 10, 2048)
        self.score = Linear(2048, n_class)
        self.cls_loc = Linear(2048, 4 * n_class)
        self.apply(lambda x : normal_init(x, 0, 0.01))

    def __call__(self, x, rois):
        # global context module
        device = x.device
        h = self.global_context_module(x)
        # psroi max align
        func_roi = torch.cat((torch.zeros([rois.shape[0],1], device=device),torch.tensor(rois).to(device)), dim=1)
        pool = self.psroi_max_align_pooling(h, func_roi)
        pool = self.flatten(pool)
        # fc
        fc1 = F.relu(self.fc1(pool))
        roi_cls_locs = self.cls_loc(fc1)
        roi_scores = self.score(fc1)
        return roi_cls_locs, roi_scores