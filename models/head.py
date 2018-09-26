import torch
from torch.nn import Conv2d, Linear, Module
from torch.nn import functional as F
from functions.psroi_pooling.modules.psroi_pool import PSRoIPool
from functions.roi_align.modules.roi_align import RoIAlignMax
from utils.utils import normal_init
from functions.RoIAlign_pytorch.roi_align.roi_align import RoIAlign

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
                 spatial_scale = 1/16.,
                 roi_align = False):
        super(LightHeadRCNNResNet101_Head, self).__init__()
        self.n_class = n_class
        self.spatial_scale = spatial_scale
        self.roi_size = roi_size
        self.global_context_module = GlobalContextModule(2048,256,490,15)
        self.roi_align = roi_align
        self.flatten = Flatten()
        self.fc1 = Linear(self.roi_size * self.roi_size * 10, 2048)
        self.score = Linear(2048, n_class)
        self.cls_loc = Linear(2048, 4 * n_class)
        self.apply(lambda x : normal_init(x, 0, 0.01))
        if self.roi_align:
            self.pooling = RoIAlignMax(self.roi_size, self.roi_size, self.spatial_scale, 2)
            self.conv1x1 = Conv2d(self.roi_size * self.roi_size * 10, 10, 1, bias = False)

    def __call__(self, x, rois):
        # global context module
        device = x.device
        h = self.global_context_module(x)
        if self.roi_align:
            func_roi = torch.cat((torch.zeros([rois.shape[0],1], device=device),torch.tensor(rois).to(device)), dim=1) 
            pool = self.pooling(h, func_roi)
            pool = self.conv1x1(pool)
        else:
            # psroi max align
            pool = position_sensitive_roi_align_pooling(h, 
                                                 torch.tensor(rois).to(device), 
                                                 torch.zeros([rois.shape[0]], device=device),
                                                 crop_size=(roi_size*2, roi_size*2), 
                                                 num_spatial_bins=(roi_size, roi_size),
                                                 scale = self.spatial_scale,
                                                 out_dim = 10)                                                 
        pool = self.flatten(pool)
        # fc
        fc1 = F.relu(self.fc1(pool))
        roi_cls_locs = self.cls_loc(fc1)
        roi_scores = self.score(fc1)
        return roi_cls_locs, roi_scores

def position_sensitive_roi_align_pooling(features, 
                                         boxes,
                                         box_image_indices, 
                                         crop_size=(14, 14), 
                                         num_spatial_bins=(7, 7),
                                         scale = 1/16.,
                                         out_dim = 10):
    """
    Arguments:
        features: a float tensor with shape [1, height, width, depth].
        scale: image and box resize scale, float
        out_dim: output dimension, int
        boxes: a float tensor with shape [num_boxes, 4]. The i-th row of the tensor
            specifies the coordinates of a box in the `box_image_indices[i]` image
            and is specified in unnormalized coordinates [x1, y1, x2, y2].
        box_image_indices: an int tensor with shape [num_boxes]. It has values in range [0, batch).
        crop_size: a tuple with two integers (crop_height, crop_width).
        num_spatial_bins: a tuple with two integers (spatial_bins_y, spatial_bins_x).
            Represents the number of position-sensitive bins in y and x directions.
            Both values should be >= 1. `crop_height` should be
            divisible by `spatial_bins_y`, and similarly for width.
            The number of `features` channels should be divisible by (spatial_bins_y * spatial_bins_x).
    Returns:
        a float tensor with shape [num_boxes, spatial_bins_y * spatial_bins_x, crop_channels],
            where `crop_channels = depth/(spatial_bins_y * spatial_bins_x)`.
    """
    total_bins = 1
    bin_crop_size = []

    size = [features.shape[3], features.shape[2]]
    size_tensor = torch.tensor(size + size, dtype=torch.float)
    boxes *= scale
    boxes /= size_tensor
    # box都归一化到【0,1】

    for num_bins, crop_dim in zip(num_spatial_bins, crop_size):
        assert num_bins >= 1
        assert crop_dim % num_bins == 0
        total_bins *= num_bins
        bin_crop_size.append(crop_dim // num_bins)

    crop_resize = CropAndResizeFunction(*bin_crop_size)

    depth = features.shape[1]
    assert depth == total_bins * out_dim, 'input dimensions should equal to out_dim * total_bins_number'    

    xmin, ymin, xmax, ymax = torch.unbind(boxes, dim=1)
    # 输入的boxes格式是x1y1x2y2 !
    spatial_bins_x, spatial_bins_y = num_spatial_bins
    step_x = (xmax - xmin) / spatial_bins_x
    step_y = (ymax - ymin) / spatial_bins_y


    # split each box into `total_bins` bins
    position_sensitive_boxes = []
    for bin_y in range(spatial_bins_y):
        for bin_x in range(spatial_bins_x):
            box_coordinates = [
                ymin + bin_y * step_y,
                xmin + bin_x * step_x,
                ymin + (bin_y + 1) * step_y,
                xmin + (bin_x + 1) * step_x,
            ]
            # 这里制造的box的格式是y1x1y2x2 !
            position_sensitive_boxes.append(torch.stack(box_coordinates, dim=1))

    feature_splits = torch.split(features, out_dim, dim=1)
    # it a list of float tensors with
    # shape [batch_size, image_height, image_width, out_dim]
    # and it has length `total_bins`

    feature_crops = []
    for split, box in zip(feature_splits, position_sensitive_boxes):
        crop = crop_resize(split, box, box_image_indices)
        # 这个函数输入格式是y1x1y2x2 !!!
        # shape [num_boxes, crop_height/spatial_bins_y, crop_width/spatial_bins_x, depth/total_bins]
        # do max pooling over spatial positions within the bin
        crop, _ = torch.max(crop.view(crop.shape[0], crop.shape[1], -1), dim=-1)
        # shape [num_boxes, 1, depth/total_bins]
        feature_crops.append(crop.unsqueeze(1))
    return torch.cat(feature_crops, dim=1)