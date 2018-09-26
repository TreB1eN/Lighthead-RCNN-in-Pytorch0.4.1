import torch
from torch.nn import functional as F
import pdb

def smooth_l1_loss(inputs, targets, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (inputs - targets)
    abs_diff = diff.abs().detach()
    flag = (abs_diff < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) + (1 - flag) * (abs_diff - 0.5 / sigma2))
    """
    论文里没见有这个sigma参数啊，
    sigma = 1时就说论文里原来的smooth_l1_loss
    当sigma > 1时，相当于变相的把关注的区间变窄了
    关注的权重放大了
    """
    return torch.sum(y, dim=1)

def fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma = 3., reduce='mean'):
    in_weight = torch.zeros_like(pred_loc)
    # Localization loss is calculated only for positive rois.
    in_weight[torch.tensor(gt_label, dtype=torch.long) > 0] = 1
    loc_loss = smooth_l1_loss(pred_loc, gt_loc, in_weight, sigma)
    # Normalize by total number of negtive and positive rois.
    if reduce == 'mean':
        loc_loss = torch.sum(loc_loss) / torch.sum(gt_label >= 0).to(torch.float)
    elif reduce != 'no':
        warnings.warn('no reduce option: {}'.format(reduce))
    return loc_loss

def OHEM_loss(roi_cls_locs,
               roi_scores,
               gt_roi_locs,
               gt_roi_labels,
               n_ohem_sample = 256,
               roi_sigma = 1.0):
    n_sample = roi_cls_locs.shape[0]
    roi_cls_locs = roi_cls_locs.reshape((n_sample, -1, 4))
    roi_locs = roi_cls_locs[torch.arange(n_sample, dtype=torch.long), gt_roi_labels]
    roi_loc_loss = fast_rcnn_loc_loss(roi_locs, gt_roi_locs, gt_roi_labels, roi_sigma, reduce='no')
    roi_cls_loss = F.cross_entropy(roi_scores, gt_roi_labels, reduce=False)
    assert roi_loc_loss.shape == roi_cls_loss.shape

    n_ohem_sample = min(n_ohem_sample, n_sample)
    total_roi_loc_loss = roi_loc_loss.detach()
    total_roi_cls_loss = roi_cls_loss.detach()
    roi_cls_loc_loss = total_roi_loc_loss + total_roi_cls_loss
    _, indices = roi_cls_loc_loss.sort(descending=True)
    indices = indices[:n_ohem_sample]
    # indices = cuda.to_gpu(indices)
    roi_loc_loss = torch.sum(roi_loc_loss[indices]) / n_ohem_sample
    roi_cls_loss = torch.sum(roi_cls_loss[indices]) / n_ohem_sample

    return roi_loc_loss, roi_cls_loss, total_roi_loc_loss.mean().item(), total_roi_cls_loss.mean().item()