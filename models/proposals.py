import numpy as np
import pdb
from utils.bbox_tools import bbox2loc, bbox_iou, loc2bbox, enumerate_shifted_anchor, xywh_2_x1y1x2y2
from utils.nms import nms_cpu
from functions.nms.nms_wrapper import nms
import torch

class ProposalTargetCreator(object):
    """Assign ground truth bounding boxes to given RoIs.

    The :meth:`__call__` of this class generates training targets
    for each object proposal.
    This is used to train Faster RCNN [#]_.

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        n_sample (int): The number of sampled regions.
        pos_ratio (float): Fraction of regions that is labeled as a
            foreground.
        pos_iou_thresh (float): IoU threshold for a RoI to be considered as a
            foreground.
        neg_iou_thresh_hi (float): RoI is considered to be the background
            if IoU is in
            [:obj:`neg_iou_thresh_hi`, :obj:`neg_iou_thresh_hi`).
        neg_iou_thresh_lo (float): See above.

    """

    def __init__(self,
                 n_sample = -1,
                 pos_ratio = 0.25,
                 pos_iou_thresh = 0.5,
                 neg_iou_thresh_hi = 0.5,
                 neg_iou_thresh_lo = 0.0,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo  # NOTE: py-faster-rcnn默认的值是0.1
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std

    def __call__(self, roi, bbox, label):
        """Assigns ground truth to sampled proposals.

        This function samples total of :obj:`self.n_sample` RoIs
        from the combination of :obj:`roi` and :obj:`bbox`.
        The RoIs are assigned with the ground truth class labels as well as
        bounding box offsets and scales to match the ground truth bounding
        boxes. As many as :obj:`pos_ratio * self.n_sample` RoIs are
        sampled as foregrounds.

        Offsets and scales of bounding boxes are calculated using
        :func:`model.utils.bbox_tools.bbox2loc`.
        Also, types of input arrays and output arrays are same.

        Here are notations.

        * :math:`S` is the total number of sampled RoIs, which equals :obj:`self.n_sample`.
        * :math:`L` is number of object classes possibly including the background.

        Args:
            roi (array): Region of Interests (RoIs) from which we sample.
                Its shape is :math:`(R, 4)`
            bbox (array): The coordinates of ground truth bounding boxes.
                Its shape is :math:`(R', 4)`.
            label (array): Ground truth bounding box labels. Its shape
                is :math:`(R',)`. Its range is :math:`[0, L - 1]`, where
                :math:`L` is the number of foreground classes.
            loc_normalize_mean (tuple of four floats): Mean values to normalize
                coordinates of bouding boxes.
            loc_normalize_std (tupler of four floats): Standard deviation of
                the coordinates of bounding boxes.

        Returns:
            (array, array, array):

            * **sample_roi**: Regions of interests that are sampled. \
                Its shape is :math:`(S, 4)`.
            * **gt_roi_loc**: Offsets and scales to match \
                the sampled RoIs to the ground truth bounding boxes. \
                Its shape is :math:`(S, 4)`.
            * **gt_roi_label**: Labels assigned to sampled RoIs. Its shape is \
                :math:`(S,)`. Its range is :math:`[0, L]`. The label with \
                value 0 is the background.

        """
        """
        RPN会产生大约2000个RoIs，这2000个RoIs不是都拿去训练，
        而是利用ProposalTargetCreator 选择128个RoIs用以训练。选择的规则如下：
        RoIs和gt_bboxes 的IoU大于0.5的，选择一些（比如32个）
        选择 RoIs和gt_bboxes的IoU小于0.5，同时大于等于0（或者0.1）的选择一些（比如 128-32=96个）作为负样本
        为了便于训练，对选择出的128个RoIs，还对他们的gt_roi_loc 进行标准化处理（减去均值除以标准差）
        """
        n_bbox, _ = bbox.shape
        roi = np.concatenate((roi, bbox), axis=0)
        ''' 
        这一步是人为地把最正确的答案bbox也加入到roi当中来，毫无疑问这些bbox肯定是概率最高的（1.）正样本，作为一个很好的监督信号。
        '''
        
        if self.n_sample > 0:
            n_sample = self.n_sample
        else:
            n_sample = len(roi)

        pos_roi_per_image = np.round(n_sample * self.pos_ratio)
        # 正样本的数量默认是 128 * 0.25 = 32
        iou = bbox_iou(roi, bbox) 
        # 每一个roi和每一个gt_bbox(不止一个gt object)计算iou, 这个时候roi格式已经是x1y1x2y2了，bbox格式是xywh
        gt_assignment = iou.argmax(axis=1) # 
        max_iou = iou.max(axis=1)
        # Offset range of classes from [0, n_fg_class - 1] to [1, n_fg_class].
        # The label with value 0 is the background.
        gt_roi_label = label[gt_assignment] + 1
        # 加1是因为背景是0

        # Select foreground RoIs as those with >= pos_iou_thresh IoU.
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))
        # 如果所有正样本加起来也没有32个，那也只能有多少用多少了
        if pos_index.size > 0:
            pos_index = np.random.choice(pos_index, size=pos_roi_per_this_image, replace=False)
            # 从pos_index,即正样本的index中抽取pos_roi_per_this_image个出来

        # Select background RoIs as those within
        # [neg_iou_thresh_lo, neg_iou_thresh_hi).
        # 接下来处理负样本,和正样本的提取没太大区别
        neg_index = np.where((max_iou < self.neg_iou_thresh_hi) & (max_iou >= self.neg_iou_thresh_lo))[0]
        neg_roi_per_this_image = n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(neg_index, size=neg_roi_per_this_image, replace=False)

        # The indices that we're selecting (both positive and negative).
        keep_index = np.append(pos_index, neg_index)
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0  # negative labels --> 0
        sample_roi = roi[keep_index]
        
        # Compute offsets and scales to match sampled RoIs to the GTs.
        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        gt_roi_loc = (gt_roi_loc - np.array(self.loc_normalize_mean, np.float32)) / np.array(self.loc_normalize_std, np.float32)
        # 通过bbox2loc算出来正确的偏移量，还要做一下归一化
        # 这个均值和方差是哪来的 ？
        """
        sample_roi.shape, gt_roi_loc.shape, gt_roi_label.shape = 
        ((128, 4), (128, 4), (128,))
        """
        return sample_roi, gt_roi_loc, gt_roi_label


class AnchorTargetCreator(object):
    """Assign the ground truth bounding boxes to anchors.

    Assigns the ground truth bounding boxes to anchors for training Region
    Proposal Networks introduced in Faster R-CNN [#]_.

    Offsets and scales to match anchors to the ground truth are
    calculated using the encoding scheme of
    :func:`model.utils.bbox_tools.bbox2loc`.

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        n_sample (int): The number of regions to produce.
        pos_iou_thresh (float): Anchors with IoU above this
            threshold will be assigned as positive.
        neg_iou_thresh (float): Anchors with IoU below this
            threshold will be assigned as negative.
        pos_ratio (float): Ratio of positive regions in the
            sampled regions.

    """

    def __init__(self,
                 n_sample=256,
                 pos_iou_thresh=0.7,
                 neg_iou_thresh=0.3,
                 pos_ratio=0.5):
        self.n_sample = n_sample # 256
        self.pos_iou_thresh = pos_iou_thresh # 0.7
        self.neg_iou_thresh = neg_iou_thresh # 0.3
        self.pos_ratio = pos_ratio # 0.5

    def __call__(self, bbox, anchor, img_size):
        """
        example:
        bbox.shape , anchor.shape, img_size :
        ((4, 4), (16650, 4), (600, 800))
        
        注意这里是拿预定义的anchor来和gt比较计算的，
        rpn网络的预测就是原来的anchor,
        先修正一次。
        后面roi loss还会再修正一次
        """
        """Assign ground truth supervision to sampled subset of anchors.

        Types of input arrays and output arrays are same.

        Here are notations.

        * :math:`S` is the number of anchors.
        * :math:`R` is the number of bounding boxes.

        Args:
            bbox (array): Coordinates of bounding boxes. Its shape is
                :math:`(R, 4)`.
            anchor (array): Coordinates of anchors. Its shape is
                :math:`(S, 4)`.
            img_size (tuple of ints): A tuple :obj:`H, W`, which
                is a tuple of height and width of an image.

        Returns:
            (array, array):

            #NOTE: it's scale not only  offset
            * **loc**: Offsets and scales to match the anchors to \
                the ground truth bounding boxes. Its shape is :math:`(S, 4)`.
            * **label**: Labels of anchors with values \
                :obj:`(1=positive, 0=negative, -1=ignore)`. Its shape \
                is :math:`(S,)`.

        """

        img_H, img_W = img_size
        # (600, 800)
        n_anchor = len(anchor) # for example:16650
        inside_index = _get_inside_index(anchor, img_H, img_W)
        anchor = anchor[inside_index]
        if len(bbox) == 0:
            label = np.empty((len(inside_index), ), dtype=np.int32)
            label.fill(-1)
            label[np.random.choice(range(len(inside_index)), self.n_sample, replace=False)] = 0
            label = _unmap(label, n_anchor, inside_index, fill=-1)
            return [], label
        # 上面两步把一些超出图片之外的anchor过滤掉
        argmax_ious, label = self._create_label(inside_index, anchor, bbox)
        # argmax_ious每一个anchor最匹配的gt object的序号，label是根据它得出来的标记（-1,0,1）
        # 1是正样本，0是负样本，-1表示不关心，不参与后续计算
        # 正负样本之和应该是self.n_sample
        
        # compute bounding box regression targets
        loc = bbox2loc(anchor, bbox[argmax_ious])
        # bbox[argmax_ious] 就是每一个anchor对应的最匹配的gt_bbox了
        # 这里有点浪费了，因为不是每一个anchor都要算loc的

        # map up to original set of anchors
        """
        记得inside_index吗？ 
        因为后续计算还是会用所有的anchor做输入,
        所以这里在求出了loc和label之后，
        还要映射回anchor原来的尺寸
        """
        label = _unmap(label, n_anchor, inside_index, fill=-1)
        loc = _unmap(loc, n_anchor, inside_index, fill=0)
        """
        所以总的来说，AnchorTargetCreator做的事情是：
        根据每一个预先设定的anchor和这张图片的gt_bbox去计算iou,
        再用求得的iou来给每一个anchor打标签，
        1是正样本，0是负样本，-1表示不关心，不参与后续计算
        打标签是通过
        正负样本之和应该是self.n_sample，比例是self.pos_ratio
        打标签的依据是：
        1. iou < 0.3的都算负样本
        2. 对每一个gt_object，标记和它iou最高的的anchor为正样本
            可能同时有多个anchor同时iou最高（相等）
        3. 剩下的anchor里面，iou大于0.7的也算正样本
        4. 还要平衡一下正负样本的数量和比例
        
        它不但打标签，还会计算每一个anchor和它最匹配的gt_bbox的loc,
        用于后续的bbox回归loss计算
        最后，返回的是loc和label # ((16650,), (16650, 4))
        """
        return loc, label # ((16650,), (16650, 4))

    def _create_label(self, inside_index, anchor, bbox):
        # label: 1 is positive, 0 is negative, -1 is dont care
        label = np.empty((len(inside_index), ), dtype=np.int32)
        label.fill(-1)
        
        argmax_ious, max_ious, gt_argmax_ious = self._calc_ious(anchor, bbox, inside_index)
        """
        这个函数计算每一个anchor和每一个bbox的的iou,
        然后返回每一个anchor最匹配的gt object的序号（argmax_ious），
        以及从大到小的iou值（max_ious），
        还有与每一个object的iou最大的anchor的序号列表
        （gt_argmax_ious，最大iou可能重复，所以gt_argmax_ious的size可能比gt object数目多）
        """

        # assign negative labels first so that positive labels can clobber them
        label[max_ious < self.neg_iou_thresh] = 0
        # 对rpn来说，iou < 0.3的都算负样本，
        # 注意这里negative mining的方式和roi head里不一样，你要想清楚为什么

        # positive label: for each gt, anchor with highest iou
        label[gt_argmax_ious] = 1
        # 对每一个gt_object，标记和它iou最高的的anchor为正样本
        # 可能同时有多个anchor同时iou最高（相等）

        # positive label: above threshold IOU
        label[max_ious >= self.pos_iou_thresh] = 1
        # 剩下的anchor里面，iou大于0.7的也算正样本
        
        #接下来平衡一下正负样本的数量、、、
        # 正样本如果多了，砍掉一些（如果少了，就算了）
        # 有了正样本数量，剩下的就都是负样本了，
        # 一样也是多了就砍掉，少了就算了，不过应该不会少。。。

        # subsample positive labels if we have too many
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1

        # subsample negative labels if we have too many
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        return argmax_ious, label
        # argmax_ious是ious从大到小排的序号，label是根据它得出来的标记（-1,0,1）

    def _calc_ious(self, anchor, bbox, inside_index):
        """
        这个函数计算每一个anchor和每一个bbox的的iou,
        然后返回每一个anchor最匹配的gt object的序号（argmax_ious），
        以及从大到小的iou值（max_ious），
        还有与每一个object的iou最大的anchor的序号列表
        （gt_argmax_ious，最大iou可能重复，所以gt_argmax_ious的size可能比gt object数目多）
        """
        # ious between the anchors and the gt boxes
        ious = bbox_iou(anchor, bbox)
        """
        (Pdb) anchor.shape
        (5834, 4)
        (Pdb) bbox.shape
        (3, 4)
        ious.shape
        (5834,3)
        """
        argmax_ious = ious.argmax(axis=1) # (5834,)
        max_ious = ious[np.arange(len(inside_index)), argmax_ious] # (5834,)
        gt_argmax_ious = ious.argmax(axis=0)
        # (3,),这张图片有3个gt object，分别对应的最大iou的anchor在第几行
        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
        # (3,),拿上面得出的idx，求出每一个gt object对应的最大iou的值
        gt_argmax_ious = np.where(ious == gt_max_ious)[0]
        # 可能存在两个anchor和一个gt bbox有着相同iou的情况，这里重新取一次index,
        # 比如 gt_argmax_ious.shape = (14,)
        # 注意这里只需要知道是哪个anchor蒙中了，
        # 我都不关心蒙中了哪个class
        # 因为rpn的loss计算的只是有没有object,不关心哪一个class

        return argmax_ious, max_ious, gt_argmax_ious


def _unmap(data, count, index, fill=0):
    # Unmap a subset of item (data) back to the original set of items (of
    # size count)

    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=data.dtype)
        ret.fill(fill)
        ret[index] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[index, :] = data
    return ret


def _get_inside_index(anchor, H, W):
    # Calc indicies of anchors which are located completely inside of the image
    # whose size is speficied.
    index_inside = np.where((anchor[:, 0] >= 0) & (anchor[:, 1] >= 0) &
                            (anchor[:, 2] <= W) & (anchor[:, 3] <= H))[0]
    return index_inside


class ProposalCreator:
    # unNOTE: I'll make it undifferential
    # unTODO: make sure it's ok
    # It's ok
    """Proposal regions are generated by calling this object.

    The :meth:`__call__` of this object outputs object detection proposals by
    applying estimated bounding box offsets
    to a set of anchors.

    This class takes parameters to control number of bounding boxes to
    pass to NMS and keep after NMS.
    If the paramters are negative, it uses all the bounding boxes supplied
    or keep all the bounding boxes returned by NMS.

    This class is used for Region Proposal Networks introduced in
    Faster R-CNN [#]_.

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        nms_thresh (float): Threshold value used when calling NMS.
        n_train_pre_nms (int): Number of top scored bounding boxes
            to keep before passing to NMS in train mode.
        n_train_post_nms (int): Number of top scored bounding boxes
            to keep after passing to NMS in train mode.
        n_test_pre_nms (int): Number of top scored bounding boxes
            to keep before passing to NMS in test mode.
        n_test_post_nms (int): Number of top scored bounding boxes
            to keep after passing to NMS in test mode.
        force_cpu_nms (bool): If this is :obj:`True`,
            always use NMS in CPU mode. If :obj:`False`,
            the NMS mode is selected based on the type of inputs.
        min_size (int): A paramter to determine the threshold on
            discarding bounding boxes based on their sizes.

    """

    def __init__(
            self,
            parent_model,
            nms_thresh=0.7,
            n_train_pre_nms=12000,
            n_train_post_nms=2000,
            n_test_pre_nms=6000,
            n_test_post_nms=1000,
            min_size=16  # 这里的min_size是指原图中的尺寸，如果预测出来的框映射回原图比16*16还小，就丢弃了
    ):
        self.parent_model = parent_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, loc, score, anchor, img_size, scale=1.):
        """input should  be ndarray
        Propose RoIs.

        Inputs :obj:`loc, score, anchor` refer to the same anchor when indexed
        by the same index.

        On notations, :math:`R` is the total number of anchors. This is equal
        to product of the height and the width of an image and the number of
        anchor bases per pixel.

        Type of the output is same as the inputs.

        Args:
            loc (array): Predicted offsets and scaling to anchors.
                Its shape is :math:`(R, 4)`.
            score (array): Predicted foreground probability for anchors.
                Its shape is :math:`(R,)`.
            anchor (array): Coordinates of anchors. Its shape is
                :math:`(R, 4)`.
            img_size (tuple of ints): A tuple :obj:`height, width`,
                which contains image size after scaling.
            scale (float): The scaling factor used to scale an image after
                reading it from a file.
        loc.shape, score.shape,anchor.shape, img_size, scale = 
        ((16650, 4),
         (16650,),
         (16650, 4),
         (600, 800),
         tensor([ 1.6000], dtype=torch.float64))
         16650 = 37(hh) * 50(ww) * 9
        Returns:
            array:
            An array of coordinates of proposal boxes.
            Its shape is :math:`(S, 4)`. :math:`S` is less than
            :obj:`self.n_test_post_nms` in test time and less than
            :obj:`self.n_train_post_nms` in train time. :math:`S` depends on
            the size of the predicted bounding boxes and the number of
            bounding boxes discarded by NMS.

        """
        # NOTE: when test, remember
        # faster_rcnn.eval()
        # to set self.traing = False
        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms
            
        # Convert anchors into proposal via bbox transformations
        roi = loc2bbox(anchor, loc)
        # 这个函数输出的roi是x1y1x2y2格式
        """
        loc2bbox这个函数把之前rpn网络算出来的hh*ww*15个loc和hh*ww*15个anchor
        结合起来，套公式，算出最终预测出来的hh*ww*15个bbox，这里直接就叫roi了
        """

        # Clip predicted boxes to image.
        roi[:, 0:4:2] = np.clip(roi[:, 0:4:2], 0, img_size[1])
        # roi[:, [0,2]] 跟 roi[:, slice(0, 4, 2)] 不是一样嘛
        # 求出[y1,y2]之后用np.clip去掉bboxes伸出到图像尺寸之外的部分
        # 注意这里的img_size是原始图像经过放缩之后，输入到神经网络的size
        roi[:, 1:4:2] = np.clip(roi[:, 1:4:2], 0, img_size[0])

        # Remove predicted boxes with either height or width < threshold.

        # 这里的scale（比如说1.6），代表了原始图像经过了scale倍的放大
        # 所以原图16个像素，经过了1.6倍的放大到网络的输入，这里应该用25.6来判断是否丢弃
        min_size = self.min_size * scale
        ws = roi[:, 2] - roi[:, 0]
        hs = roi[:, 3] - roi[:, 1]
        keep = np.where((hs >= min_size) & (ws >= min_size))[0]
        # 重新计算高和宽是为了淘汰掉一批小于25.6的框
        roi = roi[keep, :]
        score = score[keep]
        # 剩下来的roi和对应的score,score是这个roi里是前景的概率

        # Sort all (proposal, score) pairs by score from highest to lowest.
        # Take top pre_nms_topN (e.g. 6000).
        order = score.ravel().argsort()[::-1]  # 把score从大到小排序，取相应的序号
        if n_pre_nms > 0:  # 无论如何n_pre_nms都是大于0的吧 ？
            order = order[:n_pre_nms]
        roi = roi[order, :]  # 取最大的n_pre_nms个roi出来
#         score = score[order]
        # Apply nms (e.g. threshold = 0.7).
        # Take after_nms_topN (e.g. 300).

        keep = nms(torch.cat((torch.tensor(roi), torch.tensor(score[order]).unsqueeze(1)), dim=1).cuda(), self.nms_thresh).tolist()
        # 调用CUPY版本的 nms
        if n_post_nms > 0:
            keep = keep[:n_post_nms]
        roi = roi[keep]
        # 最终输出n_post_nms个roi
        return roi