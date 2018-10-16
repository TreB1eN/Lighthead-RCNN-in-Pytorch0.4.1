import torch
import numpy as np
import six
import pdb 

def adjust_bbox(scale, bboxes, detect=False):
    if not detect:
        bboxes = bboxes * scale
    else:
        bboxes = bboxes / scale
    return bboxes

def horizontal_flip_boxes(bboxes, size):
    temp = (size - bboxes[:,0]).copy()
    bboxes[:,0] = size - bboxes[:,2]
    bboxes[:,2] = temp
    return bboxes

def y1x1y2x2_2_x1y1x2y2(bbox):
    if type(bbox) == np.ndarray:
        new_bbox = bbox.copy()
    if type(bbox) == torch.Tensor: 
        new_bbox = bbox.clone()
    new_bbox[:,0::2] = bbox[:,1::2]
    new_bbox[:,1::2] = bbox[:,0::2]
    return new_bbox

def x1y1x2y2_2_xcycwh(bbox):
    if type(bbox) == np.ndarray:
        new_bbox = bbox.copy()
    if type(bbox) == torch.Tensor: 
        new_bbox = bbox.clone()
    new_bbox[:,:2] = 0.5 * (bbox[:,:2] + bbox[:,2:])
    new_bbox[:,2:] = bbox[:,2:] - bbox[:,:2]
    return new_bbox    

def xywh_2_x1y1x2y2(bbox):
    '''
    accept both numpy.array and torch.tensor
    transform the bbox from xywh format to x1y1x2y2 format
    '''
    if type(bbox) == np.ndarray:
        if bbox.ndim == 1:
            return torch.tensor([bbox[0],bbox[1],bbox[0] + bbox[2],bbox[1] + bbox[3]])
        else:
            new_bbox = bbox.copy()
            new_bbox[:,2:] += new_bbox[:,:2]
            return new_bbox
    if type(bbox) == torch.Tensor:    
        if bbox.ndimension() == 1:
            return np.array([bbox[0],bbox[1],bbox[0] + bbox[2],bbox[1] + bbox[3]])
        else:
            new_bbox = bbox.clone()
            new_bbox[:,2:] += new_bbox[:,:2]
            return new_bbox

def x1y1x2y2_2_xywh(bbox):
    if type(bbox) == np.ndarray:
        new_bbox = bbox.copy()
    if type(bbox) == torch.Tensor: 
        new_bbox = bbox.clone()
    new_bbox[:,2:] = bbox[:,2:] - bbox[:,:2]
    return new_bbox 
        
def trim_pred_bboxes(bboxes, size):
    """
    bboxes : torch.tensor, shape = [n,4], format = xcycwh
    """
    return x1y1x2y2_2_xcycwh(torch.clamp(xcycwh_2_x1y1x2y2(bboxes), 0, size))

def loc2bbox(src_bbox, loc):
    """Decode bounding boxes from bounding box offsets and scales.

    Given bounding box offsets and scales computed by
    :meth:`bbox2loc`, this function decodes the representation to
    coordinates in 2D image coordinates.

    Given scales and offsets :math:`t_y, t_x, t_h, t_w` and a bounding
    box whose center is :math:`(y, x) = p_y, p_x` and size :math:`p_h, p_w`,
    the decoded bounding box's center :math:`\\hat{g}_y`, :math:`\\hat{g}_x`
    and size :math:`\\hat{g}_h`, :math:`\\hat{g}_w` are calculated
    by the following formulas.

    * :math:`\\hat{g}_y = p_h t_y + p_y`
    * :math:`\\hat{g}_x = p_w t_x + p_x`
    * :math:`\\hat{g}_h = p_h \\exp(t_h)`
    * :math:`\\hat{g}_w = p_w \\exp(t_w)`

    The decoding formulas are used in works such as R-CNN [#]_.

    The output is same type as the type of the inputs.

    .. [#] Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik. \
    Rich feature hierarchies for accurate object detection and semantic \
    segmentation. CVPR 2014.

    Args:
        src_bbox (array): A coordinates of bounding boxes.
            Its shape is :math:`(R, 4)`. These coordinates are
            :math:`p_{xmin}, p_{ymin}, p_{xmax}, p_{ymax}`.
            src_bbox格式是x1y1x2y2 
            add support for torch.tensor
        loc (array): An array with offsets and scales.
            The shapes of :obj:`src_bbox` and :obj:`loc` should be same.
            This contains values :math:`t_x, t_y, t_w, t_h`.
            add support for torch.tensor

    Returns:
        array:
        Decoded bounding box coordinates. Its shape is :math:`(R, 4)`. \
        The second axis contains four values \
        :math:`\\hat{g}_{xmin}, \\hat{g}_{ymin}, \\hat{g}_{xmax}`, \\hat{g}_{ymax}, .

    """
    """
    因为经过RPN网络计算出来的是loc则是由4个位置参数(tx,ty,tw,th)组成，
    这样比直接回归座标更好，tx,ty,tw,th都是根据预设的anchor算出来的偏移量
    直接去看公式吧
    这里src_bbox就是anchors,
    loc 就是经过RPN算出来的偏移量，
    他们size都是[hh*ww*9,4]
    因为每个点上都有9个预设的anchor,
    rpn算出来的结果也包括对这9个点的预测
    """
#     pdb.set_trace()
    assert type(src_bbox) == type(loc), 'src_bbox and loc are not the same class'
    assert type(src_bbox) == torch.Tensor or type(src_bbox) == np.ndarray, 'should be tensor or array'
    tensor_flag = isinstance(src_bbox, torch.Tensor)
    if src_bbox.shape[0] == 0:
        if tensor_flag:
            return torch.zeros((0, 4), dtype=loc.dtype).to(loc.device)
        else:
            return np.zeros((0, 4), dtype=loc.dtype)
    if not tensor_flag:
        src_bbox = src_bbox.astype(src_bbox.dtype, copy=False)

    src_width = src_bbox[:, 2] - src_bbox[:, 0]
    src_height = src_bbox[:, 3] - src_bbox[:, 1]
    src_ctr_x = src_bbox[:, 0] + 0.5 * src_width
    src_ctr_y = src_bbox[:, 1] + 0.5 * src_height
    
    #上面4步算出anchors的中心和宽高，便于下一步计算

    dx = loc[:, 0::4] #等价于 loc[:,0].reshape(-1,1)
    dy = loc[:, 1::4] #等价于 loc[:,1].reshape(-1,1)
    dw = loc[:, 2::4] #等价于 loc[:,2].reshape(-1,1)
    dh = loc[:, 3::4] #等价于 loc[:,3].reshape(-1,1)
    # 其实就是把loc里四个值一维取出
    
    if tensor_flag:
        ctr_x = dx * src_width.unsqueeze(-1) + src_ctr_x.unsqueeze(-1)
        # 注意公式，dy还要乘anchor的h才是最终偏移量
        """
        (Pdb) src_height.shape
        (16650,)
        (Pdb) src_height[:, np.newaxis].shape
        (16650, 1)
        """
        # 公式指的是从anchors + 预测的偏移 得出最终预测的dst_bbox
        # 这里开始套公式求得新的中心点
        ctr_y = dy * src_height.unsqueeze(-1) + src_ctr_y[:, np.newaxis]    

        h = torch.exp(dh) * src_height.unsqueeze(-1)
        # 新的高和宽
        w = torch.exp(dw) * src_width.unsqueeze(-1)

        dst_bbox = torch.zeros(loc.shape, dtype=loc.dtype).to(loc.device)
    else:
        ctr_x = dx * src_width[:, np.newaxis] + src_ctr_x[:, np.newaxis]
        # 注意公式，dy还要乘anchor的h才是最终偏移量
        """
        (Pdb) src_height.shape
        (16650,)
        (Pdb) src_height[:, np.newaxis].shape
        (16650, 1)
        """
        # 公式指的是从anchors + 预测的偏移 得出最终预测的dst_bbox
        # 这里开始套公式求得新的中心点
        ctr_y = dy * src_height[:, np.newaxis] + src_ctr_y[:, np.newaxis]    

        h = np.exp(dh) * src_height[:, np.newaxis]
        # 新的高和宽
        w = np.exp(dw) * src_width[:, np.newaxis]

        dst_bbox = np.zeros(loc.shape, dtype=loc.dtype)
    # 接下来再把中心点+宽高 转换成 [x1,y1,x2,y2]的形式
    dst_bbox[:, 0::4] = ctr_x - 0.5 * w
    dst_bbox[:, 1::4] = ctr_y - 0.5 * h    
    dst_bbox[:, 2::4] = ctr_x + 0.5 * w
    dst_bbox[:, 3::4] = ctr_y + 0.5 * h   
    
    """
    他这里的对x,y坐标的计算是先算出来变换h和w之前的中心点，
    这里算中心点用的anchor的h和w，也就是没校准的h和w
    再加上x和y的偏移量，得到新的中心点
    最后再减去校准之后的h和w,把中心点换算回，y1x1y2x2
    这样计算和直接加减x,y的偏移量，直接计算h和w的方法数值上是有区别的。
    这样算是正统的anchor box的计算方法 ？
    """
    #返回的是经过RPN输出的偏移量校准过的最终预测的hh*ww*9个bboxes
    return dst_bbox

def bbox2loc(src_bbox, dst_bbox):
    """Encodes the source and the destination bounding boxes to "loc".

    Given bounding boxes, this function computes offsets and scales
    to match the source bounding boxes to the target bounding boxes.
    Mathematcially, given a bounding box whose center is
    :math:`(y, x) = p_y, p_x` and
    size :math:`p_h, p_w` and the target bounding box whose center is
    :math:`g_y, g_x` and size :math:`g_h, g_w`, the offsets and scales
    :math:`t_x, t_y, t_w, t_h` can be computed by the following formulas.

    * :math:`t_y = \\frac{(g_y - p_y)} {p_h}`
    * :math:`t_x = \\frac{(g_x - p_x)} {p_w}`
    * :math:`t_h = \\log(\\frac{g_h} {p_h})`
    * :math:`t_w = \\log(\\frac{g_w} {p_w})`

    The output is same type as the type of the inputs.
    The encoding formulas are used in works such as R-CNN [#]_.

    .. [#] Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik. \
    Rich feature hierarchies for accurate object detection and semantic \
    segmentation. CVPR 2014.

    Args:
        src_bbox (array): An image coordinate array whose shape is
            :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
            These coordinates are
            :math:`p_{ymin}, p_{xmin}, p_{ymax}, p_{xmax}`.
            add support for torch.tensor
        dst_bbox (array): An image coordinate array whose shape is
            :math:`(R, 4)`.
            These coordinates are
            :math:`p_{xmin}, p_{ymin}, p_{xmax}, p_{ymax}`.
            add support for torch.tensor

    Returns:
        array:
        Bounding box offsets and scales from :obj:`src_bbox` \
        to :obj:`dst_bbox`. \
        This has shape :math:`(R, 4)`.
        The second axis contains four values :math:`t_x, t_y, t_w, t_h`.

    """
    """
    根据预测的roi和gt_bbox去计算正确的偏移量，作为loss的输入
    """
    width = src_bbox[:, 2] - src_bbox[:, 0]
    height = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_x = src_bbox[:, 0] + 0.5 * width
    ctr_y = src_bbox[:, 1] + 0.5 * height    

    base_width = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_height = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_x = dst_bbox[:, 0] + 0.5 * base_width
    base_ctr_y = dst_bbox[:, 1] + 0.5 * base_height

#     eps = np.finfo(height.dtype).eps #求一个机器极小值
    eps = np.finfo(np.float32).eps
    height = np.maximum(height, eps)
    width = np.maximum(width, eps)

    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = np.log(base_height / height)
    dw = np.log(base_width / width)
    """
    这里也是先计算两边的中心点，以中心点的距离来算偏移
    跟loc2bbox是对称的
    """
    loc = np.vstack((dx, dy, dw, dh)).transpose()
    return loc


def bbox_iou(bbox_a, bbox_b):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.

    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    This function accepts both :obj:`numpy.ndarray` and :obj:`cupy.ndarray` as
    inputs. Please note that both :obj:`bbox_a` and :obj:`bbox_b` need to be
    same type.
    The output is same type as the type of the inputs.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
            
        both inputs use x1y1x2y2 format

    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    """
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError

    # top left
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # bottom right
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)

def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2], anchor_scales=[2, 4, 8, 16, 32]):
    """Generate anchor base windows by enumerating aspect ratio and scales.

    Generate anchors that are scaled and modified to the given aspect ratios.
    Area of a scaled anchor is preserved when modifying to the given aspect
    ratio.

    :obj:`R = len(ratios) * len(anchor_scales)` anchors are generated by this
    function.
    The :obj:`i * len(anchor_scales) + j` th anchor corresponds to an anchor
    generated by :obj:`ratios[i]` and :obj:`anchor_scales[j]`.

    For example, if the scale is :math:`8` and the ratio is :math:`0.25`,
    the width and the height of the base window will be stretched by :math:`8`.
    For modifying the anchor to the given aspect ratio,
    the height is halved and the width is doubled.

    Args:
        base_size (number): The width and the height of the reference window.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.

    Returns:
        ~numpy.ndarray:
        An array of shape :math:`(R, 4)`.
        Each element is a set of coordinates of a bounding box.
        The second axis corresponds to
        :math:`(y_{min}, x_{min}, y_{max}, x_{max})` of a bounding box.

    """
    py = base_size / 2.
    px = base_size / 2.
    """
    base_size = 16,根据取feature这一层相对原始尺寸的大小，
    就是缩小了16倍，换句话说，feature这一层一个stride，
    相当于在原始图像上移动16个像素。
     """

    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4),
                           dtype=np.float32)
    # anchor_base.shape = [3*3,4] 先预生成9个anchor_boxes
    # 这里还细心地把中心点考虑进去了
    for i in six.moves.range(len(ratios)):
        for j in six.moves.range(len(anchor_scales)):
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])            

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = px - w / 2.
            anchor_base[index, 1] = py - h / 2.
            anchor_base[index, 2] = px + w / 2.
            anchor_base[index, 3] = py + h / 2.
    return anchor_base # anchor_base的格式 [-w/2, -h/2, w/2, h/2]

def enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    # return (K*A, 4)
    # !TODO: add support for torch.CudaTensor
    # xp = cuda.get_array_module(anchor_base)
    # it seems that it can't be boosed using GP
    # feat_stride = 16
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_y = np.arange(0, height * feat_stride, feat_stride)    
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    # shift_x.shape,shift_y.shape 都等于 [height,width]
    # 相当于构建了一个 [height,width] 的网格，每一个网格的大小是16 * 16
    shift = np.stack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel()), axis=1)
    # shift.shape = (height*width, 4)
    # shift 相当于这些网格左上角和（0,0）点的偏移量，所以叫shift

    A = anchor_base.shape[0]  # 9,anchor数量
    K = shift.shape[0]  # height*width,所有网格数
    anchor = anchor_base.reshape((1, A, 4)) + shift.reshape((1, K, 4)).transpose((1, 0, 2))
    # [1,15,4] + [K,1,4] = [K,15,4]
    # K个网格，每个网格匹配上9种anchor_size
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    # [K*9,4], K*9 个predefined anchors
    return anchor
    # anchor的格式是x1y1x2y2
