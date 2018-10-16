from torch.nn import Conv2d, Linear
from datetime import datetime

try:
    import pycocotools.coco
    import pycocotools.cocoeval
    _available = True
except ImportError:
    _available = False

import chainer
from chainer import iterators
from chainercv.datasets import coco_bbox_label_names
from chainercv.datasets import COCOBboxDataset
from chainercv.evaluations import eval_detection_coco
from chainercv.utils import apply_to_iterator
from chainercv.utils import ProgressHook

def eva_coco(dataset, func, limit = 1000, preset = 'evaluate'):
    total = limit if limit else len(dataset)
    orig_ids = dataset.ids.copy()
    dataset.ids = dataset.ids[:total]
    iterator = iterators.SerialIterator(dataset, 1, repeat=False, shuffle=False)
    in_values, out_values, rest_values = apply_to_iterator(func, 
                                                           iterator, 
                                                           hook=ProgressHook(len(dataset)))
    pred_bboxes, pred_labels, pred_scores = out_values
    gt_bboxes, gt_labels, gt_areas, gt_crowdeds = rest_values
    result = eval_detection_coco(pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, gt_areas, gt_crowdeds)
    keys = [
        'map/iou=0.50:0.95/area=all/max_dets=100',
        'map/iou=0.50/area=all/max_dets=100',
        'map/iou=0.75/area=all/max_dets=100',
        'map/iou=0.50:0.95/area=small/max_dets=100',
        'map/iou=0.50:0.95/area=medium/max_dets=100',
        'map/iou=0.50:0.95/area=large/max_dets=100',
        'mar/iou=0.50:0.95/area=all/max_dets=1',
        'mar/iou=0.50:0.95/area=all/max_dets=10',
        'mar/iou=0.50:0.95/area=all/max_dets=100',
        'mar/iou=0.50:0.95/area=small/max_dets=100',
        'mar/iou=0.50:0.95/area=medium/max_dets=100',
        'mar/iou=0.50:0.95/area=large/max_dets=100',
    ]
    print('')
    results = []
    for key in keys:
        print('{:s}: {:f}'.format(key, result[key]))
        results.append(result[key])
    dataset.ids = orig_ids
    return results

def normal_init(m, mean, stddev):
    if type(m) == Linear or type(m) == Conv2d:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
        
def get_time():
    return (str(datetime.now())[:-10]).replace(' ','-').replace(':','-')
