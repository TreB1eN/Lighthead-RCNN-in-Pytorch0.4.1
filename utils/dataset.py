from pycocotools.coco import COCO
from torch.utils.data import Dataset
from collections import namedtuple
import torch
import numpy as np
import random
from PIL import Image
from utils.bbox_tools import adjust_bbox, horizontal_flip_boxes, xywh_2_x1y1x2y2, y1x1y2x2_2_x1y1x2y2
from chainercv.datasets import COCOBboxDataset
from chainercv.chainer_experimental.datasets.sliceable import ConcatenatedDataset
from chainercv.transforms.image.resize import resize
import pdb

class coco_dataset(Dataset):
    def __init__(self, conf, mode = 'train'):
        assert mode == 'train' or mode == 'val', 'mode shoulde be train or val'    
        if mode == 'train':
            self.training = True
            self.orig_dataset = ConcatenatedDataset(COCOBboxDataset(data_dir=conf.coco_path, split='train'),
                                                    COCOBboxDataset(data_dir=conf.coco_path, split='valminusminival'))
            print('train dataset imported')
        else:
            self.training = False
            self.orig_dataset = COCOBboxDataset(data_dir=conf.coco_path, 
                                                split='minival', 
                                                use_crowded=True, 
                                                return_crowded=True, 
                                                return_area=True)
            print('minival dataset imported')
        self.pair = namedtuple('pair', ['img', 'bboxes', 'labels', 'scale'])
        self.maps = conf.maps
        self.min_sizes = conf.min_sizes
        self.max_size = conf.max_size
        self.mean = conf.mean
#         self.std = conf.std
    
    def __len__(self):
        return len(self.orig_dataset)
    
    def __getitem__(self, index):
        if self.training:
            img, bboxes, labels = self.orig_dataset[index]
        else:
            img, bboxes, labels, _, _ = self.orig_dataset[index]
        img, scale = self.prepare_img(img, not self.training)
        if len(bboxes) == 0:
#             print('index {} dosent have objects'.format(index))
            return self.pair(img, [], [], scale)
        bboxes = y1x1y2x2_2_x1y1x2y2(bboxes)
        bboxes = adjust_bbox(scale, bboxes)
        if self.training:
            if random.random() > 0.5:
                img[:] = img[:, :, ::-1]
                bboxes = horizontal_flip_boxes(bboxes, img.shape[-1])
#         img = torch.tensor(img).unsqueeze(0)
        img_size = [img.shape[1], img.shape[2]] # H,W
        bboxes[:, slice(0, 4, 2)] = np.clip(bboxes[:, slice(0, 4, 2)], 0, img_size[1])
        # roi[:, [0,2]] 跟 roi[:, slice(0, 4, 2)] 不是一样嘛
        # 求出[y1,y2]之后用np.clip去掉bboxes伸出到图像尺寸之外的部分
        # 注意这里的img_size是原始图像经过放缩之后，输入到神经网络的size
        bboxes[:, slice(1, 4, 2)] = np.clip(bboxes[:, slice(1, 4, 2)], 0, img_size[0])
        return self.pair(img, bboxes, labels, scale)
    
    def prepare_img(self, img, infer = False):
        """Preprocess an image for feature extraction.

        The length of the shorter edge is scaled to :obj:`conf.min_size`.
        After the scaling, if the length of the longer edge is longer than
        :obj:`conf.max_size`, the image is scaled to fit the longer edge
        to :obj:`conf.max_size`.

        Args:
            img (np.array): RGB img [3,H,W] 

        Returns:
            A preprocessed image.
            resize scale
        """
        W, H = img.shape[2], img.shape[1]
        if infer:
            min_size = self.min_sizes[-1]
        else:
            min_size = random.choice(self.min_sizes)
        scale = min_size / min(H, W)
        if scale * max(H, W) > self.max_size:
            scale = self.max_size / max(H, W)
        img = resize(img, (int(H * scale), int(W * scale)))
        img = (img - self.mean).astype(np.float32, copy=False)
        return img, scale

def prepare_img(conf, img, resolution = -1):
    """Preprocess an image for feature extraction.

    The length of the shorter edge is scaled to :obj:`conf.min_size`.
    After the scaling, if the length of the longer edge is longer than
    :obj:`conf.max_size`, the image is scaled to fit the longer edge
    to :obj:`conf.max_size`.

    Args:
        img (np.array): RGB img [3,H,W] 

    Returns:
        A preprocessed image.
        resize scale
    """
    W, H = img.shape[2], img.shape[1]
    min_size = conf.min_sizes[resolution]
    scale = min_size / min(H, W)
    if scale * max(H, W) > conf.max_size:
        scale = conf.max_size / max(H, W)
    img = resize(img, (int(H * scale), int(W * scale)))
    img = (img - conf.mean).astype(np.float32, copy=False)
    return img, scale    

def rcnn_collate_fn(batch):
    imgs_group = []
    bboxes_group = []
    labels_group = []
    scales_group = []
    for item in batch:
        imgs_group.append(item.img.unsqueeze(0))            
        bboxes_group.append(item.bboxes)
        labels_group.append(item.labels)
        scales_group.append(item.scale)
    return torch.cat(imgs_group), bboxes_group, labels_group, scales_group    

def synthesize_bbox_id(targets, id_2_correct_id):
    labels = []
    bboxes = []
    for target in targets:
        label = id_2_correct_id[str(target['category_id'])]
        bbox = torch.Tensor(target['bbox']).view(1,4)
        # get rid of targets that are too small
        # try number 17975 data, there is even a wrong data with h = 0,took me so long to debug
        if bbox[:,2:].min().item() < 2.:
            continue
        labels.append(label)
        bboxes.append(bbox)
    return torch.tensor(labels, dtype=torch.long), torch.cat(bboxes)

def get_coco_class_name_map(anno):
    coco=COCO(anno)
    cats = coco.loadCats(coco.getCatIds())
    class_2_id = {}
    id_2_class = {}
    for pair in cats:
        id_2_class[pair['id']] = pair['name']
        class_2_id[pair['name']] = pair['id']
    return class_2_id,id_2_class

def get_id_maps(conf):
    coco_class_2_id, coco_id_2_class = get_coco_class_name_map(conf.val_anno_path)

    id_2_correct_id = {}
    correct_id_2_id = {}
    id_2_correct_id = dict(zip(coco_id_2_class.keys(), range(80)))
    correct_id_2_id = dict(zip(range(80), coco_id_2_class.keys()))

    correct_id_2_class = {}
    class_2_correct_id = {}
    for k, v in coco_id_2_class.items():
        correct_id_2_class[id_2_correct_id[k]] = v
        class_2_correct_id[v] = id_2_correct_id[k]

    id_2_correct_id = {}
    correct_id_2_id = {}
    id_2_correct_id = dict(zip(coco_id_2_class.keys(), range(80)))
    correct_id_2_id = dict(zip(range(80), coco_id_2_class.keys()))

    correct_id_2_class = {}
    class_2_correct_id = {}
    for k, v in coco_id_2_class.items():
        correct_id_2_class[id_2_correct_id[k]] = v
        class_2_correct_id[v] = id_2_correct_id[k]

    maps = [
        id_2_correct_id, correct_id_2_id, correct_id_2_class, class_2_correct_id
    ]
    return maps