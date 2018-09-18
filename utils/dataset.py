from pycocotools.coco import COCO
from torchvision import datasets
from collections import namedtuple
import torch
import numpy as np
from torchvision.transforms.functional import hflip
import random
from PIL import Image
from utils.bbox_tools import adjust_bbox, horizontal_flip_boxes, xywh_2_x1y1x2y2
from utils.vis_tools import get_class_colors, draw_bbox_class, de_preprocess
from torch.utils.data import Dataset
import pdb

class coco_dataset(Dataset):
    def __init__(self, conf, path, anno_path):
        self.orig_dataset = datasets.CocoDetection(path, anno_path)
        self.pair = namedtuple('pair', ['img', 'bboxes', 'labels', 'scale'])
        self.maps = conf.maps
        self.transform = conf.transform
        self.min_size = conf.min_size
        self.max_size = conf.max_size
#         with open(conf.data_path/'valid_index.pkl', 'rb') as f:
#             self.valid_ids = pickle.load(f)
    
    def __len__(self):
        return len(self.orig_dataset)
    
    def __getitem__(self, index):
        img, targets = self.orig_dataset[index]
        img, scale = self.prepare_img(img)
        if targets == []:
            img = (self.transform(img) * 255).unsqueeze(0) 
            return self.pair(img, [], [], scale)
        labels,bboxes = synthesize_bbox_id(targets, self.maps[0])
        bboxes = adjust_bbox(scale, bboxes)
        if random.random() > 0.5:
            img = hflip(img)
            bboxes = horizontal_flip_boxes(bboxes, img.size[0])

        img = self.transform(img).unsqueeze(0) 
        bboxes = xywh_2_x1y1x2y2(bboxes.numpy())
        img_size = [img.shape[2], img.shape[3]]
        bboxes[:, slice(0, 4, 2)] = np.clip(bboxes[:, slice(0, 4, 2)], 0, img_size[1])
        # roi[:, [0,2]] 跟 roi[:, slice(0, 4, 2)] 不是一样嘛
        # 求出[y1,y2]之后用np.clip去掉bboxes伸出到图像尺寸之外的部分
        # 注意这里的img_size是原始图像经过放缩之后，输入到神经网络的size
        bboxes[:, slice(1, 4, 2)] = np.clip(bboxes[:, slice(1, 4, 2)], 0, img_size[0])
        return self.pair(img, bboxes, labels.numpy(), scale)  
    
    def prepare_img(self, img):
        """Preprocess an image for feature extraction.

        The length of the shorter edge is scaled to :obj:`conf.min_size`.
        After the scaling, if the length of the longer edge is longer than
        :obj:`conf.max_size`, the image is scaled to fit the longer edge
        to :obj:`conf.max_size`.

        Args:
            img (~PIL.Image): An PIL image. 

        Returns:
            A preprocessed image.
            resize scale
        """
        W, H = img.size
        scale = 1.
        scale = self.min_size / min(H, W)
        if scale * max(H, W) > self.max_size:
            scale = self.max_size / max(H, W)
        img = img.resize((int(W * scale), int(H * scale)), Image.BICUBIC)
        return img, scale

def prepare_img(conf, img):
    """Preprocess an image for feature extraction.

    The length of the shorter edge is scaled to :obj:`conf.min_size`.
    After the scaling, if the length of the longer edge is longer than
    :obj:`conf.max_size`, the image is scaled to fit the longer edge
    to :obj:`conf.max_size`.

    Args:
        img (~PIL.Image): An PIL image. 

    Returns:
        A preprocessed image.
        resize scale
    """
    H, W = img.size
    scale = 1.
    scale = conf.min_size / min(H, W)
    if scale * max(H, W) > conf.max_size:
        scale = conf.max_size / max(H, W)
    img = img.resize((int(H * scale), int(W * scale)), Image.BICUBIC)
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