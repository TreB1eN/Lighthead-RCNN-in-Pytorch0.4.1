import argparse
import torch
from LightheadRCNN_Learner import LightHeadRCNN_Learner
from utils.dataset import coco_dataset, prepare_img

def parse_args():
    parser = argparse.ArgumentParser(description="eval Lighthead-RCNN")
    parser.add_argument("-model", "--file", help="model file name, placed under final folder",default='lighthead_rcnn_model_gpu.pth', type=str)
    parser.add_argument("-n", "--limit", help="eval examples number", default=3000, type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    learner = LightHeadRCNN_Learner(training = False)
    learner.load_state_dict(torch.load(learner.conf.work_space/'final'/args.file))
    learner.val_dataset =  coco_dataset(learner.conf, mode = 'val')
    learner.val_length = len(learner.val_dataset)
    results = learner.eva_on_coco(limit = args.limit)