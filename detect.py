import pdb
import torch
from PIL import Image
from LightheadRCNN_Learner import LightHeadRCNN_Learner
from utils.dataset import coco_dataset, prepare_img
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='detect over image')
    parser.add_argument('-f','--file_path', help='image url to detect', default='data/room.jpg', type=str)
    parser.add_argument('-o','--output_path', help='detection output path', default='data/room_detected.jpg', type=str)
    parser.add_argument('-m','--model_name', help='trained model path', default='final/lighthead_rcnn_model_gpu.pth', type=str)
    args = parser.parse_args()
    learner = LightHeadRCNN_Learner(training=False)
    learner.load_state_dict(torch.load(learner.conf.work_space/args.model_name))
    img = Image.open(args.file_path)
    predicted_img = learner.predict_on_img(img, preset='detect', return_img=True, with_scores=False, original_size=True)
    predicted_img.save(args.output_path)