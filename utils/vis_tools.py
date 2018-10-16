import colorsys
import random
import numpy as np
from PIL import Image,ImageDraw,ImageFont
# from utils.box_utils import *
from torchvision import transforms as trans
import torch
from utils.bbox_tools import xywh_2_x1y1x2y2

def get_class_colors(conf):
    colors = random_colors(conf.class_num)
    class_2_color = {}
    for i,c in enumerate([*conf.correct_id_2_class.keys()]):
        class_2_color[c] = (colors[i][0], colors[i][1], colors[i][2])
    return class_2_color

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    colors = (np.array(colors) * 255).astype(int)
    return colors

def draw_bbox_class(conf, img, labels, bboxes, id_2_class, class_2_color, scores = []):
    """
    img : PIL Image
    labels : torch.tensor or Np.array [N,] 
    bboxes : torch.tensor or Np.array [N, 4] x1y1x2y2 format
    id_2_class : Dictionary
    class_2_color : Dictionary
    """

    if len(bboxes) == 0:
        return img
    truefont = ImageFont.truetype('data/fonts/arial.ttf',size=conf.font_size)
    x_max, y_max = img.size
    draw = ImageDraw.Draw(img)
    for i in range(bboxes.shape[0]):
        x1y1x2y2 = bboxes[i]
        x_corner,y1 = x1y1x2y2[0].item(),x1y1x2y2[1].item()
        
        if len(scores) == 0:
            text = id_2_class[str(labels[i].item())]
            
        else: 
            text = '{}_{:.2%}'.format(id_2_class[str(labels[i].item())], scores[i])
        
        text_w, text_h = draw.textsize(text,font=truefont)
        
        if y1 < text_h:
            y_corner = 0
        else:
            y_corner = y1 - text_h
            
        draw.rectangle([(x_corner,y_corner),(x_corner + text_w, y_corner + text_h)],fill = class_2_color[str(labels[i].item())])
        draw.rectangle(x1y1x2y2,outline=class_2_color[str(labels[i].item())])
        draw.text((x_corner,y_corner),text=text, fill='black', font=truefont)
    return img


def show_util_with_conf(conf,idx,imgs, labels_group, bboxes_group, confidences_group, cls_conf_group, correct_id_2_class, class_2_color):
    if torch.sum(torch.abs(bboxes_group[idx])).item() == 0:
        return trans.ToPILImage()(de_preprocess(conf, imgs[idx].cpu()))
    return draw_bbox_class_with_conf(conf, trans.ToPILImage()(de_preprocess(conf, imgs[idx].cpu())),\
                                     labels_group[idx].cpu(), bboxes_group[idx].cpu(),\
                                     correct_id_2_class, class_2_color, \
                                     confidences_group[idx].cpu(), cls_conf_group[idx].cpu())

def show_util(conf,idx,imgs, labels_group, bboxes_group, correct_id_2_class, class_2_color):
    if torch.sum(torch.abs(bboxes_group[idx])).item() == 0:
        return trans.ToPILImage()(de_preprocess(conf, imgs[idx].cpu()))
    return draw_bbox_class(conf,\
        trans.ToPILImage()(de_preprocess(conf, imgs[idx].cpu())),\
        labels_group[idx].cpu(), bboxes_group[idx].cpu(), correct_id_2_class, class_2_color)

def de_preprocess(conf, tensor, cuda=False):
    if cuda:
        return (tensor + conf.mean_tensor.cuda())/255
    else:
        return (tensor + conf.mean_tensor)/255

def to_img(conf, tensor):
    return trans.ToPILImage()(de_preprocess(conf, tensor))