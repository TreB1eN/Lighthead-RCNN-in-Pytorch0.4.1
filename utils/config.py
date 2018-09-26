from pathlib import Path
import torch
from torchvision import transforms as trans
import json

class Config(object):
    data_path = Path('data')
    coco_path = data_path/'coco2017'
    anno_path = coco_path/'annotations'
    train_path = coco_path/'train2017'
    val_path = coco_path/'val2017'
    test_path = coco_path/'test2017'
    train_anno_path = anno_path/'instances_train2017.json'
    val_anno_path = anno_path/'instances_val2017.json'
    pretrained_model_path = 'models/lighthead-rcnn-extractor-pretrained.pth'
    work_space = Path('work_space')
    log_path = work_space/'log'
    min_sizes = [600, 700, 800, 900, 1000]
#     min_sizes = [1000] # delete this when finish debug
    max_size = 1400
    class_num = 80
    roi_size = 7
    font_size = 10
    spatial_scale = 1/16.
    with open(data_path/'coco_maps.json', 'r') as f:
        maps = json.load(f)
    correct_id_2_class = maps[2]
    
    use_roi_align = True
    
#     eva_num_during_training = 2000
#     coco_eva_num_during_training = 2000
    # test only
    board_loss_interval = 100
    eval_interval = 10
    eval_coco_interval = 4
    board_pred_image_interval = 5
    save_interval = 12
    
    eva_num_during_training = 1000
    coco_eva_num_during_training = 2500
    
    mean = [0.48145765, 0.45469216, 0.40384355]
    std = [1., 1., 1.]
    transform = trans.Compose([
        trans.ToTensor(),
        trans.Normalize(mean, std),
        lambda img : img * 255.
    ])
    
    std_tensor = torch.Tensor(std).view(3,1,1)
    mean_tensor = torch.Tensor(mean).view(3,1,1)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 1
    lr = 0.0005 * 1.25
    momentum = 0.9
    weight_decay = 1e-4
    
    rpn_sigma = 3.
    roi_sigma = 1.
    n_ohem_sample = 256
    
    loc_normalize_mean = (0., 0., 0., 0.),
    loc_normalize_std = (0.1, 0.1, 0.2, 0.2)
    
    loc_std_tensor = torch.tensor(loc_normalize_std, dtype=torch.float).to(device)
    loc_mean_tensor = torch.tensor(loc_normalize_mean, dtype=torch.float).to(device)
