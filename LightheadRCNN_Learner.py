import pdb
from tqdm import tqdm
import os
import torch
import numpy as np
import torch
from torch.nn import Module
from models.region_proposal_network import RegionProposalNetwork
from models.backbone import ResNet101Extractor
from models.proposals import AnchorTargetCreator, ProposalTargetCreator
from models.head import LightHeadRCNNResNet101_Head
from models.losses import OHEM_loss, fast_rcnn_loc_loss
from tensorboardX import SummaryWriter
from utils.config import Config
from torchvision import transforms as trans
from utils.dataset import coco_dataset, prepare_img
from utils.vis_tools import draw_bbox_class, get_class_colors, to_img, de_preprocess
from utils.bbox_tools import loc2bbox, x1y1x2y2_2_xywh, xywh_2_x1y1x2y2, adjust_bbox, y1x1y2x2_2_x1y1x2y2
from utils.utils import get_time, eva_coco
from functions.nms.nms_wrapper import nms, soft_nms
from torch.optim import SGD
from models.model_utils import get_trainables
from torch.nn import functional as F
from collections import namedtuple
import json
import pycocotools.coco
import pycocotools.cocoeval
from PIL import Image

class LightHeadRCNN_Learner(Module):
    def __init__(self, training=True):
        super(LightHeadRCNN_Learner, self).__init__()
        self.conf = Config()
        self.class_2_color = get_class_colors(self.conf)   

        self.extractor = ResNet101Extractor(self.conf.pretrained_model_path).to(self.conf.device)
        self.rpn = RegionProposalNetwork().to(self.conf.device)
#         self.head = LightHeadRCNNResNet101_Head(self.conf.class_num + 1, self.conf.roi_size).to(self.conf.device)
        self.loc_normalize_mean=(0., 0., 0., 0.),
        self.loc_normalize_std=(0.1, 0.1, 0.2, 0.2)
        self.head = LightHeadRCNNResNet101_Head(self.conf.class_num + 1, 
                                                self.conf.roi_size, 
                                                roi_align = self.conf.use_roi_align).to(self.conf.device)
        self.class_2_color = get_class_colors(self.conf)
        self.detections = namedtuple('detections', ['roi_cls_locs', 'roi_scores', 'rois'])
             
        if training:
            self.train_dataset = coco_dataset(self.conf, mode = 'train')
            self.train_length = len(self.train_dataset)
            self.val_dataset =  coco_dataset(self.conf, mode = 'val')
            self.val_length = len(self.val_dataset)
            self.anchor_target_creator = AnchorTargetCreator()
            self.proposal_target_creator = ProposalTargetCreator(loc_normalize_mean = self.loc_normalize_mean, 
                                                                 loc_normalize_std = self.loc_normalize_std)
            self.step = 0
            self.optimizer = SGD([
                {'params' : get_trainables(self.extractor.parameters())},
                {'params' : self.rpn.parameters()},
                {'params' : [*self.head.parameters()][:8], 'lr' : self.conf.lr*3},
                {'params' : [*self.head.parameters()][8:]},
            ], lr = self.conf.lr, momentum=self.conf.momentum, weight_decay=self.conf.weight_decay)
            self.base_lrs = [params['lr'] for params in self.optimizer.param_groups]
            self.warm_up_duration = 5000
            self.warm_up_rate = 1 / 5
            self.train_outputs = namedtuple('train_outputs',
                                            ['loss_total', 
                                             'rpn_loc_loss', 
                                             'rpn_cls_loss', 
                                             'ohem_roi_loc_loss', 
                                             'ohem_roi_cls_loss',
                                             'total_roi_loc_loss',
                                             'total_roi_cls_loss'])                                      
            self.writer = SummaryWriter(self.conf.log_path)
            self.board_loss_every = self.train_length // self.conf.board_loss_interval
            self.evaluate_every = self.train_length // self.conf.eval_interval
            self.eva_on_coco_every = self.train_length // self.conf.eval_coco_interval
            self.board_pred_image_every = self.train_length // self.conf.board_pred_image_interval
            self.save_every = self.train_length // self.conf.save_interval
            # only for debugging
#             self.board_loss_every = 5
#             self.evaluate_every = 6
#             self.eva_on_coco_every = 7
#             self.board_pred_image_every = 8
#             self.save_every = 10
        
    def set_training(self):
        self.train()
        self.extractor.set_bn_eval()
        
    def lr_warmup(self):
        assert self.step <= self.warm_up_duration, 'stop warm up after {} steps'.format(self.warm_up_duration)
        rate = self.warm_up_rate + (1 - self.warm_up_rate) * self.step / self.warm_up_duration
        for i, params in enumerate(self.optimizer.param_groups):
            params['lr'] = self.base_lrs[i] * rate
           
    def lr_schedule(self, epoch):
        if epoch < 13:
            return
        elif epoch < 16:
            rate = 0.1
        else:
            rate = 0.01
        for i, params in enumerate(self.optimizer.param_groups):
            params['lr'] = self.base_lrs[i] * rate
        print(self.optimizer)
    
    def forward(self, img_tensor, scale, bboxes=None, labels=None, force_eval=False):
        img_tensor = img_tensor.to(self.conf.device)
        img_size = (img_tensor.shape[2], img_tensor.shape[3]) # H,W
        rpn_feature, roi_feature = self.extractor(img_tensor)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(rpn_feature, img_size, scale)
        if self.training or force_eval:
            gt_rpn_loc, gt_rpn_labels = self.anchor_target_creator(bboxes, anchor, img_size)
            gt_rpn_labels = torch.tensor(gt_rpn_labels, dtype=torch.long).to(self.conf.device)
            if len(bboxes) == 0:                
                rpn_cls_loss = F.cross_entropy(rpn_scores[0], gt_rpn_labels, ignore_index = -1)
                return self.train_outputs(rpn_cls_loss, 0, 0, 0, 0, 0, 0)
            sample_roi, gt_roi_locs, gt_roi_labels = self.proposal_target_creator(rois, bboxes, labels)
            roi_cls_locs, roi_scores = self.head(roi_feature, sample_roi)
#             roi_cls_locs, roi_scores, pool, h, rois = self.head(roi_feature, sample_roi)
            
            gt_rpn_loc = torch.tensor(gt_rpn_loc, dtype=torch.float).to(self.conf.device)
            gt_roi_locs = torch.tensor(gt_roi_locs, dtype=torch.float).to(self.conf.device)
            gt_roi_labels = torch.tensor(gt_roi_labels, dtype=torch.long).to(self.conf.device)
            
            rpn_loc_loss = fast_rcnn_loc_loss(rpn_locs[0], gt_rpn_loc, gt_rpn_labels, sigma=self.conf.rpn_sigma)
            
            rpn_cls_loss = F.cross_entropy(rpn_scores[0], gt_rpn_labels, ignore_index = -1)
            
            ohem_roi_loc_loss, \
            ohem_roi_cls_loss, \
            total_roi_loc_loss, \
            total_roi_cls_loss = OHEM_loss(roi_cls_locs, 
                                           roi_scores, 
                                           gt_roi_locs, 
                                           gt_roi_labels, 
                                           self.conf.n_ohem_sample, 
                                           self.conf.roi_sigma)
            
            loss_total = rpn_loc_loss + rpn_cls_loss + ohem_roi_loc_loss + ohem_roi_cls_loss
            
#             if loss_total.item() > 1000.:
#                 print('ohem_roi_loc_loss : {}, ohem_roi_cls_loss : {}'.format(ohem_roi_loc_loss, ohem_roi_cls_loss))
#                 torch.save(pool, 'pool_debug.pth')
#                 torch.save(h, 'h_debug.pth')
#                 np.save('rois_debug', rois)
#                 torch.save(roi_cls_locs, 'roi_cls_locs_debug.pth')
#                 torch.save(roi_scores, 'roi_scores_debug.pth')
#                 torch.save(gt_roi_locs, 'gt_roi_locs_debug.pth')
#                 torch.save(gt_roi_labels, 'gt_roi_labels_debug.pth')
#                 pdb.set_trace()
            
            return self.train_outputs(loss_total, 
                                      rpn_loc_loss.item(), 
                                      rpn_cls_loss.item(), 
                                      ohem_roi_loc_loss.item(), 
                                      ohem_roi_cls_loss.item(),
                                      total_roi_loc_loss,
                                      total_roi_cls_loss)
        
        else:
            roi_cls_locs, roi_scores = self.head(roi_feature, rois)
            return self.detections(roi_cls_locs, roi_scores, rois)
        
    def eval_predict(self, img, preset = 'evaluate', use_softnms = False):
        if type(img) == list:
            img = img[0]
        img = Image.fromarray(img.transpose(1,2,0).astype('uint8'))
        bboxes, labels, scores = self.predict_on_img(img, preset, use_softnms, original_size = True)
        bboxes = y1x1y2x2_2_x1y1x2y2(bboxes)
        return [bboxes], [labels], [scores]
        
    def predict_on_img(self, img, preset = 'evaluate', use_softnms=False, return_img = False, with_scores = False, original_size = False):
        '''
        inputs :
        imgs : PIL Image
        return : PIL Image (if return_img) or bboxes_group and labels_group
        '''
        self.eval()
        self.use_preset(preset)
        with torch.no_grad():
            orig_size = img.size # W,H
            img = np.asarray(img).transpose(2,0,1)
            img, scale = prepare_img(self.conf, img, -1)
            img = torch.tensor(img).unsqueeze(0)
            img_size = (img.shape[2], img.shape[3]) # H,W
            detections = self.forward(img, scale)
            n_sample = len(detections.roi_cls_locs)
            n_class = self.conf.class_num + 1
            roi_cls_locs = detections.roi_cls_locs.reshape((n_sample, -1, 4)).reshape([-1,4])
            roi_cls_locs = roi_cls_locs * torch.tensor(self.loc_normalize_std, device=self.conf.device) + torch.tensor(self.loc_normalize_mean, device=self.conf.device)
            rois = torch.tensor(detections.rois.repeat(n_class,0), dtype=torch.float).to(self.conf.device)
            raw_cls_bboxes = loc2bbox(rois, roi_cls_locs)
            torch.clamp(raw_cls_bboxes[:,0::2], 0, img_size[1], out = raw_cls_bboxes[:,0::2] )
            torch.clamp(raw_cls_bboxes[:,1::2], 0, img_size[0], out = raw_cls_bboxes[:,1::2] )
            raw_cls_bboxes = raw_cls_bboxes.reshape([n_sample, n_class, 4])
            raw_prob = F.softmax(detections.roi_scores, dim=1)
            bboxes, labels, scores = self._suppress(raw_cls_bboxes, raw_prob, use_softnms)
            if len(bboxes) == len(labels) == len(scores) == 0:
                if not return_img:  
                    return [], [], []
                else:
                    return to_img(self.conf, img[0])
            _, indices = scores.sort(descending=True)
            bboxes = bboxes[indices]
            labels = labels[indices]
            scores = scores[indices]
            if len(bboxes) > self.max_n_predict:
                bboxes = bboxes[:self.max_n_predict]
                labels = labels[:self.max_n_predict]
                scores = scores[:self.max_n_predict]
        # now, implement drawing
        bboxes = bboxes.cpu().numpy()
        labels = labels.cpu().numpy()
        scores = scores.cpu().numpy()
        if original_size:
            bboxes = adjust_bbox(scale, bboxes, detect=True)
        if not return_img:        
            return bboxes, labels, scores
        else:
            if with_scores:
                scores_ = scores
            else:
                scores_ = []
            predicted_img =  to_img(self.conf, img[0])
            if original_size:
                predicted_img = predicted_img.resize(orig_size)
            if len(bboxes) != 0 and len(labels) != 0:
                predicted_img = draw_bbox_class(self.conf, 
                                                predicted_img, 
                                                labels, 
                                                bboxes, 
                                                self.conf.correct_id_2_class, 
                                                self.class_2_color, 
                                                scores = scores_)
            
            return predicted_img
    
    def _suppress(self, raw_cls_bboxes, raw_prob, use_softnms):
        bbox = []
        label = []
        prob = []
        for l in range(1, self.conf.class_num + 1):
            cls_bbox_l = raw_cls_bboxes[:, l, :]
            prob_l = raw_prob[:, l]
            mask = prob_l > self.score_thresh
            if not mask.any():
                continue
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            if use_softnms:
                keep, _  = soft_nms(torch.cat((cls_bbox_l, prob_l.unsqueeze(-1)), dim=1).cpu().numpy(),
                                    Nt = self.conf.softnms_Nt,
                                    method = self.conf.softnms_method,
                                    sigma = self.conf.softnms_sigma,
                                    min_score = self.conf.softnms_min_score)
                keep = keep.tolist()
            else:
#                 prob_l, order = torch.sort(prob_l, descending=True)
#                 cls_bbox_l = cls_bbox_l[order]
                keep = nms(torch.cat((cls_bbox_l, prob_l.unsqueeze(-1)), dim=1), self.nms_thresh).tolist()
            bbox.append(cls_bbox_l[keep])
            # The labels are in [0, 79].
            label.append((l - 1) * torch.ones((len(keep),), dtype = torch.long))
            prob.append(prob_l[keep])
        if len(bbox) == 0:
            print("looks like there is no prediction have a prob larger than thresh")
            return [], [], []
        bbox = torch.cat(bbox)
        label = torch.cat(label)
        prob = torch.cat(prob)
        return bbox, label, prob
    
    def board_scalars(self, 
                      key, 
                      loss_total, 
                      rpn_loc_loss, 
                      rpn_cls_loss, 
                      ohem_roi_loc_loss, 
                      ohem_roi_cls_loss, 
                      total_roi_loc_loss, 
                      total_roi_cls_loss):
        self.writer.add_scalar('{}_loss_total'.format(key), loss_total, self.step)
        self.writer.add_scalar('{}_rpn_loc_loss'.format(key), rpn_loc_loss, self.step)
        self.writer.add_scalar('{}_rpn_cls_loss'.format(key), rpn_cls_loss, self.step)
        self.writer.add_scalar('{}_ohem_roi_loc_loss'.format(key), ohem_roi_loc_loss, self.step)
        self.writer.add_scalar('{}_ohem_roi_cls_loss'.format(key), ohem_roi_cls_loss, self.step)
        self.writer.add_scalar('{}_total_roi_loc_loss'.format(key), total_roi_loc_loss, self.step)
        self.writer.add_scalar('{}_total_roi_cls_loss'.format(key), total_roi_cls_loss, self.step)
    
    def use_preset(self, preset):
        """Use the given preset during prediction.

        This method changes values of :obj:`self.nms_thresh` and
        :obj:`self.score_thresh`. These values are a threshold value
        used for non maximum suppression and a threshold value
        to discard low confidence proposals in :meth:`predict`,
        respectively.

        If the attributes need to be changed to something
        other than the values provided in the presets, please modify
        them by directly accessing the public attributes.

        Args:
            preset ({'visualize', 'evaluate', 'debug'): A string to determine the
                preset to use.

        """
        if preset == 'visualize':
            self.nms_thresh = 0.5
            self.score_thresh = 0.25
            self.max_n_predict = 40
        elif preset == 'evaluate':
            self.nms_thresh = 0.5
            self.score_thresh = 0.0
            self.max_n_predict = 100
#         """
#         We finally replace origi-nal 0.3 threshold with 0.5 for Non-maximum Suppression
#         (NMS). It improves 0.6 points of mmAP by improving the
#         recall rate especially for the crowd cases.
#         """
        elif preset == 'debug':
            self.nms_thresh = 0.5
            self.score_thresh = 0.0
            self.max_n_predict = 10
        else:
            raise ValueError('preset must be visualize or evaluate')
    
    def fit(self, epochs=20, resume=False, from_save_folder=False):
        if resume:
            self.resume_training_load(from_save_folder)
        self.set_training()        
        running_loss = 0.
        running_rpn_loc_loss = 0.
        running_rpn_cls_loss = 0.
        running_ohem_roi_loc_loss = 0.
        running_ohem_roi_cls_loss = 0.
        running_total_roi_loc_loss = 0.
        running_total_roi_cls_loss = 0.
        map05 = None
        val_loss = None
        
        epoch = self.step // self.train_length
        while epoch <= epochs:
            print('start the training of epoch : {}'.format(epoch))
            self.lr_schedule(epoch)
#             for index in tqdm(np.random.permutation(self.train_length), total = self.train_length):
            for index in tqdm(range(self.train_length), total = self.train_length):
                try:
                    inputs = self.train_dataset[index]
                except:
                    print('loading index {} from train dataset failed}'.format(index))
#                     print(self.train_dataset.orig_dataset._datasets[0].id_to_prop[self.train_dataset.orig_dataset._datasets[0].ids[index]])
                    continue
                self.optimizer.zero_grad()
                train_outputs = self.forward(torch.tensor(inputs.img).unsqueeze(0),
                                             inputs.scale,
                                             inputs.bboxes,
                                             inputs.labels)
                train_outputs.loss_total.backward()
                if epoch == 0:
                    if self.step <= self.warm_up_duration:
                        self.lr_warmup()
                self.optimizer.step()
                torch.cuda.empty_cache()
                
                running_loss += train_outputs.loss_total.item()
                running_rpn_loc_loss += train_outputs.rpn_loc_loss
                running_rpn_cls_loss += train_outputs.rpn_cls_loss
                running_ohem_roi_loc_loss += train_outputs.ohem_roi_loc_loss
                running_ohem_roi_cls_loss += train_outputs.ohem_roi_cls_loss
                running_total_roi_loc_loss += train_outputs.total_roi_loc_loss
                running_total_roi_cls_loss += train_outputs.total_roi_cls_loss
                
                if self.step != 0:
                    if self.step % self.board_loss_every == 0:
                        self.board_scalars('train', 
                                           running_loss / self.board_loss_every, 
                                           running_rpn_loc_loss / self.board_loss_every, 
                                           running_rpn_cls_loss / self.board_loss_every,
                                           running_ohem_roi_loc_loss / self.board_loss_every, 
                                           running_ohem_roi_cls_loss / self.board_loss_every,
                                           running_total_roi_loc_loss / self.board_loss_every, 
                                           running_total_roi_cls_loss / self.board_loss_every)
                        running_loss = 0.
                        running_rpn_loc_loss = 0.
                        running_rpn_cls_loss = 0.
                        running_ohem_roi_loc_loss = 0.
                        running_ohem_roi_cls_loss = 0.
                        running_total_roi_loc_loss = 0.
                        running_total_roi_cls_loss = 0.

                    if self.step % self.evaluate_every == 0:
                        val_loss, val_rpn_loc_loss, \
                        val_rpn_cls_loss, \
                        ohem_val_roi_loc_loss, \
                        ohem_val_roi_cls_loss, \
                        total_val_roi_loc_loss, \
                        total_val_roi_cls_loss = self.evaluate(num = self.conf.eva_num_during_training)
                        self.set_training() 
                        self.board_scalars('val', 
                                           val_loss, 
                                           val_rpn_loc_loss, 
                                           val_rpn_cls_loss, 
                                           ohem_val_roi_loc_loss,
                                           ohem_val_roi_cls_loss,
                                           total_val_roi_loc_loss,
                                           total_val_roi_cls_loss)
                    
                    if self.step % self.eva_on_coco_every == 0:
                        try:
                            cocoEval = self.eva_on_coco(limit = self.conf.coco_eva_num_during_training)
                            self.set_training() 
                            map05 = cocoEval[1]
                            mmap = cocoEval[0]
                        except:
                            print('eval on coco failed')
                            map05 = -1
                            mmap = -1
                        self.writer.add_scalar('0.5IoU MAP', map05, self.step)
                        self.writer.add_scalar('0.5::0.9 - MMAP', mmap, self.step)
                    
                    if self.step % self.board_pred_image_every == 0:
                        for i in range(20):
                            img, _, _, _ , _= self.val_dataset.orig_dataset[i]  
                            img = Image.fromarray(img.astype('uint8').transpose(1,2,0))
                            predicted_img = self.predict_on_img(img, preset='visualize', return_img=True, with_scores=True, original_size=True) 
#                             if type(predicted_img) == tuple: 
#                                 self.writer.add_image('pred_image_{}'.format(i), trans.ToTensor()(img), global_step=self.step)
#                             else: ## should be deleted after test
                            self.writer.add_image('pred_image_{}'.format(i), trans.ToTensor()(predicted_img), global_step=self.step)
                            self.set_training()
                    
                    if self.step % self.save_every == 0:
                        try:
                            self.save_state(val_loss, map05)
                        except:
                            print('save state failed')
                            self.step += 1
                            continue
                    
                self.step += 1
            epoch = self.step // self.train_length
            try:
                self.save_state(val_loss, map05, to_save_folder=True)
            except:
                print('save state failed')
    
    def eva_on_coco(self, limit = 1000, preset = 'evaluate', use_softnms = False):
        self.eval() 
        return eva_coco(self.val_dataset.orig_dataset, lambda x : self.eval_predict(x, preset, use_softnms), limit, preset)
    
    def evaluate(self, num=None):
        self.eval()        
        running_loss = 0.
        running_rpn_loc_loss = 0.
        running_rpn_cls_loss = 0.
        running_ohem_roi_loc_loss = 0.
        running_ohem_roi_cls_loss = 0.
        running_total_roi_loc_loss = 0.
        running_total_roi_cls_loss = 0.
        if num == None:
            total_num = self.val_length
        else:
            total_num = num
        with torch.no_grad():
            for index in tqdm(range(total_num)):
                inputs = self.val_dataset[index]
                if inputs.bboxes == []:
                    continue
                val_outputs = self.forward(torch.tensor(inputs.img).unsqueeze(0),
                                           inputs.scale,
                                           inputs.bboxes,
                                           inputs.labels,
                                           force_eval = True)
                running_loss += val_outputs.loss_total.item()
                running_rpn_loc_loss += val_outputs.rpn_loc_loss
                running_rpn_cls_loss += val_outputs.rpn_cls_loss
                running_ohem_roi_loc_loss += val_outputs.ohem_roi_loc_loss
                running_ohem_roi_cls_loss += val_outputs.ohem_roi_cls_loss
                running_total_roi_loc_loss += val_outputs.total_roi_loc_loss
                running_total_roi_cls_loss += val_outputs.total_roi_cls_loss
        return running_loss / total_num, \
                running_rpn_loc_loss / total_num, \
                running_rpn_cls_loss / total_num, \
                running_ohem_roi_loc_loss / total_num, \
                running_ohem_roi_cls_loss / total_num,\
                running_total_roi_loc_loss / total_num, \
                running_total_roi_cls_loss / total_num
    
    def save_state(self, val_loss, map05, to_save_folder=False, model_only=False):
        if to_save_folder:
            save_path = self.conf.work_space/'save'
        else:
            save_path = self.conf.work_space/'model'
        time = get_time()
        torch.save(
            self.state_dict(), save_path /
            ('model_{}_val_loss:{}_map05:{}_step:{}.pth'.format(time,
                                                                val_loss, 
                                                                map05, 
                                                                self.step)))
        if not model_only:
            torch.save(
                self.optimizer.state_dict(), save_path /
                ('optimizer_{}_val_loss:{}_map05:{}_step:{}.pth'.format(time,
                                                                        val_loss, 
                                                                        map05, 
                                                                        self.step)))
    
    def load_state(self, fixed_str, from_save_folder=False, model_only=False):
        if from_save_folder:
            save_path = self.conf.work_space/'save'
        else:
            save_path = self.conf.work_space/'model'          
        self.load_state_dict(torch.load(save_path/'model_{}'.format(fixed_str)))
        print('load model_{}'.format(fixed_str))
        if not model_only:
            self.optimizer.load_state_dict(torch.load(save_path/'optimizer_{}'.format(fixed_str)))
            print('load optimizer_{}'.format(fixed_str))
    
    def resume_training_load(self, from_save_folder=False):
        if from_save_folder:
            save_path = self.conf.work_space/'save'
        else:
            save_path = self.conf.work_space/'model'  
        sorted_files = sorted([*save_path.iterdir()],  key=lambda x: os.path.getmtime(x), reverse=True)
        seeking_flag = True
        index = 0
        while seeking_flag:
            if index > len(sorted_files) - 2:
                break
            file_a = sorted_files[index]
            file_b = sorted_files[index + 1]
            if file_a.name.startswith('model'):
                fix_str = file_a.name[6:]
                self.step = int(fix_str.split(':')[-1].split('.')[0]) + 1
                if file_b.name == ''.join(['optimizer', '_', fix_str]):                    
                    self.load_state(fix_str, from_save_folder)
                    return
                else:
                    index += 1
                    continue
            elif file_a.name.startswith('optimizer'):
                fix_str = file_a.name[10:]
                self.step = int(fix_str.split(':')[-1].split('.')[0]) + 1
                if file_b.name == ''.join(['model', '_', fix_str]):
                    self.load_state(fix_str, from_save_folder)
                    return
                else:
                    index += 1
                    continue
            else:
                index += 1
                continue
        print('no available files founded')
        return      