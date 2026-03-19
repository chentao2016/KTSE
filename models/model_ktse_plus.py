import os, time
import os.path as osp
import argparse
import glob
import random
import pdb
from turtle import pos

import numpy as np
from numpy.core.fromnumeric import size
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import itertools

# Image tools
import cv2
import PIL
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg') 
from torchvision import transforms

import voc12.data
from tools import utils, pyutils
from tools.imutils import save_img, denorm, _crf_with_alpha

# import resnet38d
from networks import resnet38d, vgg16d, PAR
from networks import resnet101
from skimage.measure import label as sk_label
import math
from scipy import ndimage



palette = [0,0,0,  128,0,0,  0,128,0,  128,128,0,  0,0,128,  128,0,128,  0,128,128,  128,128,128,  
					 64,0,0,  192,0,0,  64,128,0,  192,128,0,  64,0,128,  192,0,128,  64,128,128,  192,128,128,  
					 0,64,0,  128,64,0,  0,192,0,  128,192,0,  0,64,128,  128,64,128,  0,192,128,  128,192,128,  
					 64,64,0,  192,64,0,  64,192,0, 192,192,0]

# 定义长宽比离散值
ASPECT_RATIOS = [8, 4, 2, 1, 0.5, 0.25, 0.125]
NUM_EXT = 5  # 边界扩展像素数

def get_paste_mask(refined_pseudo_label, percentile_threshold=0.25):

    B, H, W = refined_pseudo_label.shape
    
    if B == 0:
        return torch.zeros(B, dtype=torch.float32, device=refined_pseudo_label.device)
    
    area_uncer = torch.sum(refined_pseudo_label == 255, dim=(1, 2)).float()  # shape=(B,)
    area_fg = torch.sum((refined_pseudo_label != 0) & (refined_pseudo_label != 255), dim=(1, 2)).float()  # shape=(B,)
    
  
    ratio_uncer = torch.where(
        area_fg > 0,
        area_uncer / area_fg,
        torch.ones_like(area_uncer)  
    )

    threshold_value = torch.quantile(ratio_uncer, percentile_threshold)
    paste_mask = (ratio_uncer > threshold_value).float()
    
    return paste_mask




def smooth_cam_map(cam_map, sigma=2.0):
    cam_map_np = cam_map.detach().cpu().numpy()
    smoothed = ndimage.gaussian_filter(cam_map_np, sigma=sigma)
    return torch.from_numpy(smoothed).to(cam_map.device)

def get_discrete_aspect_ratio(width, height):
    if height == 0:
        return float('inf')
    aspect_ratio = width / height
    area_value = width * height
    return aspect_ratio, area_value

def get_seg_loss2(pred, label, ignore_index=255):
    bg_label = label.clone()
    bg_label[label!=0] = ignore_index
    bg_loss = F.cross_entropy(pred, bg_label.type(torch.long), ignore_index=ignore_index)
    fg_label = label.clone()
    fg_label[label==0] = ignore_index
    fg_loss = F.cross_entropy(pred, fg_label.type(torch.long), ignore_index=ignore_index)

    return bg_loss  + fg_loss * 0.0125

def get_seg_loss(pred, label, cls_label, cam_ori_fg, ignore_index=255):
    bg_label = label.clone()
    bg_label[label!=0] = ignore_index
    bg_loss = F.cross_entropy(pred, bg_label.type(torch.long), ignore_index=ignore_index)
    seg_loss = bg_loss
    fg_label = label.clone()

    num_class = cls_label.sum(1)



    simple_ind_batch = (num_class==1)
    if simple_ind_batch.sum()>0:
        simple_ind_batch = torch.nonzero(simple_ind_batch).squeeze(1)
        simple_pred = pred[simple_ind_batch]
        simple_fg_label =  fg_label[simple_ind_batch]
        simple_fg_label[simple_fg_label==0] = ignore_index
        fg_loss = F.cross_entropy(simple_pred, simple_fg_label.type(torch.long), ignore_index=ignore_index)
        seg_loss += 0.0125 * fg_loss
    else:
        fg_loss = torch.Tensor([0])



    complex_ind_batch = (num_class>1)
    if complex_ind_batch.sum()>0:
        complex_ind_batch = torch.nonzero(complex_ind_batch).squeeze(1)
        complex_cam = pred[:,1:,:,:][complex_ind_batch]
        complex_cam_ori_fg = cam_ori_fg[complex_ind_batch]
        complex_cam_max = torch.max(complex_cam, dim=1)[0]
        mask_max = 1 - (complex_cam == complex_cam.max(dim=1, keepdim=True)[0]).to(dtype=torch.int32)
        max_remove = torch.mul(mask_max, complex_cam)   #remove the max value
        cam_max_2nd = torch.max(max_remove, dim=1)[0]

        mask_2ndmax = (complex_cam_ori_fg > 0.2).squeeze(1)  
        loss_inter = torch.sum((cam_max_2nd - complex_cam_max.detach()) * mask_2ndmax)/(torch.sum(mask_2ndmax) + 1e-5)
        seg_loss += 0.005 * loss_inter
    else:
        loss_inter = torch.Tensor([0])

    return seg_loss, bg_loss, fg_loss, loss_inter 



def denormalize_img(imgs=None, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    _imgs = torch.zeros_like(imgs)
    _imgs[:,0,:,:] = imgs[:,2,:,:] * std[2] + mean[2]
    _imgs[:,1,:,:] = imgs[:,1,:,:] * std[1] + mean[1]
    _imgs[:,2,:,:] = imgs[:,0,:,:] * std[0] + mean[0]
    _imgs = (_imgs*255).type(torch.uint8)

    return _imgs

def denormalize_img2(imgs=None):
    #_imgs = torch.zeros_like(imgs)
    imgs = denormalize_img(imgs)

    return imgs /255.0

def _refine_cams(ref_mod, images, cams, valid_key, orig_size):

    refined_cams = ref_mod(images, cams)
    refined_cams = F.interpolate(refined_cams, size=orig_size, mode="bilinear", align_corners=False)
    refined_label = refined_cams.argmax(dim=1)
    refined_label = valid_key[refined_label]

    return refined_label

def refine_cams_with_bkg_v2(ref_mod=None, images=None, cams=None, cls_labels=None, RCAM_T =None, names = None, label_vis_path= None, down_scale=2):

    cams = cams/( F.adaptive_max_pool2d(cams, (1, 1)) + 1e-5)

    b,_,h,w = images.shape
    _images = F.interpolate(images.float(), size=[h//down_scale, w//down_scale], mode="bilinear", align_corners=False)


    bkg_h = torch.ones(size=(b,1,h,w))*RCAM_T[0]  
    bkg_h = bkg_h.to(cams.device)
    bkg_l = torch.ones(size=(b,1,h,w))*RCAM_T[1]   
    bkg_l = bkg_l.to(cams.device)

    bkg_cls = torch.ones(size=(b,1))
    bkg_cls = bkg_cls.to(cams.device)
    cls_labels = torch.cat((bkg_cls, cls_labels), dim=1)

    refined_label = torch.ones(size=(b, h, w)) * 255   
    refined_label = refined_label.to(cams.device)
    refined_label_h = refined_label.clone()
    refined_label_l = refined_label.clone()

    cams = F.interpolate(cams, size=[h, w], mode="bilinear", align_corners=False)

    
    cams_with_bkg_h = torch.cat((bkg_h, cams), dim=1)
    _cams_with_bkg_h = F.interpolate(cams_with_bkg_h, size=[h//down_scale, w//down_scale], mode="bilinear", align_corners=False)#.softmax(dim=1)
    cams_with_bkg_l = torch.cat((bkg_l, cams), dim=1)
    _cams_with_bkg_l = F.interpolate(cams_with_bkg_l, size=[h//down_scale, w//down_scale], mode="bilinear", align_corners=False)#.softmax(dim=1)

    for idx in range(b):

        valid_key = torch.nonzero(cls_labels[idx,...])[:,0]
        valid_cams_h = _cams_with_bkg_h[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)
        valid_cams_l = _cams_with_bkg_l[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)

        _refined_label_h = _refine_cams(ref_mod=ref_mod, images=_images[[idx],...], cams=valid_cams_h, valid_key=valid_key, orig_size=(h, w))
        _refined_label_l = _refine_cams(ref_mod=ref_mod, images=_images[[idx],...], cams=valid_cams_l, valid_key=valid_key, orig_size=(h, w))
        
        refined_label_h[idx] = _refined_label_h[0]
        refined_label_l[idx] = _refined_label_l[0]

    refined_label = refined_label_h.clone()
    refined_label[refined_label_h == 0] = 255
    refined_label[(refined_label_h + refined_label_l) == 0] = 0



    return refined_label


def set_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

class GlobalAlignLoss(nn.Module):
    def __init__(self):
        super(GlobalAlignLoss, self).__init__()
    def forward(self,anchor,positive):

        return F.l1_loss(positive,anchor)


class model_WSSS():

    def __init__(self, args, logger):

        self.args = args
        self.categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                           'bus', 'car', 'cat', 'chair', 'cow',
                           'diningtable', 'dog', 'horse', 'motorbike', 'person',
                           'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

        # Common things
        self.phase = 'train'
        self.dev = 'cuda'
        self.bce = nn.BCEWithLogitsLoss()
        self.bs = args.batch_size
        self.logger = logger

        # Hyper-parameters
        self.T = args.T  
        self.M = args.M  
        self.TH = args.TH  
        self.W = args.W  
        self.RCAM_T = args.RCAM_T

        # Model attributes
        self.net_names = ['net_main']
        self.base_names = ['cls','kt','seg','inter','global_align','local_align','seg_fg', 'seg_two']
        self.loss_names = ['loss_' + bn for bn in self.base_names]
        self.acc_names = ['acc_' + bn for bn in self.base_names]

        self.nets = []
        self.opts = []

        # Evaluation-related
        self.running_loss = [0] * len(self.loss_names)
        self.right_count = [0] * len(self.acc_names)
        self.wrong_count = [0] * len(self.acc_names)
        self.accs = [0] * len(self.acc_names)
        self.count = 0
        self.num_count = 0

        self.val_wrong = 0
        self.val_right = 0

        # Define networks
        self.net_main = resnet38d.Net_gpp()

        self.global_align_loss = GlobalAlignLoss()

        self.L2 = nn.MSELoss()
        self.L1 = nn.L1Loss()

        self.par = PAR.PAR(num_iter=10, dilations=[1,2,4,8,12,24])


        self.net_main.load_state_dict(resnet38d.convert_mxnet_to_torch('/opt/data/private/weak2025/KTSE-cross_diff_early/pretrained/ilsvrc-cls_rna-a1_cls1000_ep-0001.params'), strict=False)


       

    # Save networks
    def save_model(self, epo, ckpt_path):
        epo_str = str(epo).zfill(3)
        torch.save(self.net_main.module.state_dict(), ckpt_path + '/' + epo_str + 'net_main.pth')

    # Load networks
    def load_model(self, epo, ckpt_path):
        epo_str = str(epo).zfill(3)
        self.net_main.load_state_dict(torch.load(ckpt_path + '/' + epo_str + 'net_main.pth'), strict=True)

        self.net_main = torch.nn.DataParallel(self.net_main.to(self.dev))

    # Set networks' phase (train/eval)
    def set_phase(self, phase):

        if phase == 'train':
            self.phase = 'train'
            for name in self.net_names:
                getattr(self, name).train()
            self.logger.info('Phase : train')

        else:
            self.phase = 'eval'
            for name in self.net_names:
                getattr(self, name).eval()
            self.logger.info('Phase : eval')

    # Set optimizers and upload networks on multi-gpu
    def train_setup(self):

        args = self.args

        param_main = self.net_main.get_parameter_groups()


        self.opt_main = utils.PolyOptimizer([
            {'params': param_main[0], 'lr': 1 * args.lr, 'weight_decay': args.wt_dec},
            {'params': param_main[1], 'lr': 2 * args.lr, 'weight_decay': 0},  
            {'params': param_main[2], 'lr': 10 * args.lr, 'weight_decay': args.wt_dec}, 
            {'params': param_main[3], 'lr': 20 * args.lr, 'weight_decay': 0}  
        ],
            lr=args.lr, weight_decay=args.wt_dec, max_step=args.max_step)

     

        self.logger.info('Poly-optimizer for net_main is defined.')
        self.logger.info('* Base learning rate : ' + str(args.lr))
        self.logger.info('* non-scratch layer weight lr : ' + str(args.lr))
        self.logger.info('* non-scratch layer bias lr : ' + str(2 * args.lr))
        self.logger.info('* scratch layer weight lr : ' + str(10 * args.lr))
        self.logger.info('* scratch layer bias lr : ' + str(20 * args.lr))
        self.logger.info('* Weight decaying : ' + str(args.wt_dec) + ', max step : ' + str(args.max_step))

        

        self.net_main = torch.nn.DataParallel(self.net_main.to(self.dev))
        self.logger.info('Networks are uploaded on multi-gpu.')

        self.nets.append(self.net_main)

    # Unpack data pack from data_loader
    def unpack(self, pack):

        if self.phase == 'train':
            # self.img_o = pack['img_ori'].to(self.dev)  # B x 3 x H x W
            self.img_a = pack['img_a'].to(self.dev)  # B x 3 x H x W
            self.img_p = pack['img_p'].to(self.dev)  # B x 3 x H x W
            self.label = pack['label'].to(self.dev)  # B x 20
            self.name = pack['name']  # list of image names
            self.img_a2 = pack['img_a2'].to(self.dev)  # B x 3 x H x W
            self.img_p2 = pack['img_p2'].to(self.dev)  # B x 3 x H x W
            self.label2 = pack['label2'].to(self.dev)  # B x 20
            self.name2 = pack['name2']  # list of image names


        if self.phase == 'eval':
            self.img = pack[1]
            # To handle MSF dataset
            for i in range(8):
                self.img[i] = self.img[i].to(self.dev)
            self.label = pack[2].to(self.dev)
            self.name = pack[0][0]

        self.split_label()

    # Do forward/backward propagation and call optimizer to update the networks
    def update(self, epo):

        # Tensor dimensions
        B = self.img_a.shape[0]
        H = self.img_a.shape[2]
        W = self.img_a.shape[3]
        C = 20  # Number of cls

        self.B = B
        self.C = C

        p_idx = []
        n_idx = []

        j_list = [x for x in range(B)]

        for i in range(B):
            anchor = torch.nonzero(self.label[i]).detach().cpu().numpy()
            end = j_list[-1]
            for j in j_list:
                left = torch.nonzero(self.label[j]).detach().cpu().numpy()
                intersection = np.intersect1d(anchor, left)
                if j == end and len(p_idx) == i:
                    p_idx.append(i)

                if len(intersection) > 0 and len(p_idx) < (i + 1):  
                    p_idx.append(j)
                if len(intersection) == 0 and len(n_idx) < (i + 1):
                    n_idx.append(j)

                if len(p_idx) == (i + 1) and len(n_idx) == (i + 1):
                    random.shuffle(j_list)
      
        #
        ################################################### Update network ###################################################
        #
        self.img = self.img_a
        self.img2 = self.img_a2


        
        self.opt_main.zero_grad()


        #global_align
        if epo>= self.T:
            cam, self.out, gpp, cam_seg = self.net_main(self.img)
            cam2, self.out2, gpp2, cam_seg2 = self.net_main(self.img2)
        else:
            cam, self.out, gpp, cam_seg = self.net_main(self.img)



        

        cam_ori_norm = self.max_norm(cam)                                                                 #added mask
        cam_ori_fg = torch.max(cam_ori_norm*self.label.view(B,C,1,1),dim=1,keepdim=True)[0] # B 1 H W
        mask_ori_pos = (cam_ori_fg<self.TH[0])                                                            #added mask

        cam_large = F.interpolate(F.relu(cam),size=(H,W),mode='bilinear',align_corners=False)

        cam_norm = self.max_norm(cam_large)
        cam_fg = torch.max(cam_norm*self.label.view(B,C,1,1),dim=1,keepdim=True)[0] # B 1 H W
        

        mask_pos = (cam_fg<self.TH[0])

        #masked_image = self.img*mask_pos

      
        self.loss_cls = self.W[0] * self.bce(self.out[:self.bs], self.label[:self.bs])
                
        loss = self.loss_cls



        



        ################################################### global_align ###################################################
        

        if self.W[1] > 0 and epo>= self.T:

            inputs_denorm = denormalize_img2(self.img.clone())
            inputs_denorm2 = denormalize_img2(self.img2.clone())

            label_vis_path = osp.join('./experiments', 'label_vis')
        
            if not os.path.exists(label_vis_path):
                os.makedirs(label_vis_path)

            refined_pseudo_label = refine_cams_with_bkg_v2(self.par, inputs_denorm, cams=cam, cls_labels=self.label, RCAM_T = self.RCAM_T, names = self.name, label_vis_path = label_vis_path)
            refined_pseudo_label2 = refine_cams_with_bkg_v2(self.par, inputs_denorm2, cams=cam2, cls_labels=self.label2, RCAM_T = self.RCAM_T, names = self.name2, label_vis_path = label_vis_path)

            paste_mask_flag = get_paste_mask(refined_pseudo_label)  
            masked_image = self.img*mask_pos

            if self.W[1] > 0 and epo>= (self.T+1):

                


                #add code begin

                if not hasattr(self, 'memory_bank'):
                    self.memory_bank = {}  
                    self.memory_bank_queue = {}  
                    self.num_cls_mem = 0
                    self.num_im_mem = 0


                for i in range(self.B):
                    if torch.sum(self.label[i]) == 1:
                        cls_idx = torch.argmax(self.label[i]).item()
                        

                        refined_pseudo_label_i = refined_pseudo_label[i]  # H W
                        
                        fg_reliable = ((refined_pseudo_label_i > 0) & (refined_pseudo_label_i != 255)).float()  # H W


                        
                        
                        
                        fg_reliable_np = fg_reliable.cpu().numpy()
                        labeled_array, num_features = sk_label(fg_reliable_np, connectivity=2, return_num=True)
                        
                        if num_features > 0:
                            
                            max_area = 0
                            max_label = 0
                            for lbl in range(1, num_features + 1):
                                area = np.sum(labeled_array == lbl)
                                if area > max_area:
                                    max_area = area
                                    max_label = lbl
                            
                            
                            fg_reliable_np = (labeled_array == max_label).astype(np.float32)
                        
                            dilation_size = 3  
                            kernel = np.ones((dilation_size, dilation_size), dtype=np.float32)
                            fg_reliable_np = ndimage.binary_dilation(fg_reliable_np, structure=kernel).astype(np.float32)
                            
                            fg_reliable = torch.from_numpy(fg_reliable_np).to(fg_reliable.device)
                            
                            
                            fg_area = np.sum(fg_reliable_np)
                            total_area = H * W
                            fg_ratio = fg_area / total_area
                            
                            
                            min_ratio, max_ratio = 0.1, 0.6                                                       
                            if min_ratio <= fg_ratio <= max_ratio:
                                
                                coords = np.where(fg_reliable_np == 1)
                                if len(coords[0]) > 0 and len(coords[1]) > 0:
                                    y_min, y_max = np.min(coords[0]), np.max(coords[0])
                                    x_min, x_max = np.min(coords[1]), np.max(coords[1])
                                    
                                    #
                                    y_min, y_max = max(0, y_min), min(H-1, y_max)
                                    x_min, x_max = max(0, x_min), min(W-1, x_max)
                                    
                                    
                                    width = x_max - x_min + 1
                                    height = y_max - y_min + 1
                                    aspect_ratio, area_value = get_discrete_aspect_ratio(width, height)
                                    
                                    
                                    im_fg = self.img[i:i+1, :, y_min:y_max+1, x_min:x_max+1]
                                    mask_fg = fg_reliable[y_min:y_max+1, x_min:x_max+1].unsqueeze(0).unsqueeze(0)

                                    cam_large_i = cam_large.detach()[i,cls_idx][y_min:y_max+1, x_min:x_max+1]
                                    positive_add = torch.sum(cam_large_i)/total_area
                                    
                                    
                                    
                                    if cls_idx not in self.memory_bank:
                                        self.memory_bank[cls_idx] = []
                                        self.memory_bank_queue[cls_idx] = []
                                        self.num_cls_mem += 1
                                    
                                    
                                    if len(self.memory_bank[cls_idx]) >= 30:  
                                        
                                        removed_item = self.memory_bank_queue[cls_idx].pop(0)
                                        self.memory_bank[cls_idx].remove(removed_item)
                                        self.num_im_mem -= 1
                                    
                                    
                                    item = {
                                        'im_fg': im_fg,
                                        'mask_fg': mask_fg,
                                        'aspect_ratio': aspect_ratio,
                                        'area_value': area_value,
                                        'original_size': (height, width),
                                        'class': cls_idx,
                                        'positive_add': positive_add
                                    }
                                    self.memory_bank[cls_idx].append(item)
                                    self.memory_bank_queue[cls_idx].append(item)
                                    self.num_im_mem += 1

                
                pasted_masked_image = masked_image.clone()
                paste_labels = torch.zeros_like(self.label)  

                
                if self.num_cls_mem >= 10 and self.num_im_mem >= 20:                                                                     
                    for i in range(self.B):
                        if paste_mask_flag[i] == 1:
                           
                            label_i = self.label[i]
                            if torch.sum(label_i) == 1:
                                cam_fg_i = cam_fg[i].squeeze()  # H W
                            else:
                                
                                valid_indices = torch.where(label_i == 1)[0]  
                                
                                valid_indices_list = valid_indices.tolist()
                                k = random.choice(valid_indices_list)
                                cam_fg_i = cam_norm[i, k, :, :]  # H W

                            
                            cam_fg_smoothed = smooth_cam_map(cam_fg_i, sigma=2.0)
                            
                            
                            cam_fg_binary = (cam_fg_smoothed > self.TH[0]).float()  
                            cam_fg_binary_np = cam_fg_binary.cpu().numpy()
                            
                            
                            labeled_array, num_features = sk_label(cam_fg_binary_np, connectivity=2, return_num=True)
                            
                            if num_features > 0:
                                
                                max_val, max_idx = torch.max(cam_fg_smoothed.view(-1), dim=0)
                                max_y, max_x = max_idx // W, max_idx % W
                                max_label = labeled_array[max_y, max_x]
                                
                                if max_label > 0:
                                    
                                    coords = np.where(labeled_array == max_label)
                                    y_min, y_max = np.min(coords[0]), np.max(coords[0])
                                    x_min, x_max = np.min(coords[1]), np.max(coords[1])
                                    
                                   
                                    y_min_ext = max(0, y_min - NUM_EXT)
                                    y_max_ext = min(H-1, y_max + NUM_EXT)
                                    x_min_ext = max(0, x_min - NUM_EXT)
                                    x_max_ext = min(W-1, x_max + NUM_EXT)
                                    
                                    
                                    target_width = x_max_ext - x_min_ext + 1
                                    target_height = y_max_ext - y_min_ext + 1
                                    target_aspect_ratio, target_area_value = get_discrete_aspect_ratio(target_width, target_height)
                                    
                                   
                                    current_classes = torch.where(self.label[i] == 1)[0].tolist()
                                    available_classes = [cls for cls in self.memory_bank.keys() if cls not in current_classes]
                                    
                                    if available_classes:
                                       
                                        selected_cls = random.choice(available_classes)
                                                             
                                        
                                        
                                        available_items = self.memory_bank[selected_cls]
                                        if available_items:
                                            

                                            AREA_WEIGHT = 0.2    
                                            
                                            def calculate_weighted_diff(item):
                                                
                                                ar_diff = abs(math.log(item['aspect_ratio']) - math.log(target_aspect_ratio))
                                                
                                                
                                                area_diff = abs(math.log(item['area_value']) - math.log(target_area_value))
                                                
                                                total_diff = (1 - AREA_WEIGHT) * ar_diff + AREA_WEIGHT * area_diff
                                                return total_diff
                                            
                                            
                                            best_item = min(available_items, key=calculate_weighted_diff)

                                            im_fg = best_item['im_fg']
                                            mask_fg = best_item['mask_fg']
                                            positive_add = best_item['positive_add']
                                            
                                            
                                            im_fg_resized = F.interpolate(im_fg, size=(target_height, target_width), 
                                                                        mode='bilinear', align_corners=False)
                                            mask_fg_resized = F.interpolate(mask_fg, size=(target_height, target_width), 
                                                                        mode='bilinear', align_corners=False)
                                            
                                            
                                            blended_region = im_fg_resized  
                                            pasted_masked_image[i:i+1, :, y_min_ext:y_max_ext+1, x_min_ext:x_max_ext+1] = blended_region
                                            
                                            
                                            paste_labels[i, selected_cls] = 1 * positive_add
                #add code end
                masked_image = pasted_masked_image

            
            if torch.rand(1).item() < 0.1:
                label_vis_path = osp.join('./experiments', 'vis')
                if not os.path.exists(label_vis_path):
                    os.makedirs(label_vis_path)
                img_np = denorm(masked_image.clone()[0]).cpu().detach().data.permute(1, 2, 0).numpy()

                
                save_img(osp.join(label_vis_path, str(epo).zfill(3) + '_' + self.name[0] + '.png'), img_np)

            
            cam_mask, self.out_mask, gpp_masked, _ = self.net_main(masked_image) 


            gpp = cam
            gpp_masked = cam_mask

           
           
            gpp_masked_up = F.interpolate(gpp_masked,size=(H,W),mode='bilinear',align_corners=False)
            gpp_up = F.interpolate(gpp,size=(H,W),mode='bilinear',align_corners=False)
            self.loss_local_align =  torch.mean(F.relu(1.3*gpp_masked_up * mask_pos - gpp_up * mask_pos - 0.05)*self.label.view(B,C,1,1))
            

            loss +=  self.loss_local_align 

            


            cam_seg_large = F.interpolate(F.relu(cam_seg),size=(H,W),mode='bilinear',align_corners=False)

            seg_loss, self.loss_seg, self.loss_seg_fg, self.loss_inter = get_seg_loss(cam_seg_large, refined_pseudo_label.type(torch.long), self.label, cam_fg, ignore_index=255)

            loss +=  seg_loss
         

            num_pair = 4

            mix_img_two = torch.cat((self.img[:num_pair], self.img2[:num_pair]), dim=2)                   
            mix_cam_two = torch.cat((cam[:num_pair], cam2[:num_pair]), dim=2)                   
            mix_label = (self.label[:num_pair].type(torch.bool) | self.label2[:num_pair].type(torch.bool)).type(torch.float32)
            cam_two, _, _, cam_seg_two = self.net_main(mix_img_two)                 

            mix_cam_two_up = F.interpolate(F.relu(mix_cam_two),size=(2* H,W),mode='bilinear',align_corners=False)
            cam_two_up = F.interpolate(F.relu(cam_two),size=(2* H,W),mode='bilinear',align_corners=False)
            self.loss_kt =  torch.mean(F.relu(mix_cam_two_up - cam_two_up - 0.05)*mix_label.view(mix_label.shape[0],mix_label.shape[1],1,1))
            loss += self.loss_kt

            mix_seg_label_two = torch.cat((refined_pseudo_label[:num_pair], refined_pseudo_label2[:num_pair]), dim=1)                   
            cam_seg_large_two = F.interpolate(F.relu(cam_seg_two),size=(2* H,W),mode='bilinear',align_corners=False)
            self.loss_seg_two = get_seg_loss2(cam_seg_large_two, mix_seg_label_two.type(torch.long), ignore_index=255)
            loss +=  self.loss_seg_two



            anchor = F.adaptive_avg_pool2d(F.relu(gpp),(1,1)).view(B,C)*self.label
            positive = F.adaptive_avg_pool2d(F.relu(gpp_masked),(1,1)).view(B,C)*self.label


            if self.W[1] > 0 and epo>= (self.T+1):
                if self.num_cls_mem >= 10 and self.num_im_mem >= 20:     
                    anchor = anchor + paste_labels  #*0.1
                    self.loss_global_align = self.W[1]*self.global_align_loss(anchor,positive)
                    

                else:
                    self.loss_global_align = self.W[1]*self.global_align_loss(anchor,positive)
            else:
                self.loss_global_align = self.W[1]*self.global_align_loss(anchor,positive)
            
            
            
            loss += self.loss_global_align


        else:
            self.loss_kt = torch.Tensor([0])
            self.loss_seg = torch.Tensor([0])
            self.loss_inter = torch.Tensor([0])
            self.loss_global_align = torch.Tensor([0])
            self.loss_local_align = torch.Tensor([0])
            self.loss_seg_two = torch.Tensor([0])
            self.loss_seg_fg = torch.Tensor([0])




        loss.backward()
        self.opt_main.step()
              
       
        ################################################### Export ###################################################


        for i in range(len(self.loss_names)):
            self.running_loss[i] += getattr(self, self.loss_names[i]).item()
        self.count += 1

        #self.count_rw(self.label, self.out, 0)
    
       
    # Initialization for msf-infer
    def infer_init(self):
        n_gpus = torch.cuda.device_count()
        self.net_main_replicas = torch.nn.parallel.replicate(self.net_main.module, list(range(n_gpus)))

    # (Multi-Thread) Infer MSF-CAM and save image/cam_dict/crf_dict
    def infer_multi(self, epo, val_path, dict_path, crf_path, vis=False, dict=False, crf=False):

        if self.phase != 'eval':
            self.set_phase('eval')

        epo_str = str(epo).zfill(3)
        gt = self.label[0].cpu().detach().numpy()
        self.gt_cls = np.nonzero(gt)[0]

        _, _, H, W = self.img[2].shape
        n_gpus = torch.cuda.device_count()

        def _work(i, img):
            with torch.no_grad():
                with torch.cuda.device(i % n_gpus):
                    cam, _ , _,_ = self.net_main_replicas[i % n_gpus](img.cuda())
                    cam = F.upsample(cam, (H, W), mode='bilinear', align_corners=False)[0]
                    cam = F.relu(cam)
                    cam = cam.cpu().numpy() * self.label.clone().cpu().view(20, 1, 1).numpy()

                    if i % 2 == 1:
                        cam = np.flip(cam, axis=-1)
                    return cam

        thread_pool = pyutils.BatchThreader(_work, list(enumerate(self.img)), batch_size=8, prefetch_size=0,
                                            processes=8)
        cam_list = thread_pool.pop_results()
        cam = np.sum(cam_list, axis=0)
        cam_max = np.max(cam, (1, 2), keepdims=True)
        norm_cam = cam / (cam_max + 1e-5)

        self.cam_dict = {}
        for i in range(20):
            if self.label[0, i] > 1e-5:
                self.cam_dict[i] = norm_cam[i]

                

        if vis:
            img_np = denorm(self.img[2][0]).cpu().detach().data.permute(1, 2, 0).numpy()
            for c in self.gt_cls:
                save_img(osp.join(val_path, epo_str + '_' + self.name + '_cam_' + self.categories[c] + '.png'), img_np,
                         norm_cam[c])

        if dict:
            np.save(osp.join(dict_path, self.name + '.npy'), self.cam_dict)

        if crf:
            for a in self.args.alphas:
                crf_dict = _crf_with_alpha(self.cam_dict, self.name, alpha=a)
                np.save(osp.join(crf_path, str(a).zfill(2), self.name + '.npy'), crf_dict)

    # Print loss/accuracy (and re-initialize them)
    def print_log(self, epo, iter):

        loss_str = ''
        acc_str = ''

        for i in range(len(self.loss_names)):
            loss_str += self.loss_names[i] + ' : ' + str(round(self.running_loss[i] / self.count, 5)) + ', '

        for i in range(len(self.acc_names)):
            if self.right_count[i] != 0:
                acc = 100 * self.right_count[i] / (self.right_count[i] + self.wrong_count[i])
                acc_str += self.acc_names[i] + ' : ' + str(round(acc, 2)) + ', '
                self.accs[i] = acc

        self.logger.info(loss_str[:-2])
        self.logger.info(acc_str[:-2])

        self.running_loss = [0] * len(self.loss_names)
        self.right_count = [0] * len(self.acc_names)
        self.wrong_count = [0] * len(self.acc_names)
        self.count = 0

    def count_rw(self, label, out, idx):
        for b in range(self.bs):  # 8
            gt = label[b].cpu().detach().numpy()
            gt_cls = np.nonzero(gt)[0]
            num = len(np.nonzero(gt)[0])
            pred = out[b].cpu().detach().numpy()
            pred_cls = pred.argsort()[-num:][::-1]

            for c in gt_cls:
                if c in pred_cls:
                    self.right_count[idx] += 1
                else:
                    self.wrong_count[idx] += 1

 

    # Max_norm
    def max_norm(self, cam_cp):
        N, C, H, W = cam_cp.size()
        cam_cp = F.relu(cam_cp)
        max_v = torch.max(cam_cp.view(N, C, -1), dim=-1)[0].view(N, C, 1, 1)
        min_v = torch.min(cam_cp.view(N, C, -1), dim=-1)[0].view(N, C, 1, 1)
        cam_cp = F.relu(cam_cp - min_v - 1e-5) / (max_v - min_v + 1e-5)
        return cam_cp

    def cam_l1(self, cam1, cam2):
        return torch.mean(torch.abs(cam2.detach() - cam1))

    def split_label(self):

        bs = self.label.shape[0] if self.phase == 'train' else 1  # self.label.shape[0]
        self.label_exist = torch.zeros(bs, 20).cuda()
        # self.label_remain = self.label.clone()
        for i in range(bs):
            label_idx = torch.nonzero(self.label[i], as_tuple=False)
            rand_idx = torch.randint(0, len(label_idx), (1,))
            target = label_idx[rand_idx][0]
            # self.label_remain[i, target] = 0
            self.label_exist[i, target] = 1
        self.label_remain = self.label - self.label_exist

        self.label_all = self.label  # [:16]
