### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import glob
from data_prep.renderpose import *

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        ### label maps    
        self.dir_label = os.path.join(opt.dataroot, opt.phase + '_label')              
        self.label_paths = sorted(make_dataset(self.dir_label))
        

        ### real images
        if opt.isTrain or self.opt.shand_gen:
            self.dir_image = os.path.join(opt.dataroot, opt.phase + '_img')  
            self.image_paths = sorted(make_dataset(self.dir_image))

        ### load face bounding box coordinates size 128x128
        # if opt.face_discrim or opt.face_generator:
        #     self.dir_facetext = os.path.join(opt.dataroot, opt.phase + '_facetexts128')
        #     print('----------- loading face bounding boxes from %s ----------' % self.dir_facetext)
        #     self.facetext_paths = sorted(make_dataset(self.dir_facetext))

        ### load hand keypoints
        # if opt.train_hand:
        #     _train_keypoint_path = os.path.join(opt.train_keypoints_dir, "*.json")
        #     self.train_keypoints = glob.glob(_train_keypoint_path)
        #     self.train_keypoints.sort()
        #     print("Train Keypoints Loaded")
        #     _test_keypoint_path = os.path.join(opt.test_keypoints_dir, "*.json")
        #     self.test_keypoints = glob.glob(_test_keypoint_path)
        #     self.test_keypoints.sort()
        #     print("Test Keypoints Loaded")

        self.dataset_size = len(self.label_paths) 
      
    def __getitem__(self, index):        
        ### label maps
        paths = self.label_paths
        label_path = paths[index]              
        label = Image.open(label_path).convert('RGB')        
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label)
        original_label_path = label_path

        image_tensor = next_label = next_image = face_tensor = handpts_real_tensor = handpts_fake_tensor = 0
        ### real images 
        if self.opt.isTrain or self.opt.shand_gen:
            image_path = self.image_paths[index]   
            image = Image.open(image_path).convert('RGB')    
            transform_image = get_transform(self.opt, params)     
            image_tensor = transform_image(image).float()

        is_next = index < len(self) - 1
        if self.opt.gestures:
            is_next = is_next and (index % 64 != 63)

        """ Load the next label, image pair """
        if is_next:

            paths = self.label_paths
            label_path = paths[index+1]              
            label = Image.open(label_path).convert('RGB')        
            params = get_params(self.opt, label.size)          
            transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            next_label = transform_label(label).float()
            
            if self.opt.isTrain or self.opt.shand_gen:
                image_path = self.image_paths[index+1]   
                image = Image.open(image_path).convert('RGB')
                transform_image = get_transform(self.opt, params)      
                next_image = transform_image(image).float()

        """ If using the face generator and/or face discriminator """
        # if self.opt.face_discrim or self.opt.face_generator:
        #     facetxt_path = self.facetext_paths[index]
        #     facetxt = open(facetxt_path, "r")
        #     face_tensor = torch.IntTensor(list([int(coord_str) for coord_str in facetxt.read().split()]))

        # input_dict = {'label': label_tensor.float(), 'image': image_tensor, 
        #               'path': original_label_path, 'face_coords': face_tensor,
        #               'next_label': next_label, 'next_image': next_image }
        
        """ If using for hand keypoints """
        # if self.opt.train_hand:
        #     _, facepts_r, r_handpts_r, l_handpts_r = readkeypointsfile_json(self.train_keypoints[index])
        #     _, facepts_f, r_handpts_f, l_handpts_f = readkeypointsfile_json(self.test_keypoints[index])
        #     handpts_real = r_handpts_r + l_handpts_r
        #     handpts_fake = r_handpts_f + l_handpts_f
        #     handpts_real_tensor = torch.tensor(handpts_real)
        #     handpts_fake_tensor = torch.tensor(handpts_fake)
        
        # input_dict = {'label': label_tensor.float(), 'image': image_tensor,
        #               'path': original_label_path, 'next_label': next_label,
        #               'next_image': next_image, 'hand_real': handpts_real_tensor, 'hand_fake': handpts_fake_tensor }
        
        input_dict = {'label': label_tensor.float(), 'image': image_tensor, 
                'path': original_label_path, 'next_label': next_label, 'next_image': next_image }
        
        return input_dict

    def __len__(self):
        return len(self.label_paths)

    def name(self):
        return 'AlignedDataset'