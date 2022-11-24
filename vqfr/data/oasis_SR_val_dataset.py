import cv2
import math
import numpy as np
import os.path as osp
import random
import os
from os import listdir
import pdb
import monai
import torch
import torch.utils.data as data
from torchvision.transforms.functional import (adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation,
                                               normalize)
from scipy.ndimage.interpolation import zoom

from vqfr.data import degradations as degradations
from vqfr.data.data_util import paths_from_folder
from vqfr.data.transforms import augment
from vqfr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from vqfr.utils.registry import DATASET_REGISTRY
from monai.transforms import (
    CastToTyped,
    LoadImaged,
    EnsureTyped,
)

@DATASET_REGISTRY.register()
class OASISSRvalDataset(data.Dataset):

    def __init__(self, opt):
        super(OASISSRvalDataset, self).__init__()
        self.opt = opt
        
        self.path = '/dataF0/Free/tzheng/Registrated_database'
        self.movedfilenames = [os.path.join(self.path, folder)+'/moved_scaled_normed.nii.gz' for folder in listdir(self.path)]
        self.fixedfilenames = [os.path.join(self.path, folder)+'/fixed_scaled_normed.nii.gz' for folder in listdir(self.path)]
        
        self.movedfilenames = self.movedfilenames[:2]
        self.fixedfilenames = self.fixedfilenames[:2]
    
        self.batchs_percase = 100
        self.patchsize = 256
        self.channel = 1
        self.keys = ("move", "fix")
        train_files = [{self.keys[0]:move, self.keys[1]: fix} for move, fix in zip(self.movedfilenames, self.fixedfilenames)]
        train_transforms = self.get_xforms("train", self.keys)
        # self.train_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms) # maybe this is not proper in this dataset
        # num_workers has a huge impact on loading speed!
        
        self.train_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms, num_workers=8) # maybe this is not proper in this dataset
        # now the setting is based on image size = 256
        self._raw_shape = [int(len(self.movedfilenames) * self.batchs_percase)] + [1,128,128]
        print ('loaded dataset')

    def get_component_coordinates(self, index, status):
        components_bbox = self.components_list[f'{index:08d}']
        if status[0]:  # hflip
            # exchange right and left eye
            tmp = components_bbox['left_eye']
            components_bbox['left_eye'] = components_bbox['right_eye']
            components_bbox['right_eye'] = tmp
            # modify the width coordinate
            components_bbox['left_eye'][0] = self.out_size - components_bbox['left_eye'][0]
            components_bbox['right_eye'][0] = self.out_size - components_bbox['right_eye'][0]
            components_bbox['mouth'][0] = self.out_size - components_bbox['mouth'][0]

        # get coordinates
        locations = []
        for part in ['left_eye', 'right_eye', 'mouth']:
            mean = components_bbox[part][0:2]
            half_len = components_bbox[part][2]
            if 'eye' in part:
                half_len *= self.eye_enlarge_ratio
            loc = np.hstack((mean - half_len + 1, mean + half_len))
            loc = torch.from_numpy(loc).float()
            locations.append(loc)
        return locations

    def __getitem__(self, index):
        casenum = index // self.batchs_percase
        img1, img2 = self.random_slice_both(self.train_ds[casenum][self.keys[0]], self.train_ds[casenum][self.keys[1]])
        img1 = np.repeat(img1, 3, axis=0)
        img2 = np.repeat(img2, 3, axis=0)
        down_img2 = zoom(img1,(1,0.125,0.125), order=2,mode='reflect')
        down_img2 = zoom(down_img2,(1,8,8), order=2,mode='reflect')
       
        return {'lq': down_img2, 'gt': img2, 'gt_path': str(casenum), 'lq_path': str(casenum)}

    def __len__(self):
        return int(len(self.fixedfilenames) * self.batchs_percase)

    # take both slices from two corresponding arrays
    def random_slice_both(self, array1, array2):
        array1 = np.squeeze(array1)
        array2 = np.squeeze(array2)

        slicenum = random.randint(int(array1.shape[2]*0.2), int(array1.shape[2]*0.8))
        slice1 = array1[..., slicenum].cpu().detach().numpy()
        slice2 = array2[..., slicenum].cpu().detach().numpy()
        slice1 = slice1[np.newaxis,:,:]
        slice2 = slice2[np.newaxis,:,:]
        
        return slice1, slice2

    @staticmethod
    def get_xforms(self, mode="train", keys=("move","fix")):
        """returns a composed transform for train/val/infer."""
        xforms = [
            LoadImaged(keys, dtype=np.float32),
        ]
        dtype = (np.float32, np.float32)
        xforms.extend([CastToTyped(keys, dtype=dtype), EnsureTyped(keys)])
        return monai.transforms.Compose(xforms)