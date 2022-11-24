import os
from os import listdir
import numpy as np
import random

import monai
from scipy.ndimage.interpolation import zoom
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from vqfr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from vqfr.data.transforms import augment, paired_random_crop
from vqfr.utils import FileClient, imfrombytes, img2tensor
from vqfr.utils.matlab_functions import rgb2ycbcr
from vqfr.utils.registry import DATASET_REGISTRY
from monai.transforms import (
    CastToTyped,
    LoadImaged,
    EnsureTyped
)

@DATASET_REGISTRY.register()
class OASISPairedDataset(data.Dataset):
    def __init__(self, opt):
        super(OASISPairedDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.path = '/dataF0/Free/tzheng/Registrated_database'
        self.movedfilenames = [os.path.join(self.path, folder)+'/moved_scaled_normed.nii.gz' for folder in listdir(self.path)]
        self.fixedfilenames = [os.path.join(self.path, folder)+'/fixed_scaled_normed.nii.gz' for folder in listdir(self.path)]
    
        if self.opt['phase'] == 'train':
            self.movedfilenames = self.movedfilenames[:10]
            self.fixedfilenames = self.fixedfilenames[:10]
        else:
            self.movedfilenames = self.movedfilenames[-2:]
            self.fixedfilenames = self.fixedfilenames[-2:]
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

    def __getitem__(self, index):
        casenum = index // self.batchs_percase
        img1, img2 = self.random_slice_both(self.train_ds[casenum][self.keys[0]], self.train_ds[casenum][self.keys[1]])
        img1 = np.repeat(img1, 3, axis=0)
        img2 = np.repeat(img2, 3, axis=0)
        down_img1 = zoom(img1,(1,0.125,0.125), order=2,mode='reflect')
    
        return {'lq': down_img1.astype(np.float32).copy(), 'gt': img1.astype(np.float32).copy(), 'lq_path': str(casenum), 'gt_path': str(casenum)}

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
            # HistogramNormalized(keys, min=-1, max=1),
            # ResizeWithPadOrCropd(keys, spatial_size=(208, 208, 176)) # this is very slow
        ]
        dtype = (np.float32, np.float32)
        xforms.extend([CastToTyped(keys, dtype=dtype), EnsureTyped(keys)])
        return monai.transforms.Compose(xforms)

    def __len__(self):
        return int(len(self.fixedfilenames) * self.batchs_percase)
