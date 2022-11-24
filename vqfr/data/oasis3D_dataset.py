import random
import time
import pdb
import glob
import os
from os import path as osp
from os import listdir
import numpy as np
import monai
import torch
from torch.utils import data as data
from torchvision.transforms.functional import normalize
from scipy.ndimage.interpolation import zoom

from vqfr.data.transforms import augment
from vqfr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from vqfr.utils.registry import DATASET_REGISTRY
from monai.transforms import (
    CastToTyped,
    LoadImaged,
    EnsureTyped,
)

# the dataset for training VAE (reconstruction)
@DATASET_REGISTRY.register()
class OASIS3DDataset(data.Dataset):
    """FFHQ dataset for StyleGAN.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            io_backend (dict): IO backend type and other kwarg.
            mean (list | tuple): Image mean.
            std (list | tuple): Image std.
            use_hflip (bool): Whether to horizontally flip.

    """

    def __init__(self, opt):
        super(OASIS3DDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_folder = opt['dataroot_gt']
        self.mean = opt['mean']
        self.std = opt['std']
        self.istrain = opt['istrain']

        self.path = '/dataF0/Free/tzheng/Registrated_database'
        self.fixedfilenames = [os.path.join(self.path, folder)+'/fixed_scaled_normed.nii.gz' for folder in listdir(self.path)]
        
        if self.istrain == True:
            self.fixedfilenames = self.fixedfilenames[:30]
        else:
            self.fixedfilenames = self.fixedfilenames[-5:]

        self.batchs_percase = 100
        self.patchsize = 256
        self.channel = 1
        self.keys = ("move", "fix")
        train_files = [{self.keys[0]:move, self.keys[1]: fix} for move, fix in zip(self.fixedfilenames, self.fixedfilenames)]
        train_transforms = self.get_xforms("train", self.keys)
        # self.train_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms) # maybe this is not proper in this dataset
        # num_workers has a huge impact on loading speed!
        self.train_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms, num_workers=8) # maybe this is not proper in this dataset
        # now the setting is based on image size = 256
        self._raw_shape = [int(len(self.fixedfilenames) * self.batchs_percase)] + [1,128,128]
        print ('loaded dataset')

    def __getitem__(self, index):
        casenum = index // self.batchs_percase
        img_gt = self.random_volume(self.train_ds[casenum][self.keys[1]])
        return {'gt': torch.from_numpy(img_gt.astype(np.float32)).clone(), 'gt_path': str(casenum)}

    def random_slice(self, array1):
        array1 = np.squeeze(array1)
        slicenum = random.randint(int(array1.shape[2]*0.2), int(array1.shape[2]*0.8))
        slice1 = array1[..., slicenum].cpu().detach().numpy()
        
        slice1 = np.pad(slice1,[(40,40),(24,24)], 'edge')
        slice1 = zoom(slice1, 1.3)
        slice1 = slice1[int((slice1.shape[0] - 256) / 2) : int((slice1.shape[0] - 256) / 2) + 256, int((slice1.shape[1] - 256) / 2) : int((slice1.shape[1] - 256) / 2) + 256]
        slice1 = slice1[np.newaxis,:,:]
        
        return slice1

    def random_volume(self, array1):
        array1 = np.squeeze(array1)
        x = random.randint(0, int(array1.shape[0])-129)
        y = random.randint(0, int(array1.shape[1])-129)
        z = random.randint(0, int(array1.shape[2])-129)
        
        subvol = array1[x:x+128, y:y+128, z:z+128].cpu().detach().numpy()
        subvol = subvol[np.newaxis,:,:]
        
        return subvol
    
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
