import os
from glob import glob
from pathlib import Path
import shutil
import numpy as np
import csv
import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import torchvision.datasets as datasets
from torchvision.datasets import CIFAR10


class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase=='train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset() # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)
        
        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0]*len(img_paths))
                tot_labels.extend([0]*len(img_paths))
                tot_types.extend(['good']*len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1]*len(img_paths))
                tot_types.extend([defect_type]*len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"
        
        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)
        
        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, os.path.basename(img_path[:-4]), img_type


#class Dataset_maker(torch.utils.data.Dataset):
#    def __init__(self, root, category, config, is_train=True):
#        self.image_transform = transforms.Compose(
#            [
#                transforms.Resize((config.data.image_size, config.data.image_size)),
#                transforms.ToTensor(),  # Scales data into [0,1]
#                transforms.Lambda(lambda t: (t * 2) - 1)  # Scale between [-1, 1]
#            ]
#        )
#        self.config = config
#        self.mask_transform = transforms.Compose(
#            [
#                transforms.Resize((config.data.image_size, config.data.image_size)),
#                transforms.ToTensor(),  # Scales data into [0,1]
#            ]
#        )
#        if is_train:
#            if category:
#                self.image_files = glob(
#                    os.path.join(root, category, "train", "good", "*.png")
#                )
#            else:
#                self.image_files = glob(
#                    os.path.join(root, "train", "good", "*.png")
#                )
#        else:
#            if category:
#                self.image_files = glob(os.path.join(root, category, "test", "*", "*.png"))
#            else:
#                self.image_files = glob(os.path.join(root, "test", "*", "*.png"))
#        self.is_train = is_train
#
#    def __getitem__(self, index):
#        image_file = self.image_files[index]
#        image = Image.open(image_file)
#        image = self.image_transform(image)
#        if (image.shape[0] == 1):
#            image = image.expand(3, self.config.data.image_size, self.config.data.image_size)
#        if self.is_train:
#            label = 'good'
#            return image, label
#        else:
#            if self.config.data.mask:
#                if os.path.dirname(image_file).endswith("good"):
#                    target = torch.zeros([1, image.shape[-2], image.shape[-1]])
#                    label = 'good'
#                else:
#                    if self.config.data.name == 'MVTec':
#                        target = Image.open(
#                            image_file.replace("/test/", "/ground_truth/").replace(
#                                ".png", "_mask.png"
#                            )
#                        )
#                    else:
#                        target = Image.open(
#                            image_file.replace("/test/", "/ground_truth/"))
#                    target = self.mask_transform(target)
#                    label = 'defective'
#            else:
#                if os.path.dirname(image_file).endswith("good"):
#                    target = torch.zeros([1, image.shape[-2], image.shape[-1]])
#                    label = 'good'
#                else:
#                    target = torch.zeros([1, image.shape[-2], image.shape[-1]])
#                    label = 'defective'
#
#            return image, target, label
#
#    def __len__(self):
#        return len(self.image_files)


