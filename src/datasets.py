import glob
import numpy as np
import cv2
import albumentations as A

import torch
from torch.utils.data import Dataset, DataLoader
from utils import get_label_mask, set_class_values

def get_images(root_path):
    train_images = glob.glob(f'{root_path}/images/train/*')
    train_images.sort()
    train_masks = glob.glob(f'{root_path}/labels/train/*')
    train_masks.sort()
    val_images = glob.glob(f'{root_path}/images/val/*')
    val_images.sort()
    val_masks = glob.glob(f'{root_path}/labels/val/*')
    val_masks.sort()

    return train_images, train_masks, val_images, val_masks

def train_transform(img_size):

    train_image_transform = A.Compose([
        A.Resize(img_size[0], img_size[1], always_apply=True),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.RandomSunFlare(p=0.2),
        A.RandomFog(p=0.2),
        A.Rotate(limit=25)
    ])

    return train_image_transform

def valid_transform(img_size):

    valid_image_transform = A.Compose([
        A.Resize(img_size[0], img_size[1], always_apply=True)
    ])

    return valid_image_transform

class SegmentationDataset(Dataset):
    def __init__(self,
                 image_paths,
                 labels_paths,
                 image_transform,
                 labels_color_list,
                 classes_to_train,
                 all_classes):
        self.image_paths = image_paths
        self.labels_paths = labels_paths
        self.image_transform = image_transform
        self.labels_color_list = labels_color_list
        self.classes_to_train = classes_to_train
        self.all_classes = all_classes
        self.class_values = set_class_values(
        self.all_classes, self.classes_to_train
        )

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype('float32')
        image = image / 255.0
        mask = cv2.imread(self.labels_paths[idx], cv2.IMREAD_COLOR)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB).astype('float32')

        transformed = self.image_transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        
        # Get colored label mask.
        mask = get_label_mask(mask, self.class_values, self.labels_color_list)

        image = np.transpose(image, (2, 0, 1))
        
        image = torch.tensor(image, dtype=torch.float)
        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask
    
def get_dataset(
        train_image_paths,
        train_mask_paths,
        valid_image_paths,
        valid_mask_paths,
        all_classes,
        classes_to_train,
        labels_color_list,
        img_size
):
    train_tfms = train_transform(img_size)
    valid_tfms = valid_transform(img_size)

    train_dataset = SegmentationDataset(
        train_image_paths,
        train_mask_paths,
        train_tfms,
        labels_color_list,
        classes_to_train,
        all_classes
    )
    valid_dataset = SegmentationDataset(
        valid_image_paths,
        valid_mask_paths,
        valid_tfms,
        labels_color_list,
        classes_to_train,
        all_classes
    )
    return train_dataset, valid_dataset

def get_dataloaders(train_dataset, valid_dataset, batch_size):
    
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, drop_last=False
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=batch_size, drop_last=False
    )

    return train_dataloader, valid_dataloader

