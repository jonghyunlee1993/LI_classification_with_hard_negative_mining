import cv2
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


class PatchDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform
           
    def __len__(self):    
        return len(self.df)
    
    def __getitem__(self, idx):
        x = cv2.imread(self.df.loc[idx, "fpath"])
        x = self.transform(image=x)['image']
        
        y = self.df.loc[idx, "label"]
                
        return x, torch.tensor(y).long()

class LIDataLoader:
    def __init__(self, train_df, valid_df, test_df):
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        
    def run(self):
        self.define_augmentation()
        self.define_datasets()
        self.define_balanced_sampler()
        train_dataloader, valid_dataloader, test_dataloader = self.define_dataloaders()
        
        return train_dataloader, valid_dataloader, test_dataloader, self.valid_transform
        
    def define_augmentation(self):
        self.train_transform = A.Compose([ 
            A.Resize(width=224, height=224, p=1.0),
            A.ShiftScaleRotate(scale_limit=(-0.5, 0.5), rotate_limit=(0, 0), shift_limit=(0, 0), p=1),
            
            A.OneOf([
                A.Transpose(),
                A.HorizontalFlip(),
                A.VerticalFlip()
            ], p=0.5),
            
            A.OneOf([
                A.ElasticTransform(),
                A.Rotate(25),
            ], p=0.8),

            A.OneOf([
            A.Blur(),
            A.GaussianBlur(),
            A.GaussNoise(),
            A.MedianBlur()
            ], p=0.2),

            A.OneOf([
            A.RandomBrightnessContrast()
            ], p=0.5),

            A.Normalize(p=1.0),
            ToTensorV2()
        ])

        self.valid_transform = A.Compose([ 
            A.Resize(width=224, height=224, p=1.0),
            A.Normalize(p=1.0),
            ToTensorV2()
        ])
        
    def define_datasets(self):
        self.train_dataset = PatchDataset(self.train_df, self.train_transform)
        self.valid_dataset = PatchDataset(self.valid_df, self.valid_transform)
        self.test_dataset = PatchDataset(self.test_df, self.valid_transform)
        
    def define_balanced_sampler(self):
        counts = np.bincount(self.train_df.label)
        labels_weights = 1. / counts
        weights = labels_weights[self.train_df.label]
        self.sampler = WeightedRandomSampler(weights, len(weights))

    def define_dataloaders(self):
        import os
        n_cores = os.cpu_count()
        print("Maximum number of CPU cores will be assigned ... ")
        print(f"Number of CPU cores: {n_cores}\n")
        
        train_dataloader = DataLoader(self.train_dataset, batch_size=32, sampler=self.sampler,
                                      pin_memory=True, num_workers=n_cores)
        valid_dataloader = DataLoader(self.valid_dataset, batch_size=256, shuffle=False,
                                      pin_memory=True, num_workers=n_cores)
        test_dataloader = DataLoader(self.test_dataset, batch_size=256, shuffle=False,
                                     pin_memory=True, num_workers=n_cores)
        
        return train_dataloader, valid_dataloader, test_dataloader

