# LIDC Dataset Module

from torch.utils.data import Dataset
import pandas as pd
import torch
import random
import numpy as np
import glob
import pickle
from typing import Literal, Optional


class LIDC_Dataset(Dataset):
    def __init__(
        self,
        fold:int,
        datapath:str,
        transform=None,
        label_transform=None,
        mode:Literal["train", "val", "test"]="train",
        task:Literal["Classification", "Regression"]="Classification",
        depth:int=1
    ):
        self.task = task
        self.views = ["axial", "coronal", "sagittal"]
        self.transform = transform
        self.label_transform = label_transform
        self.datapath = datapath
        self.fold = fold
        self.mode = mode
        self.depth = depth

        df = pd.read_pickle(f"{self.datapath}/ALL_annotations_df.pkl")
        if self.mode == "train":
            with open(self.datapath+f"splitted_sets/train_fold_{self.fold}.pkl", 'rb') as file:
                train_indices = pickle.load(file)
            self.X_data = df.iloc[train_indices]["path"]
            if self.task == "Regression":
                y_data = df.iloc[train_indices][["subtlety", "calcification",
                                                 "margin", "lobulation",
                                                 "spiculation", "diameter",
                                                 "texture", "sphericity"]].copy()
                self.concepts = y_data.to_numpy()
            elif self.task == "Classification":
                y_data = df.iloc[train_indices]["target"]
                self.targets = y_data.to_numpy() 
                
        elif self.mode == "val" or self.mode == "test":
            with open(self.datapath+f"/splitted_sets/test_fold_{self.fold}.pkl", 'rb') as file:
                train_indices = pickle.load(file)    
            self.X_data = df.iloc[train_indices]["path"]
            if self.task == "Regression":
                y_data = df.iloc[train_indices][["subtlety", "calcification",
                                                 "margin", "lobulation",
                                                 "spiculation", "diameter",
                                                 "texture", "sphericity"]].copy()
                self.concepts = y_data.to_numpy()
                
            elif self.task == "Classification":
                self.y_data = df.iloc[train_indices]["target"]
                self.targets = self.y_data.to_numpy() 
            
        # Dataset is small, so we can load everything to the memory.
        imgs = []
        for elt in self.X_data:
            crop = torch.load(self.datapath + f"crops/{elt}").float()
            imgs.append(crop)
        self.images = imgs

 
    def __len__(self):
        return len(self.X_data)

    
    def process_image(self, nodule_idx:int, view:Optional[str]=None, slice_:Optional[int]=None):
        # Firstly imgs are 3D volumes:
        img = self.images[nodule_idx]

        if self.depth == 1:
            # Then, I extract slice of the volume
            # at the specified view.
            if view == self.views[0]:
                img = img[:, :, slice_]
        
            elif view == self.views[1]:
                img = img[:, slice_, :]
        
            elif view == self.views[2]:
                img = img[slice_, :, :]
        
            if (len(img.shape) < 3):
                # Extracted slices are of shape: (32, 32).
                # There is need to add third dimmension -> channel.
                # After that we have (1, 32, 32).
                img = img.unsqueeze(0)
         
            # As ViT model requires 3 color channels,
            # code below makes 2 more channels by coping original channel.
            img = img.repeat(3,1,1)
            
        elif self.depth > 1:
            # Then I extract stack of slices around the central slice.
            k=self.depth//2
            if view == self.views[0]:
                img = img[:, :, slice_-k:slice_+k+1]
                img = torch.movedim(img, 2, 0)
            elif view == self.views[1]:
                img = img[:, slice_-k:slice_+k+1, :]
                img = torch.movedim(img, 1, 0)
            elif view == self.views[2]:
                img = img[slice_-k:slice_+k+1, :, :]
            img = img.unsqueeze(0)
            
        img = torch.clamp(img, -1000, 400) # Values in tensor are clamped in range (-1000, 400)
            
        # If some image transformations are specified:
        if self.transform is not None:
            img = self.transform(img)
            
        return img.float()


    def __getitem__(self, idx):
        if self.task == "Classification":
            label = self.targets[idx]
            
        elif self.task == "Regression":
            concepts = self.concepts[idx]
            if (self.label_transform is not None):
                scaler = self.label_transform
                concepts = scaler.transform(np.expand_dims(concepts, axis=0))[0]
            label = torch.tensor(concepts).float()

        if self.mode == "train":
            # In training one image of random view and slice is sent to a model.
            view = random.choice(self.views)
            slices = np.linspace(14, 18, 5).astype(int)
            slice_ = random.choice(slices) # one of the middle slices [14, 15, 16, 17, 18] is chosen.
            img = self.process_image(nodule_idx=idx, view=view, slice_=slice_)
            return [img, label]
        else:
            # During evaluation, three views of a middle slice are sent to a model.
            images = []
            for view in self.views:
                img = self.process_image(nodule_idx=idx, view=view, slice_=16)
                images.append(img)    
            return [images, label]
            

    
    def get_target(self, idx):
        target = self.targets[idx]
        return target
    