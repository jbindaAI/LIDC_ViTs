## Some useful functions

import numpy as np
import torch
import random


def compute_norm_factors(X_train, datapath:str):
    imgs = []
    for elt in X_train:
        img = torch.load(datapath+f"/crops/{elt}")
        img = torch.clamp(img, -1000, 400).float()

        view = random.choice([1, 2, 3])
        slices = np.linspace(14, 18, 5).astype(int)
        slice_ = random.choice(slices)
            
        if view == 1:
            img = img[:, :, slice_]
        elif view == 2:
            img = img[:, slice_, :]
        elif view == 3:
            img = img[slice_, :, :]
        
        imgs.append(img)
    
    imgs = torch.stack(imgs, axis=0)
    
    mean = torch.mean(imgs)
    std = torch.std(imgs)

    return [mean, std]