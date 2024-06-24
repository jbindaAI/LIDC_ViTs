## Some useful functions

import numpy as np
import torch
import random


def compute_norm_factors(X_train, datapath:str, full_volume:bool=False):
    imgs = []
    for elt in X_train:
        img = torch.load(datapath+f"/crops/{elt}")
        img = torch.clamp(img, -1000, 400).float()

        if not full_volume:
            view = random.choice([1, 2, 3])
            slices = np.linspace(14, 18, 5).astype(int)
            slice_ = random.choice(slices)
            
            if view == 1:
                img = img[:, :, slice_]
            elif view == 2:
                img = img[:, slice_, :]
            elif view == 3:
                img = img[slice_, :, :]
        
            if (len(img.shape) < 3):
                img = img.unsqueeze(0)
        
            img = img.repeat(3,1,1)

        imgs.append(img)
    
    imgs = torch.stack(imgs, axis=0)
    
    mean = torch.mean(imgs)
    std = torch.std(imgs)

    if full_volume:
        mean = [mean for _ in range(32)] # 32 channels
        std = [std for _ in range(32)]
    else:
        mean = [mean for _ in range(3)]
        std = [std for _ in range(3)]
    return [mean, std]