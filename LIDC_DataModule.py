import pytorch_lightning as pl
from LIDC_Dataset import LIDC_Dataset
from my_utils.MyRotation import MyRotation
from typing import Literal
import pickle
from torchvision.transforms import v2
from torch.utils.data import DataLoader

class DataModule(pl.LightningDataModule):
    def __init__(self,
                 datapath:str,
                 fold:int=1,
                 batch_size:int=32,
                 num_workers:int=8,
                 task: Literal['Regression','Classification']="Classification",
                 depth:int=1 
                ):
        super().__init__()
        self.fold = fold
        self.datapath = datapath
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.task = task
        self.depth = depth

    
    def setup(self, stage:str=None):
        # when model is trained on slices
        with open(self.datapath+"splitted_sets"+"/"+"fitted_factors.pkl", 'rb') as f:
            norm_factors = pickle.load(f)
        mean, std, scaler = norm_factors[f"fold_{self.fold}"]
        if self.depth == 1:
            mean = [mean for _ in range(3)]
            std = [std for _ in range(3)]
        else:
            mean = [mean for _ in range(self.depth)]
            std = [std for _ in range(self.depth)]
                     
        train_transform = v2.Compose(
            [
                v2.Resize((224, 224)),
                MyRotation([0, 90, 180, 270]),
                v2.Normalize(mean=mean, std=std)
            ])
            
        val_transform = v2.Compose(
            [
                v2.Resize((224, 224)),
                v2.Normalize(mean=mean, std=std)
            ])
            
        self.train_ds = LIDC_Dataset(
            datapath=self.datapath,
            fold=self.fold,
            transform=train_transform,
            label_transform=scaler,
            mode="train",
            task=self.task,
            depth=self.depth
        )
            
        self.val_ds = LIDC_Dataset(
            datapath=self.datapath,
            fold=self.fold,
            transform=val_transform,
            label_transform=scaler,
            mode="val",
            task=self.task,
            depth=self.depth
        )


    def train_dataloader(self):
        train_loader = DataLoader(self.train_ds,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=self.num_workers
                                 )
        return train_loader

    
    def val_dataloader(self):
        val_loader = DataLoader(self.val_ds,
                                shuffle=False,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers
                               )
        return val_loader

    
    def test_dataloader(self):
        val_loader = DataLoader(self.val_ds,
                                shuffle=False,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers
                               )
        return val_loader