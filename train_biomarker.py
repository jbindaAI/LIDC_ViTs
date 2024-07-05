import math
import pickle
import torch
import wandb
from typing import Union, Literal, Optional
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from my_utils.MyRotation import MyRotation
from Biomarker_Model import Biomarker_Model
from LIDC_DataModule import DataModule


## HYPERPARAMETERS:
MODEL_NR:int = 4
WANDB_PROJECT:str = "DINOv2_biom"
MODEL_TYPE:Literal["dino_vits8", "dino_vitb8", "dino_vits16", "dino_vitb16", 
"vit_b_16", "vit_l_16", "dinov2_vits14_reg"]="dinov2_vitb14_reg"
EPOCHS:int = 50
BATCH_SIZE:int = 16
MAX_LR:float = 3e-5
DIV_FACTOR:int = 1000 # Base LR is computed as MAX_LR/DIV_FACTOR.
N_CYCLES:int = 2
TRAINABLE_LAYERS:Union[int, Literal["all"]] = "all"
BCKB_DROPOUT:float = 0.12
LOCAL:bool = False
SAVE_TOP_CKPTS:int = 3


if LOCAL:
    datapath="/home/jbinda/INFORM/LIDC_ViTs/dataset/"
    checkpoints_path="/home/jbinda/INFORM/LIDC_ViTs/ckpt/Biomarkers/"
else:
    datapath="/home/dzban112/LIDC_ViTs/dataset/"
    checkpoints_path="/home/dzban112/LIDC_ViTs/ckpt/Biomarkers/"


for fold in range(1,6): # Iteration over folds
    # Getting value of training steps:
    with open(datapath+f"splitted_sets/train_fold_{fold}.pkl", "rb") as f:
        n_train_examples = len(pickle.load(f))
    steps_per_epoch = math.ceil(n_train_examples/BATCH_SIZE)
    
    # add a checkpoint callback that saves the model with the lowest validation loss
    checkpoint_name = f"{MODEL_TYPE}_{MODEL_NR}_{fold}"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoints_path,
        filename=checkpoint_name,
        save_top_k=SAVE_TOP_CKPTS,
        monitor="val_loss",
        mode="min",
        enable_version_counter=True
    )

    # Logger:
    wandb_logger = WandbLogger(project=WANDB_PROJECT, name=f"{MODEL_TYPE}_{MODEL_NR}_fold_{fold}", job_type='train')
    wandb_logger.experiment.config.update({
        "model_nr": MODEL_NR,
        "model_type": MODEL_TYPE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "max_lr": MAX_LR,
        "div_factor": DIV_FACTOR,
        "n_cycles": N_CYCLES,
        "trainable_layers": TRAINABLE_LAYERS, 
        "backbone_dropout": BCKB_DROPOUT,
        "local": LOCAL
    })
    
    # Cleaning cache:
    torch.cuda.empty_cache()
    
    torch.set_float32_matmul_precision('medium')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(accelerator="gpu", devices=1, 
                         precision=32, max_epochs=EPOCHS,
                         callbacks=[checkpoint_callback, lr_monitor],
                         logger=wandb_logger,
                         log_every_n_steps=25
                        )

    model = Biomarker_Model(
        model_type=MODEL_TYPE,
        trainable_layers=TRAINABLE_LAYERS,
        backbone_dropout=BCKB_DROPOUT,
        max_lr=MAX_LR,
        div_factor=DIV_FACTOR,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        n_cycles=N_CYCLES
    )

    dm = DataModule(
        fold=fold,
        datapath=datapath,
        batch_size=BATCH_SIZE,
        num_workers=8,
        task="Regression"
    )

    trainer.fit(model, dm)
    
    #Finishing run
    wandb.finish()
