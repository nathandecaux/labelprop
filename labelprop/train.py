
from pytorch_lightning import Trainer
from datetime import datetime
import torch
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
from pytorch_lightning.callbacks import ModelCheckpoint
import monai
from copy import copy,deepcopy
import medpy.metric as med
from .lightning_model import LabelProp
from .Pretraining_model import LabelProp as PretrainingModel
from .voxelmorph2d import NCC,SpatialTransformer
# from .weight_optimizer import optimize_weights
import plotext as plt
from monai.losses import GlobalMutualInformationLoss
from kornia.losses import HausdorffERLoss,SSIMLoss,MS_SSIMLoss
from .utils import *


def train_and_eval(datamodule,model_PARAMS,max_epochs,ckpt=None):
    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d%m%Y-%H%M%S")
    dir=f'checkpoints'
    checkpoint_callback = ModelCheckpoint(
        monitor='val_accuracy',
        dirpath=f'{dir}/bench',
        filename='labelprop-{epoch:02d}-{val_accuracy:.2f}-'+dt_string,
        save_top_k=1,
        mode='max',
    )
    model=LabelProp(**model_PARAMS)
    if ckpt!=None:
        model=model.load_from_checkpoint(ckpt,strict=False)
    else:
        trainer=Trainer(gpus=1,max_epochs=max_epochs,callbacks=checkpoint_callback)
        trainer.fit(model,datamodule)
        model=model.load_from_checkpoint(checkpoint_callback.best_model_path)
    datamodule.setup('fit')
    _,Y_dense=datamodule.val_dataloader().dataset[0]
    datamodule.setup('test')
    X,Y=datamodule.test_dataloader().dataset[0]
    Y=remove_annotations(Y,model_PARAMS['selected_slices'])
    # Y_up,Y_down=propagate_labels(X,Y,model)
    Y_up,Y_down=propagate_by_composition(X,Y,model)
    res=get_dices(Y_dense,Y_up,Y_down,model_PARAMS['selected_slices'])
    res['ckpt']=checkpoint_callback.best_model_path if ckpt==None else ckpt
    return model,Y_up,Y_down,res

def train(datamodule,model_PARAMS,max_epochs,ckpt=None,pretraining=False):
    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d%m%Y-%H%M%S")
    dir=f'checkpoints'
    checkpoint_callback = ModelCheckpoint(
        monitor='val_accuracy',
        dirpath=f'{dir}/bench',
        filename='labelprop-{epoch:02d}-{val_accuracy:.2f}-'+dt_string,
        save_top_k=1,
        mode='max',
    )
  
    model=LabelProp(**model_PARAMS)
    if ckpt!=None:
        model=model.load_from_checkpoint(ckpt,strict=False,**model_PARAMS)
    trainer=Trainer(gpus=1,max_epochs=max_epochs,callbacks=checkpoint_callback)
    trainer.fit(model,datamodule)
    #model=model.load_from_checkpoint(checkpoint_callback.best_model_path)
    best_ckpt=checkpoint_callback.best_model_path
    return model,best_ckpt

def inference(datamodule,model_PARAMS,ckpt,**kwargs):
    model=LabelProp(**model_PARAMS)
    model=model.load_from_checkpoint(ckpt,strict=False)
    datamodule.setup('fit')
    X,Y=datamodule.train_dataloader().dataset[0]
    # weights=get_weights(Y)
    Y_up,Y_down,Y_fused=propagate_by_composition(X,Y,model,**kwargs)
    # Y_fused=fuse_up_and_down(Y_up,Y_down,weights)
    return torch.argmax(Y_up,0),torch.argmax(Y_down,0),torch.argmax(Y_fused,0)