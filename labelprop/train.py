
# from pytorch_lightning import Trainer
from lightning import Trainer
from datetime import datetime
import torch
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
# from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import ModelCheckpoint
import monai
from copy import copy,deepcopy
from .lightning_model import LabelProp
from .Pretraining_model import LabelProp as PretrainingModel
from .voxelmorph2d import NCC,SpatialTransformer,VecInt
# from .weight_optimizer import optimize_weights
import plotext as plt
from monai.losses import GlobalMutualInformationLoss
from kornia.losses import HausdorffERLoss,SSIMLoss,MS_SSIMLoss
from .utils import *


def train_and_eval(datamodule, model_PARAMS, max_epochs, ckpt=None):
    """
    Train and evaluate a LabelProp model.

    Args:
        datamodule (DataModule): The data module containing the dataset.
        model_PARAMS (dict): The parameters for the LabelProp model.
        max_epochs (int): The maximum number of epochs to train the model.
        ckpt (str, optional): The path to a checkpoint file to load the model from. Defaults to None.

    Returns:
        tuple: A tuple containing the trained model, the propagated labels (Y_up), the inverse propagated labels (Y_down),
        and the evaluation results.
    """
    
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
        trained_model=model.load_from_checkpoint(ckpt,strict=False)
        #Create a new flow model that matches shape of the new data
        trained_model.registrator.flow=model.registrator.flow
        model=trained_model
    else:
        trainer=Trainer(accelerator="gpu",max_epochs=max_epochs,callbacks=checkpoint_callback)
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

def train(datamodule,model_PARAMS,max_epochs,ckpt=None,pretraining=False,**kwargs):
    """
    Train the model using the given data module and parameters.

    Args:
        datamodule: The data module used for training.
        model_PARAMS: The parameters for the model.
        max_epochs: The maximum number of epochs for training.
        ckpt: The checkpoint path for resuming training (optional).
        pretraining: Whether to perform pretraining (default: False).
        **kwargs: Additional keyword arguments.

    Returns:
        model: The trained model.
        best_ckpt: The path to the best model checkpoint.
    """
    
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
        trained_model=LabelProp.load_from_checkpoint(ckpt,strict=False)
        # #Create a new flow model that matches shape of the new data
        # trained_model.registrator.flow=model.registrator.flow        

        model.registrator.unet_model.load_state_dict(trained_model.registrator.unet_model.state_dict())
        model.registrator.flow.load_state_dict(trained_model.registrator.flow.state_dict())
        # if 'CRF' in trained_model.__dict__:
        #     model.CRF=trained_model.CRF
        # model=trained_model
    gpus="gpu"
    if 'device' in kwargs:
        if kwargs['device']=='cpu': gpus="cpu"

    trainer=Trainer(accelerator=gpus,max_epochs=max_epochs,callbacks=checkpoint_callback)
    trainer.fit(model,datamodule)
    #model=model.load_from_checkpoint(checkpoint_callback.best_model_path)
    best_ckpt=checkpoint_callback.best_model_path
    return model,best_ckpt

def inference(datamodule,model_PARAMS,ckpt,**kwargs):
    """
    Perform inference using a trained LabelProp model.

    Args:
        datamodule (DataModule): The data module used for training.
        model_PARAMS (dict): The parameters used to initialize the LabelProp model.
        ckpt (str): The path to the checkpoint file of the trained model.
        **kwargs: Additional keyword arguments.

    Returns:
        Tuple: A tuple containing the predicted labels for the up, down, and fused directions.
    """

    model=LabelProp(**model_PARAMS)
    trained_model=LabelProp.load_from_checkpoint(ckpt,strict=False)
    #Create a new flow model that matches shape of the new data
    # trained_model.registrator.flow=model.registrator.flow
    model.registrator.unet_model.load_state_dict(trained_model.registrator.unet_model.state_dict())
    model.registrator.flow.load_state_dict(trained_model.registrator.flow.state_dict())
    # if 'CRF' in trained_model.__dict__:
    #     model.CRF=trained_model.CRF
    # model=trained_model
    datamodule.setup('fit')
    tensors=datamodule.train_dataloader().dataset[0]
    X=tensors[0]
    Y=tensors[1]
    hints=None
    if len(tensors)==3:
        hints=tensors[2]

    # weights=get_weights(Y)
    Y_up,Y_down,Y_fused=propagate_by_composition(X,Y,hints,model,**kwargs)
    # Y_fused=fuse_up_and_down(Y_up,Y_down,weights)
    return torch.argmax(Y_up,0),torch.argmax(Y_down,0),torch.argmax(Y_fused,0)