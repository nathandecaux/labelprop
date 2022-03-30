
from .train import inference,train
from .DataLoading import LabelPropDataModule
import numpy as np
import torch
from torch.nn import functional as func
from os.path import join
import shutil

ckpt_dir='/tmp/checkpoints/'

def resample(Y,size):
    Y=func.interpolate(Y[None,None,...]*1.,size,mode='nearest')[0,0]
    return Y


def propagate_from_ckpt(img,mask,checkpoint,shape=304,z_axis=2,label='all'):

    true_shape=img.shape
    by_composition=True
    n_classes=int(np.max(mask))
    ckpt=join(ckpt_dir,checkpoint)
    shape=(288,288)#torch.load(ckpt)['hyper_parameters']['size']
    losses={'compo-reg-up':True,'compo-reg-down':True,'compo-dice-up':True,'compo-dice-down':True,'bidir-cons-reg':False,'bidir-cons-dice':False}
    model_PARAMS={'n_classes':n_classes,'way':'both','shape':shape,'selected_slices':None,'losses':losses,'by_composition':by_composition}
    print('hey ho')
    #Dataloading
    dm=LabelPropDataModule(img_path=img,mask_path=mask,lab=label,shape=shape,selected_slices=None,z_axis=z_axis)

    #Inference
    Y_up,Y_down,Y_fused=inference(datamodule=dm,model_PARAMS=model_PARAMS,ckpt=ckpt)
    if z_axis!=0:
        Y_up=torch.moveaxis(Y_up,0,z_axis)
        Y_down=torch.moveaxis(Y_down,0,z_axis)
        Y_fused=torch.moveaxis(Y_fused,0,z_axis)
    Y_up=resample(Y_up,true_shape)
    Y_down=resample(Y_down,true_shape)
    Y_fused=resample(Y_fused,true_shape)
    return Y_up.cpu().detach().numpy(),Y_down.cpu().detach().numpy(),Y_fused.cpu().detach().numpy()

def train_and_infer(img,mask,pretrained_ckpt,shape,max_epochs,z_axis=2,output_dir='~/label_prop_checkpoints',name='',pretraining=False):
    way='both'
    true_shape=img.shape
    shape=(shape,shape)
    by_composition=True
    n_classes=len(np.unique(mask))
    losses={'compo-reg-up':True,'compo-reg-down':True,'compo-dice-up':True,'compo-dice-down':True,'bidir-cons-reg':False,'bidir-cons-dice':False}
    model_PARAMS={'n_classes':n_classes,'way':way,'shape':shape,'selected_slices':None,'losses':losses,'by_composition':False}

    #Dataloading
    dm=LabelPropDataModule(img_path=img,mask_path=mask,lab='all',shape=shape,selected_slices=None,z_axis=z_axis)

    #Training and testing
    trained_model,best_ckpt=train(datamodule=dm,model_PARAMS=model_PARAMS,max_epochs=max_epochs,ckpt=pretrained_ckpt,pretraining=pretraining)
    best_ckpt=str(best_ckpt)
    Y_up,Y_down,Y_fused=inference(datamodule=dm,model_PARAMS=model_PARAMS,ckpt=best_ckpt)
    if z_axis!=0:
        Y_up=torch.moveaxis(Y_up,0,z_axis)
        Y_down=torch.moveaxis(Y_down,0,z_axis)
        Y_fused=torch.moveaxis(Y_fused,0,z_axis)
    Y_up=resample(Y_up,true_shape)
    Y_down=resample(Y_down,true_shape)
    Y_fused=resample(Y_fused,true_shape)

    if name=='': name=best_ckpt.split('/')[-1]

    shutil.copyfile(best_ckpt,join(output_dir,f'{name.split(".ckpt")[-1]}.ckpt'))
    return Y_up.cpu().detach().numpy(),Y_down.cpu().detach().numpy(),Y_fused.cpu().detach().numpy()

