
from .train import inference,train
from.utils import get_successive_fields,propagate_by_composition,SuperST
from .DataLoading import LabelPropDataModule,PreTrainingDataModule,BatchLabelPropDataModule
from .lightning_model import LabelProp
import numpy as np
import torch
from torch.nn import functional as func
from os.path import join
import shutil
import nibabel as ni
import pathlib
import json
import sys

#Get package directory
package_dir=pathlib.Path(__file__).parent.absolute()
#Add package directory to path
sys.path.append(str(package_dir))
#Get user home directory
home=pathlib.Path.home()
#Get ckpt directory from conf.json

def get_ckpt_dir():
    with open(join(package_dir,'conf.json'),'r') as f:
        conf=json.load(f)
    ckpt_dir=conf['checkpoint_dir']
    return ckpt_dir
ckpt_dir=get_ckpt_dir()
pathlib.Path(ckpt_dir).mkdir(parents=True,exist_ok=True)

def resample(Y,size):
    """
    Resample a label tensor to the given size
    """
    Y=torch.moveaxis(func.one_hot(Y.long()),-1,0)
    Y=func.interpolate(Y[None,...]*1.,size,mode='trilinear',align_corners=True)[0]

    return torch.argmax(Y,0)


def propagate_from_ckpt(img,mask,checkpoint,shape=304,z_axis=2,label='all',hints=None,**kwargs):
    """
    Propagate labels using a checkpoint
    """
    ckpt_dir=get_ckpt_dir()
    shape=int(shape/8)*8
    if str(label)=='0': label='all'
    if isinstance(img,str): img=ni.load(img).get_fdata()
    if isinstance(mask,str): mask=ni.load(mask).get_fdata()
    if isinstance(hints, str) : hints=ni.load(hints).get_fdata()
    true_shape=img.shape
    by_composition=True
    n_classes=int(np.max(mask))
    if '/' not in checkpoint:
        checkpoint=join(ckpt_dir,checkpoint)
    shape=(shape,shape)#torch.load(checkpoint)['hyper_parameters']['shape']
    losses={'compo-reg-up':True,'compo-reg-down':True,'compo-dice-up':True,'compo-dice-down':True,'bidir-cons-reg':False,'bidir-cons-dice':False}
    model_PARAMS={'n_classes':n_classes,'way':'both','shape':shape,'selected_slices':None,'losses':losses,'by_composition':False}
    print('hey ho')
    #Dataloading
    dm=LabelPropDataModule(img_path=img,mask_path=mask,lab=label,shape=shape,selected_slices=None,z_axis=z_axis,hints=hints)

    #Inference
    Y_up,Y_down,Y_fused=inference(datamodule=dm,model_PARAMS=model_PARAMS,ckpt=checkpoint,**kwargs)
    if z_axis!=0:
        Y_up=torch.moveaxis(Y_up,0,z_axis)
        Y_down=torch.moveaxis(Y_down,0,z_axis)
        Y_fused=torch.moveaxis(Y_fused,0,z_axis)
    Y_up=resample(Y_up,true_shape)
    Y_down=resample(Y_down,true_shape)
    Y_fused=resample(Y_fused,true_shape)
    return Y_up.cpu().detach().numpy(),Y_down.cpu().detach().numpy(),Y_fused.cpu().detach().numpy()

def train_and_infer(img,mask,pretrained_ckpt,shape,max_epochs,z_axis=2,output_dir='~/label_prop_checkpoints',name='',pretraining=False,hints=None,**kwargs):
    """
    Train a model and propagate provided labels
    """
    ckpt_dir=get_ckpt_dir()
    
    #Check if shape is tuple
    
    shape=int(shape/8)*8

    way='both'
    if pretrained_ckpt!=None:
        if '/' not in pretrained_ckpt:
            pretrained_ckpt=join(ckpt_dir,pretrained_ckpt)
        # shape=torch.load(pretrained_ckpt)['hyper_parameters']['shape'][0]
    if isinstance(img,str): img=ni.load(img).get_fdata()
    if isinstance(mask,str): mask=ni.load(mask).get_fdata()
    if isinstance(hints, str) : hints=ni.load(hints).get_fdata()
    true_shape=img.shape
    n_classes=len(np.unique(mask))
    losses={'compo-reg-up':True,'compo-reg-down':True,'compo-dice-up':True,'compo-dice-down':True,'bidir-cons-reg':False,'bidir-cons-dice':False}
    model_PARAMS={'n_classes':n_classes,'way':way,'shape':shape,'selected_slices':None,'losses':losses,'by_composition':False,'unsupervised':pretraining}

    #Dataloading
    if not pretraining:
        dm=LabelPropDataModule(img_path=img,mask_path=mask,lab='all',shape=shape,selected_slices=None,z_axis=z_axis,hints=hints)
    else:
        dm=PreTrainingDataModule(img_list=[img],shape=shape,z_axis=z_axis)
    #Training and testing
    trained_model,best_ckpt=train(datamodule=dm,model_PARAMS=model_PARAMS,max_epochs=max_epochs,ckpt=pretrained_ckpt,pretraining=pretraining,**kwargs)
    best_ckpt=str(best_ckpt)

    dm=LabelPropDataModule(img_path=img,mask_path=mask,lab='all',shape=shape,selected_slices=None,z_axis=z_axis,hints=hints)
    Y_up,Y_down,Y_fused=inference(datamodule=dm,model_PARAMS=model_PARAMS,ckpt=best_ckpt,**kwargs)
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


def pretrain(img_list,shape,z_axis=2,output_dir='~/label_prop_checkpoints',name='',max_epochs=100):
    """
    Pretrain a model on a list of images
    """
    shape=int(shape/8)*8

    shape=(shape,shape)
    unsupervised=True
    model_PARAMS={'shape':shape,'unsupervised':unsupervised}
    dm=PreTrainingDataModule(img_list=img_list,shape=shape,z_axis=z_axis)
    trained_model,best_ckpt=train(datamodule=dm,model_PARAMS=model_PARAMS,max_epochs=max_epochs)
    best_ckpt=str(best_ckpt)
    if name=='': name=best_ckpt.split('/')[-1]

    # 
    #copy file to output_dir with pathlib, create folder if it doesn't exist
    pathlib.Path(output_dir).mkdir(parents=True,exist_ok=True)
    shutil.copyfile(best_ckpt,join(output_dir,f'{name.split(".ckpt")[-1]}.ckpt'))
    # #Get successive deformation fields
    # trained_model.load_from_checkpoint(checkpoint_path=best_ckpt,device='cuda')
    # for i,scan_dataset in enumerate(dm.train_dataloader().dataset.datasets):
    #     X=scan_dataset[0]
    #     fields_up,fields_down=get_successive_fields(X, trained_model.to('cuda'))
    #     fields_up=torch.stack(tensors=fields_up,dim=0).detach().cpu()
    #     fields_down=torch.stack(tensors=fields_down,dim=0).detach().cpu()
    #     name=scan_dataset.name.split('/')[-1]
    #     #Save fields
    #     torch.save(fields_up,join(output_dir,f'{name}_up.pt'))
    #     torch.save(fields_down,join(output_dir,f'{name}_down.pt'))


def propagate_from_fields(img,mask,fields_up,fields_down,shape,z_axis=2,selected_slices=None,kwargs={}):
    """
    Propagate labels from successive deformation fields
    """
    shape=int(shape/8)*8

    if isinstance(img,str):
        true_shape=ni.load(img).get_fdata().shape
    else:
        true_shape=img.shape
    shape=(shape,shape)
    dm=LabelPropDataModule(img_path=img,mask_path=mask,lab='all',shape=shape,selected_slices=selected_slices,z_axis=z_axis)
    dm.setup()
    st=SuperST(size=shape)
    X,Y=dm.train_dataloader().dataset[0]
    
    if kwargs['return_weights']==True:
        Y_up,Y_down,Y_fused,weights=propagate_by_composition(X, Y, st,(fields_up,fields_down),**kwargs)
    else:
        Y_up,Y_down,Y_fused=propagate_by_composition(X, Y, st,(fields_up,fields_down),**kwargs)
    if z_axis!=0:
        Y_up=torch.moveaxis(torch.argmax(Y_up,0),0,z_axis)
        Y_down=torch.moveaxis(torch.argmax(Y_down,0),0,z_axis)
        Y_fused=torch.moveaxis(torch.argmax(Y_fused,0),0,z_axis)
    # print(Y_up.shape)
    Y_up=resample(Y_up,true_shape)
    Y_down=resample(Y_down,true_shape)
    Y_fused=resample(Y_fused,true_shape)
    if kwargs['return_weights']==False:
        return Y_up.cpu().detach().numpy(),Y_down.cpu().detach().numpy(),Y_fused.cpu().detach().numpy()
    else:
        return Y_up.cpu().detach().numpy(),Y_down.cpu().detach().numpy(),Y_fused.cpu().detach().numpy(),weights.cpu().detach().numpy()

def get_fields(img,ckpt,mask=None,z_axis=2,selected_slices=None):
    """
    Get successive deformation fields from a checkpoint
    """
    shape=torch.load(ckpt)['hyper_parameters']['shape']
    shape=int(shape/8)*8

    if mask==None: mask=img
    dm=LabelPropDataModule(img_path=img,mask_path=mask,lab='all',shape=shape,selected_slices=selected_slices,z_axis=z_axis)
    dm.setup()
    st=SuperST(size=shape)
    X,Y=dm.train_dataloader().dataset[0]
    model=LabelProp(shape=shape).load_from_checkpoint(ckpt,device='cuda')
    fields_up,fields_down=get_successive_fields(X.to('cuda'), model.to('cuda'))
    return torch.cat(fields_up,0).detach().cpu(),torch.cat(fields_down,0).detach().cpu(),X.detach().cpu(),Y.detach().cpu()


def train_dataset(img_list,mask_list,pretrained_ckpt,shape,max_epochs,z_axis=2,output_dir='~/label_prop_checkpoints',name='',**kwargs):
    """
    Train a model on a list of images
    """
    ckpt_dir=get_ckpt_dir()

    shape=int(shape/8)*8
    shape=(shape,shape)
    way='both'
    if pretrained_ckpt!=None:
        if '/' not in pretrained_ckpt:
            pretrained_ckpt=join(ckpt_dir,pretrained_ckpt)
    n_classes=len(np.unique(ni.load(mask_list[0]).get_fdata()))
    losses={'compo-reg-up':True,'compo-reg-down':True,'compo-dice-up':True,'compo-dice-down':True,'bidir-cons-reg':False,'bidir-cons-dice':False}
    model_PARAMS={'n_classes':n_classes,'way':way,'shape':shape,'selected_slices':None,'losses':losses,'by_composition':False,'unsupervised':False}
    dm=BatchLabelPropDataModule(img_list,mask_list,lab='all',shape=shape,z_axis=z_axis)
    trained_model,best_ckpt=train(datamodule=dm,model_PARAMS=model_PARAMS,max_epochs=max_epochs,ckpt=pretrained_ckpt,pretraining=False,**kwargs)
    best_ckpt=str(best_ckpt)
    
    #Save model
    if name=='':
        name=best_ckpt.split('/')[-1]
    shutil.copy(best_ckpt,join(output_dir,name))
    return best_ckpt


