
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
from .LabelProp import LabelProp
from .Pretraining_model import LabelProp as PretrainingModel
import time
def to_one_hot(Y,dim=0):
    return torch.moveaxis(F.one_hot(Y), -1, dim).float()

def to_batch(x,device='cpu'):
    return x[None,None,...].to(device)

def hardmax(Y,dim):
    return torch.moveaxis(F.one_hot(torch.argmax(Y,dim),11), -1, dim)

def get_chunks(Y):
    chunks=[]
    chunk=[]
    #Identifying chunks (i->j)
    for i in range(Y.shape[1]):
        y=Y[:,i]
        if len(torch.unique(torch.argmax(y,1)))>1:
            chunk.append(i)
        if len(chunk)==2:
            chunks.append(chunk)
            chunk=[i]
    return chunks

def binarize(Y,lab):
    return torch.stack([1-Y[lab],Y[lab]],0)

def remove_annotations(Y,selected_slices):
    if selected_slices!=None:
        for i in range(Y.shape[1]):
            if i not in selected_slices:
                Y[:,i,...]=Y[:,i,...]*0
    return Y

def create_dict(keys,values):
    new_dict=dict()
    for k,v in zip(keys,values):
        new_dict[k]=v
    return new_dict

def get_weights(Y):
    flag=False
    weights=torch.zeros((Y.shape[1]))
    n=0
    for i in range(Y.shape[1]):
        if len(torch.unique(torch.argmax(Y[:,i,...],0)))>1:
            if not flag: flag=True
            else: 
                weights[i-(n):i]=weights[i-(n):i]-n/2
                weights[i]=0
                n=1
        else:
            if flag:
                weights[i]=n
                n+=1
    return (torch.arctan(weights)/3.14+0.5)

def fuse_up_and_down(Y_up,Y_down,weights):
    Y=torch.zeros_like(Y_up)
    for i,w in enumerate(weights):
        Y[:,i]=(1-w)*Y_up[:,i]+w*Y_down[:,i]
    return Y

def get_successive_fields(X,model):
    X=X[0]
    fields_up=[]
    fields_down=[]
    for i in range(X.shape[0]-1):
        x1=X[i]
        x2=X[i+1]
        fields_up.append(model(to_batch(x1,'cuda'),to_batch(x2,'cuda'))[1])
        fields_down.append(model(to_batch(x2,'cuda'),to_batch(x1,'cuda'))[1])
    return fields_up,fields_down

def propagate_labels(X,Y,model,model_down=None):
    Y2=deepcopy(Y)
    model.eval().to('cuda')
    model.freeze()
    if model_down==None: model_down=model
    else: model_down.eval()
    X=X[0]
    count_up=0
    count_down=0
    for i,x1 in enumerate(X):
        try:
            x2=X[i+1]
        except:
            print('End of volume')
        else:
            y1=Y[:,i,...]
            if len(torch.unique(torch.argmax(Y[:,i+1,...],0)))==1 and len(torch.unique(torch.argmax(y1,0)))>1:
                _,y,_=model.register_images(to_batch(x1,'cuda'),to_batch(x2,'cuda'),y1[None,...].to('cuda'))
                Y[:,i+1,...]=y.cpu().detach()[0]
                count_up+=1
    for i in range(X.shape[0]-1,1,-1):
        x1=X[i]
        try:
            x2=X[i-1]
        except:
            print('End of volume')
        else:
            y1=Y2[:,i,...]
            if len(torch.unique(torch.argmax(y1,0)))>1 and len(torch.unique(torch.argmax(Y2[:,i-1,...],0)))==1:
                _,y,_=model_down.register_images(to_batch(x1,'cuda'),to_batch(x2,'cuda'),y1[None,...].to('cuda'))
                Y2[:,i-1,...]=y.cpu().detach()[0]
                count_down+=1
    print('counts',count_up,count_down)
    return Y,Y2

def propagate_by_composition(X,Y,model):
    Y_up=torch.clone(Y)
    Y_down=torch.clone(Y)
    n_classes=Y_up.shape[0]
    model.eval().to('cuda')
    model.freeze()
    fields_up,fields_down=get_successive_fields(X,model)
    X=X[0]
    for lab in list(range(n_classes))[1:]:
        print('label : ',lab)
        chunks=get_chunks(binarize(Y,lab))
        print('Chunks : ',chunks)
        
        for chunk in chunks:
            for i in list(range(*chunk))[1:]:
                composed_field_up=model.compose_list(fields_up[chunk[0]:i]).to('cuda')
                composed_field_down=model.compose_list(fields_down[i:chunk[1]][::-1]).to('cuda')
                Y_up[lab:lab+1,i]=model.apply_deform(Y[lab:lab+1,chunk[0]].unsqueeze(0).to('cuda'),composed_field_up).cpu().detach()[0]
                Y_down[lab:lab+1,i]=model.apply_deform(Y[lab:lab+1,chunk[1]].unsqueeze(0).to('cuda'),composed_field_down).cpu().detach()[0]
    Y_up[0]=(torch.sum(Y_up[1:],0)==0)*1.
    Y_down[0]=(torch.sum(Y_down[1:],0)==0)*1.
    return Y_up,Y_down


def complex_propagation(X,Y,model):
    Y_up=torch.clone(Y)
    Y_down=torch.clone(Y)
    n_classes=Y_up.shape[0]
    model.eval().to('cuda')
    model.freeze()
    fields_up,fields_down=get_successive_fields(X,model)
    X=X[0]
    for lab in list(range(n_classes))[1:]:
        print('label : ',lab)
        chunks=get_chunks(binarize(Y,lab))
        print('Chunks : ',chunks)
        
        for chunk in chunks:
            for i in list(range(*chunk))[1:]:
                composed_field_up=model.compose_list(fields_up[chunk[0]:i]).to('cuda')
                composed_field_down=model.compose_list(fields_down[i:chunk[1]][::-1]).to('cuda')
                Y_up[lab:lab+1,i]=model.apply_deform(Y[lab:lab+1,chunk[0]].unsqueeze(0).to('cuda'),composed_field_up).cpu().detach()[0]
                Y_down[lab:lab+1,i]=model.apply_deform(Y[lab:lab+1,chunk[1]].unsqueeze(0).to('cuda'),composed_field_down).cpu().detach()[0]
    Y_up[0]=(torch.sum(Y_up[1:],0)==0)*1.
    Y_down[0]=(torch.sum(Y_down[1:],0)==0)*1.
    return Y_up,Y_down

def compute_metrics(y_pred,y):
    dices=[]
    hausses=[]
    asds=[]
    for c in range(y.shape[1]):
        if len(torch.unique(y[:,c,...]))>1 and c>0:
            if len(torch.unique(y_pred[:,c,...]))>1:
                dice=monai.metrics.compute_meandice(y_pred[:,c:c+1,...], y[:,c:c+1,...], include_background=False)
                dices.append(dice[0])
            else:
                dices.append(torch.from_numpy(np.array([0.])))
            
    if len(torch.unique(torch.argmax(y,1)))>1:
        if len(torch.unique(torch.argmax(y_pred,1)))>1:
            hauss=med.hd(torch.argmax(y_pred,1).numpy()>0,torch.argmax(y,1).numpy()>0)
            asd=med.asd(torch.argmax(y_pred,1).numpy()>0, torch.argmax(y,1).numpy()>0)
        else:
            hauss=torch.sqrt(torch.tensor(y.shape[-1]^2+y.shape[-2]^2))
            asd=torch.sqrt(torch.tensor(y.shape[-1]^2+y.shape[-2]^2))

    dices=torch.stack(dices).mean()
    return dices,hauss,asd

def get_dices(Y_dense,Y,Y2,selected_slices):
    weights=get_weights(remove_annotations(deepcopy(Y_dense),selected_slices))
    dices_up=[]
    hauss_up=[]
    asd_up=[]
    dices_down=[]
    hauss_down=[]
    asd_down=[]
    dices_sum=[]
    hauss_sum=[]
    asd_sum=[]
    dices_weighted=[]
    hauss_weighted=[]
    asd_weighted=[]
    for i in range(Y.shape[1]):
        if i not in selected_slices and len(torch.unique(torch.argmax(Y_dense[:,i,...],0)))>1:
            d_up,h_up,a_up=compute_metrics(hardmax(Y[:,i,...],0)[None,...], Y_dense[:,i,...][None,...])
            d_down,h_down,a_down=compute_metrics(hardmax(Y2[:,i,...],0)[None,...], Y_dense[:,i,...][None,...])
            d_sum,h_sum,a_sum=compute_metrics(hardmax(Y[:,i,...]+Y2[:,i,...],0)[None,...], Y_dense[:,i,...][None,...])
            d_weighted,h_weighted,a_weighted=compute_metrics(hardmax((1-weights[i])*Y[:,i,...]+weights[i]*Y2[:,i,...],0)[None,...], Y_dense[:,i,...][None,...])
            dices_up.append(d_up)
            dices_down.append(d_down)
            dices_sum.append(d_sum)
            dices_weighted.append(d_weighted)
            hauss_up.append(h_up)
            hauss_down.append(h_down)
            hauss_sum.append(h_sum)
            hauss_weighted.append(h_weighted)
            asd_up.append(a_up)
            asd_down.append(a_down)
            asd_sum.append(a_sum)
            asd_weighted.append(a_weighted)

    dice_up=torch.nan_to_num(torch.stack(dices_up)).mean().numpy()
    dice_down=torch.nan_to_num(torch.stack(dices_down)).mean().numpy()
    dice_sum=torch.nan_to_num(torch.stack(dices_sum)).mean().numpy()
    dice_weighted=torch.nan_to_num(torch.stack(dices_weighted)).mean().numpy()
    hauss_up=np.mean(np.array(hauss_up))
    hauss_down=np.mean(np.array(hauss_down))
    hauss_sum=np.mean(np.array(hauss_sum))
    hauss_weighted=np.mean(np.array(hauss_weighted))
    asd_up=np.mean(np.array(asd_up))
    asd_down=np.mean(np.array(asd_down))
    asd_sum=np.mean(np.array(asd_sum))
    asd_weighted=np.mean(np.array(asd_weighted))
    dices=create_dict(['weights','dice_up','dice_down','dice_sum','dice_weighted','hauss_up','hauss_down','hauss_sum','hauss_weighted','asd_up','asd_down','asd_sum','asd_weighted'],[weights,dice_up,dice_down,dice_sum,dice_weighted,hauss_up,hauss_down,hauss_sum,hauss_weighted,asd_up,asd_down,asd_sum,asd_weighted])
    return dices



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
    if pretraining:
        model=PretrainingModel(**model_PARAMS)
    else:
        model=LabelProp(**model_PARAMS)
    if ckpt!=None:
        model=model.load_from_checkpoint(ckpt,strict=False)
    trainer=Trainer(gpus=1,max_epochs=max_epochs,callbacks=checkpoint_callback)
    trainer.fit(model,datamodule)
    model=model.load_from_checkpoint(checkpoint_callback.best_model_path)
    best_ckpt=checkpoint_callback.best_model_path
    return model,best_ckpt

def inference(datamodule,model_PARAMS,ckpt):
    model=LabelProp(**model_PARAMS)
    model=model.load_from_checkpoint(ckpt,strict=False)
    datamodule.setup('fit')
    X,Y=datamodule.train_dataloader().dataset[0]
    weights=get_weights(Y)
    Y_up,Y_down=propagate_by_composition(X,Y,model)
    Y_fused=fuse_up_and_down(Y_up,Y_down,weights)
    return torch.argmax(Y_up,0),torch.argmax(Y_down,0),torch.argmax(Y_fused,0)