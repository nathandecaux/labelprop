
from datetime import datetime
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from copy import deepcopy
import monai
from copy import copy,deepcopy
from .voxelmorph2d import NCC,SpatialTransformer
# from .weight_optimizer import optimize_weights
import plotext as plt
from monai.losses import GlobalMutualInformationLoss,LocalNormalizedCrossCorrelationLoss
from kornia.losses import HausdorffERLoss,SSIMLoss,MS_SSIMLoss
import json 
from torch.nn import Sigmoid

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class SuperST(torch.nn.Module):
    """Inherit from SpatialTransformer"""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.st = SpatialTransformer(*args, **kwargs)
    
    def compose_deformation(self,flow_i_k,flow_k_j):
        """ Returns flow_k_j(flow_i_k(.)) flow
        Args:
            flow_i_k 
            flow_k_j
        Returns:
            [Tensor]: Flow field flow_i_j = flow_k_j(flow_i_k(.))
        """        
        flow_i_j= flow_k_j+self.st(flow_i_k,flow_k_j)
        return flow_i_j
    
    def compose_list(self,flows):
        flows=list(flows)
        compo=flows[-1]
        for flow in reversed(flows[:-1]):
            compo=self.compose_deformation(flow,compo)
        return compo
    
    def forward(self,x,flow):
        return self.st(x,flow)
    
    def apply_deform(self,x,flow):
        return self.st(x,flow)
def to_one_hot(Y,dim=0):
    return torch.moveaxis(F.one_hot(Y), -1, dim).float()

def to_batch(x,device='cpu'):
    return x[None,None,...].to(device)

def hardmax(Y,dim):
    return torch.moveaxis(F.one_hot(torch.argmax(Y,dim),Y.max()+1), -1, dim)

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
    if len(chunks)==0:
        if len(chunk)==0:
            return None
        else:
            chunk[0]
    else:
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
    """
    Get weights for each slice based on the distance to the closest annotated slice

    Args:
        Y ([Tensor]): Sparsely annotated volume segmentation

    Returns:
        [Tensor]: Weights
    """
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
    """
    Fuse up and down predictions using weights

    Args:
        Y_up ([Tensor]): Up predictions
        Y_down ([Tensor]): Down predictions
        weights ([Tensor]): Weights

    Returns:
        [Tensor]: Fused predictions
    """
    Y=torch.zeros_like(Y_up)

    for lab in list(range(Y.shape[0])):#[1:]:
        # weights[lab]=weights[lab]/weights[lab].sum(1,keepdim=True)
        if lab==1:
            plt.cld()
            if len(weights.shape)>3:
                plt.plot(weights[1,:,0].flatten(1).mean(1).cpu().detach().numpy().tolist())
                plt.plot(weights[1,:,1].flatten(1).mean(1).cpu().detach().numpy().tolist())
            else:
                plt.plot(weights[1,:,0].cpu().detach().numpy().tolist())
                plt.plot(weights[1,:,1].cpu().detach().numpy().tolist())
            plt.show()
        # weights[lab]=F.softmax(weights[lab],1)
        for i,corr_maps in enumerate(weights[lab]):
            w=corr_maps
            Y[lab,i]=w[0]*Y_up[lab,i]+w[1]*Y_down[lab,i]
            # Y[0,i][Y[lab,i]>0.5]+=1-Y[lab,i][Y[lab,i]>0.5]#
            # Y[0,i]+=torch.nn.Sigmoid()(Y[lab:lab+1,i])[0]
    
    Y[0]=1-torch.max(Y[1:],0)[0]
    # Y[0]=1-torch.mean(Y[1:],0)
    # Y[0]=Y[0]/(Y.shape[0]-1)
    # Y[0][(torch.sum(Y_up[1:],0)==0)*1.]
    return Y

def get_successive_fields(X,model):
    """
    Generate successive fields between slices, in both directions

    Args:
        X ([Tensor]): Volume
        model ([type]): Registration model

    Returns:
        [list]: List of fields in one direction
        [list]: List of fields in the other direction
    """

    X=X[0]
    device=X.device
    fields_up=[]
    fields_down=[]
    for i in range(X.shape[0]-1):
        x1=X[i]
        x2=X[i+1]
        fields_up.append(model(to_batch(x1,device),to_batch(x2,device))[1].detach().cpu())
        fields_down.append(model(to_batch(x2,device),to_batch(x1,device))[1].detach().cpu())
    return fields_up,fields_down

def propagate_labels(X,Y,model,model_down=None):
    """
    Deprecated
    """
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

class Normalize(torch.nn.Module):
    def __init__(self, dim=2):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x / torch.sum(x, dim=self.dim, keepdim=True)

def propagate_by_composition(X,Y,hints,model,fields=None,criteria='ncc',reduction='none',func=Normalize(2),device='cuda',return_weights=False,patch_size=31,extrapolate=False):
    """
    Propagate labels by composition of transformations

    Args:
        X (torch.Tensor): input volume
        Y (torch.Tensor): input labels (sparse)
        hints (torch.Tensor): input hints
        model (torch.nn.Module): model to use for registration
        fields (list) (optionnal): list of fields to use for composition (if None, will be computed)
        criteria ('ncc' or 'distance') (optionnal): criteria to use for weighting. Default is ncc
        reduction ('none', 'mean' or 'local_mean') (optionnal): reduction method to use for weighting. Default is none
        func (torch.nn.Module) (optionnal): function to use for normalization. Default is Normalize(2)
        device (str) (optionnal): device to use. Default is 'cuda'
        return_weights (bool) (optionnal): whether to return weights. Default is False
        patch_size (int) (optionnal): patch size to compute NCC. Default is 31 (31x31 px)
        extrapolate (bool) (optionnal): whether to extrapolate propagation to the whole volume. Default is False

    Returns:
        torch.Tensor: propagated labels in a single direction
        torch.Tensor: propagated labels in the other direction
        torch.Tensor: propagated labels in both directions (fused)

        if return_weights:
            torch.Tensor: weights used for fusion

    """
    X=X.to(device)
    Y=Y.to('cpu')
    if hints!=None: hints=hints.to('cpu')
    lab_chunks={}
    model.eval().to(device)
    print(f'Using {criteria} as weighting criteria and {reduction} as reduction method')
    if fields==None:
        fields_up,fields_down=get_successive_fields(X,model)
    else:
        fields_up,fields_down=fields
    X=X[0].to(device)
    fields_up=[f.to(device) for f in fields_up]
    fields_down=[f.to(device) for f in fields_down]
    Y_up=torch.clone(Y).to('cpu')
    Y_down=torch.clone(Y).to('cpu')
    n_classes=Y_up.shape[0]
    if reduction=='none' and criteria!='distance':
        weights=torch.ones((n_classes,Y.shape[1],2,X.shape[-2],X.shape[-1]))*0.5
    else:
        weights=torch.ones((n_classes,Y.shape[1],2))*0.5
    for lab in list(range(n_classes))[1:]:
        print('label : ',lab)
        chunks=get_chunks(binarize(Y,lab))

        if chunks!=None:

            print('Chunks : ',chunks)
            lab_chunks[lab]=chunks
            if not isinstance(chunks,int):
                for chunk in chunks:
                    for i in list(range(*chunk))[1:]:
                        composed_field_up=model.compose_list(fields_up[chunk[0]:i]).to(device)
                        composed_field_down=model.compose_list(fields_down[i:chunk[1]][::-1]).to(device)
                        Y_up[lab:lab+1,i]=model.apply_deform(Y_up[lab:lab+1,i-1].unsqueeze(0).to(device),fields_up[i-1],ismask=True).cpu().detach()[0]
                        # Y_down[lab:lab+1,i]=model.apply_deform(Y[lab:lab+1,chunk[1]].unsqueeze(0).to(device),composed_field_down).cpu().detach()[0]
                        if hints!=None:
                            Y_up[lab:lab+1,i][hints[lab:lab+1,i]==1]=1.
                            Y_up[lab:lab+1,i][hints[-1,i].unsqueeze(0)==1]=0.
                        if criteria=='ncc':
                            w_up=-LocalNormalizedCrossCorrelationLoss(2,kernel_size=patch_size,reduction='none')(model.apply_deform(to_batch(X[chunk[0]],device),composed_field_up),to_batch(X[i],device)).cpu().detach()[0,0]#*(chunk[1]-i)
                            w_down=-LocalNormalizedCrossCorrelationLoss(2,kernel_size=patch_size,reduction='none')(model.apply_deform(to_batch(X[chunk[1]],device),composed_field_down),to_batch(X[i],device)).cpu().detach()[0,0]#*(i-chunk[0])
                            if reduction=='mean':
                                weights[lab,i,0]=w_up.mean()
                                weights[lab,i,1]=w_down.mean()
                            elif reduction=='local_mean':
                                weights[lab,i,0]=w_up[Y_up[lab,i]>0.5].mean()
                                weights[lab,i,1]=w_down[Y_down[lab,i]>0.5].mean()
                            else:
                                weights[lab,i,0]=w_up
                                weights[lab,i,1]=w_down
                        else:   
                            weights[lab,i,1]=torch.tensor(0.5+np.arctan(i-chunk[0]-(chunk[1]-chunk[0])/2)/3.14)#arctan(C(k−(j−i)/2))/π
                            weights[lab,i,0]=1-weights[lab,i,1]
                    for i in list(range(chunk[1],chunk[0],-1))[1:]:
                        Y_down[lab:lab+1,i]=model.apply_deform(Y_down[lab:lab+1,i+1].unsqueeze(0).to(device),fields_down[i],ismask=True).cpu().detach()[0]
                        if hints!=None:
                            Y_down[lab:lab+1,i][hints[lab:lab+1,i]==1]=1.
                            Y_down[lab:lab+1,i][hints[-1,i].unsqueeze(0)==1]=0.
            if hints!=None:
                #Get first and last slice index of hints
                hints_idx=[idx for idx in range(hints.shape[1]) if hints[lab,idx].sum()>0]
                if len(hints_idx)>0:
                    low_hint=hints_idx[0]
                    high_hint=hints_idx[-1]
                    if isinstance(chunks, int):
                        low_annotation=chunks
                        high_annotation=chunks
                    else:
                        low_annotation=chunks[0][0]
                        high_annotation=chunks[-1][1]
                    if low_annotation>low_hint:
                        for i in range(low_annotation,low_hint-1,-1)[1:]:
                            Y_down[lab:lab+1,i]=model.apply_deform(Y_down[lab:lab+1,i+1].unsqueeze(0).to(device),fields_down[i],ismask=True).cpu().detach()[0]
                            Y_down[lab:lab+1,i][hints[lab:lab+1,i]==1]=1.
                            Y_down[lab:lab+1,i][hints[-1,i].unsqueeze(0)==1]=0.
                            weights[lab,i,1]=1.                
                            weights[lab,i,0]=0.   
                    if high_annotation<high_hint:
                        for i in range(high_annotation,high_hint+1)[1:]:
                            Y_up[lab:lab+1,i]=model.apply_deform(Y_up[lab:lab+1,i-1].unsqueeze(0).to(device),fields_up[i-1],ismask=True).cpu().detach()[0]
                            Y_up[lab:lab+1,i][hints[lab:lab+1,i]==1]=1.
                            Y_up[lab:lab+1,i][hints[-1,i].unsqueeze(0)==1]=0.
                            weights[lab,i,0]=1.                
                            weights[lab,i,1]=0.

    if func and criteria != 'distance': weights=func(weights)
    if extrapolate:
        for lab in list(range(n_classes))[1:]:
            chunks=lab_chunks[lab]
            start=np.array(chunks).flatten().min()
            end=np.array(chunks).flatten().max()
            print(lab,start,end)
            for i in range(0,start-1):
                weights[:,i,0]=0
                weights[:,i,1]=1
                composed_field_down=model.compose_list(fields_down[i:start][::-1]).to(device)
                Y_down[lab:lab+1,i]=model.apply_deform(Y[lab:lab+1,start].unsqueeze(0).to(device),composed_field_down,ismask=True).cpu().detach()[0]

            for i in range(end+1,len(fields_up)+1):
                weights[:,i,0]=1
                weights[:,i,1]=0
                composed_field_up=model.compose_list(fields_up[end:i]).to(device)
                Y_up[lab:lab+1,i]=model.apply_deform(Y[lab:lab+1,end].unsqueeze(0).to(device),composed_field_up,ismask=True).cpu().detach()[0]
                
    Y_up[0]=1-torch.max(Y_up[1:],0)[0]
    Y_down[0]=1-torch.max(Y_down[1:],0)[0]
    Y_fused=fuse_up_and_down(Y_up,Y_down,weights)
    if return_weights:
        return Y_up,Y_down,Y_fused,weights
    else:
        return Y_up,Y_down,Y_fused

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

    if len(torch.unique(y_pred))>1:
        dice=monai.metrics.compute_generalized_dice(y_pred, y, include_background=True)
    else:
        dice=torch.from_numpy(np.array([0.]))
            
    # if len(torch.unique(torch.argmax(y_pred,1)))>1:
    #     hauss=med.hd(y_pred.cpu().numpy()>0,y.cpu().numpy()>0)
    #     asd=med.asd(y_pred.cpu().numpy()>0, y.cpu().numpy()>0)
    # else:
    #     hauss=torch.sqrt(torch.tensor(y.shape[-1]^2+y.shape[-2]^2))
    #     asd=torch.sqrt(torch.tensor(y.shape[-1]^2+y.shape[-2]^2))

    return dice,None,None#torch.tensor(hauss),torch.tensor(asd)



def get_dices(Y_dense,Y,Y2,selected_slices):
    """Compute dices for each slice of the volume, and for each label.

    Args:
        Y_dense (torch.Tensor): Dense ground truth segmentation.
        Y (torch.Tensor): Predicted segmentation in one direction.
        Y2 (torch.Tensor): Predicted segmentation in the other direction.
        selected_slices (list): List of index of manually segmented slices that were provided during the inference.
    """
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

