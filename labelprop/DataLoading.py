import torch
import torch.utils.data as data
import pytorch_lightning as pl
import nibabel as ni
from torch.utils.data import DataLoader,ConcatDataset
import torchio.transforms as tio
from torch.nn import functional as func
import torch


class FullScan(data.Dataset):
    def __init__(self, X, Y,lab='all',shape=256,selected_slices=None,z_axis=-1):
        self.X = X.astype('float32')
        if isinstance(lab,int):
            Y= 1.*(Y==lab)
        self.Y=Y*1.
        self.X = self.norm(torch.from_numpy(self.X),z_axis)[None,...]
        self.Y = torch.from_numpy(self.Y)[None,...]
        if z_axis!=0:
            self.X=torch.moveaxis(self.X,z_axis+1,1)
            self.Y=torch.moveaxis(self.Y,z_axis+1,1)
        
        if isinstance(shape,int): shape=(shape,shape) 
        self.Y=torch.moveaxis(func.one_hot(self.Y.long()),-1,1)
        self.X,self.Y=self.resample(self.X,self.Y,(self.Y.shape[2],shape[0],shape[1]))

        if selected_slices!=None:
            if 'bench' in selected_slices:
                n_slices=selected_slices.split('_')[1]
                #Create annotated dict with label number as key and list of slices as value
                annotated={}
                for lab in range(1,self.Y.shape[1]): annotated[lab]=[]
                for i in range(self.Y.shape[2]):
                    for lab in range(1,self.Y.shape[1]):
                        if self.Y[0,lab,i].sum()>0:
                            annotated[lab].append(i)
                selected_slices={}
                for lab in range(1,self.Y.shape[1]):
                    n=int(n_slices)-1
                    chunk_range=annotated[lab][-1]-annotated[lab][0]
                    # selected_slices[lab]=[annotated[lab][int((len(annotated[lab])-1)*(x/n))] for x in range(n+1)]
                    preselected=[annotated[lab][0]+int(chunk_range*(x/n)) for x in range(n+1)]
                    #Get values of annotated[lab] that are the closest to the preselected values
                    print(preselected)
                    if n_slices=='3':
                        median=int(len(annotated[lab])/2)
                        selected_slices[lab]=[annotated[lab][0],annotated[lab][median],annotated[lab][-1]]
                    else:
                        selected_slices[lab]=[min(annotated[lab], key=lambda x:abs(x-myNumber)) for myNumber in preselected]
                    # if n_slices=='3':
                    #     selected_slices[lab]=[annotated[lab][0],annotated[lab][median],annotated[lab][-1]]
                    # if n_slices=='5':
                        
                    # elif n_slices=='7':
                    #     additionnal_idx=
                    for i in range(self.Y.shape[2]):
                        if i not in selected_slices[lab]:
                            self.Y[:,lab,i]=self.Y[:,lab,i]*0
        self.selected_slices=selected_slices
        # self.Y=torch.moveaxis(func.one_hot(self.Y.long()), -1, 1).float()

    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]
        return x.unsqueeze(0), y
    def resample(self,X,Y,size):
        X=func.interpolate(X[None,...],size,mode='trilinear',align_corners=True)[0]
        Y=func.interpolate(Y*1.0,size,mode='trilinear',align_corners=True)
        return X,Y
    def __len__(self):
        return len(self.Y)
    


    # def norm(self, x,z_axis=-1):
    #     # x=torch.moveaxis(x, z_axis, 0)
    #     # for i in range(x.shape[0]):
    #     #     x[i]=x[i]-x[i].min()/(x[i].max()-x[i].min())
    #     # x=torch.moveaxis(x, 0, z_axis)
    #     x=x-x.min()/(x.max()-x.min())
    #     return x
    def norm(self, x,z_axis):
        norm = tio.RescaleIntensity((0, 1))
        if len(x.shape)==4:
            x = norm(x)
        elif len(x.shape)==3:
            x= norm(x[:, None, ...])[:,0, ...]
        else:
            x = norm(x[None, None, ...])[0, 0, ...]
        return x

class UnsupervisedScan(data.Dataset):
    def __init__(self, X,shape=256,z_axis=-1,name=''):
        self.X = X.astype('float32')
        self.X = self.norm(torch.from_numpy(self.X))[None,None,...]
        print('before interpol',self.X.shape)
        self.name=name
        if z_axis!=0:
            self.X=torch.moveaxis(self.X,z_axis+2,2)
        if isinstance(shape,int): shape=(shape,shape) 
        self.true_shape=(self.X.shape[-2],self.X.shape[-1])
        self.X=func.interpolate(self.X,size=(self.X.shape[2],shape[0],shape[1]),mode='trilinear',align_corners=True)[0]
        print(self.X.shape)

    def __getitem__(self, index):
        x = self.X[index]
        return x.unsqueeze(0)
    def __len__(self):
        return len(self.X)
    def norm(self, x):
        norm = tio.RescaleIntensity((0, 1))
        if len(x.shape)==4:
            x = norm(x)
        elif len(x.shape)==3:
            x= norm(x[:, None, ...])[:,0, ...]
        else:
            x = norm(x[None, None, ...])[0, 0, ...]
        return x



class LabelPropDataModule(pl.LightningDataModule):
    def __init__(self, img_path,mask_path,lab='all',shape=(288,288),selected_slices=None,z_axis=0):
        super().__init__()
        self.img_path=img_path
        self.mask_path=mask_path
        self.shape=shape
        self.lab=lab
        self.selected_slices=selected_slices
        self.z_axis=z_axis
    def setup(self, stage=None):
        if isinstance(self.img_path,str):
            img=ni.load(self.img_path).get_fdata()
            mask=ni.load(self.mask_path).get_fdata()
        else:
            img=self.img_path
            mask=self.mask_path
        self.val_dataset=None
        self.train_dataset=FullScan(img, mask,lab=self.lab,shape=self.shape,selected_slices=self.selected_slices,z_axis=self.z_axis)
        # if self.selected_slices!=None:
        #     self.val_dataset=FullScan(img, mask,lab=self.lab,shape=self.shape,selected_slices=None,z_axis=self.z_axis)
        self.test_dataset=self.train_dataset

    def train_dataloader(self,batch_size=1):
        return DataLoader(self.train_dataset, 1, num_workers=16,pin_memory=True)

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        else:
            return DataLoader(self.val_dataset, 1, num_workers=16,pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, 1, num_workers=16, pin_memory=True)

class PreTrainingDataModule(pl.LightningDataModule):
    def __init__(self, img_list,shape=(288,288),z_axis=0):
        super().__init__()
        self.img_list=img_list
        self.shape=shape
        self.z_axis=z_axis
    def setup(self, stage=None):
        training_scans=[]
        for i,img in enumerate(self.img_list):
            img_name=str(i)
            if isinstance(img,str):
                img_name=img
                img=ni.load(img).get_fdata()

            training_scans.append(UnsupervisedScan(img,shape=self.shape,z_axis=self.z_axis,name=img_name))
        self.train_dataset=ConcatDataset(training_scans)

    def train_dataloader(self,batch_size=None):
        return DataLoader(self.train_dataset, 1, num_workers=8,pin_memory=True,shuffle=False)
