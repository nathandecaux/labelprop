import torch
import torch.utils.data as data
import pytorch_lightning as pl
import nibabel as ni
from torch.utils.data import DataLoader
import torchio.transforms as tio
from torch.nn import functional as func
import torch


class FullScan(data.Dataset):
    def __init__(self, X, Y,lab='all',shape=256,selected_slices=None,z_axis=-1):
        self.X = X.astype('float32')
        if isinstance(lab,int):
            Y= 1.*(Y==lab)
        self.Y=Y*1.
        self.X = self.norm(torch.from_numpy(self.X))[None,...]
        self.Y = torch.from_numpy(self.Y)[None,...]
        if z_axis!=0:
            self.X=torch.moveaxis(self.X,z_axis+1,1)
            self.Y=torch.moveaxis(self.Y,z_axis+1,1)
        if isinstance(shape,int): shape=(shape,shape) 
        self.X,self.Y=self.resample(self.X,self.Y,(self.Y.shape[1],shape[0],shape[1]))
        if selected_slices!=None:
            if selected_slices=='bench':
                annotated=[]
                for i in range(self.Y.shape[1]):
                    if len(torch.unique(self.Y[:,i,...]))>1:
                        annotated.append(i)
                median=int(len(annotated)/2)
                selected_slices=[annotated[1],annotated[median],annotated[-2]]
            for i in range(self.Y.shape[1]):
                if i not in selected_slices:
                    self.Y[:,i,...]=self.Y[:,i,...]*0
        self.selected_slices=selected_slices
        self.Y=torch.moveaxis(func.one_hot(self.Y.long()), -1, 1).float()

    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]
        return x.unsqueeze(0), y
    def resample(self,X,Y,size):
        X=func.interpolate(X[None,...],size,mode='trilinear',align_corners=True)[0]
        Y=func.interpolate(Y[None,...],size,mode='nearest')[0]
        return X,Y
    def __len__(self):
        return len(self.Y)

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
        self.train_dataset=FullScan(img, mask,lab=self.lab,shape=self.shape,selected_slices=self.selected_slices,z_axis=self.z_axis)
        self.val_dataset=FullScan(img, mask,lab=self.lab,shape=self.shape,selected_slices=None,z_axis=self.z_axis)
        self.test_dataset=self.train_dataset

    def train_dataloader(self,batch_size=None):
        return DataLoader(self.train_dataset, 1, num_workers=8,pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, 1, num_workers=8,pin_memory=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, 1, num_workers=8, pin_memory=False)
