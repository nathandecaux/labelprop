import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import kornia
from .voxelmorph2d import VxmDense,NCC,Grad,Dice
from monai.losses import BendingEnergyLoss,GlobalMutualInformationLoss,DiceLoss,LocalNormalizedCrossCorrelationLoss
from kornia.filters import sobel, gaussian_blur2d,canny,spatial_gradient
# from pytorch_lightning import Trainer
from lightning import Trainer
from torch.utils.data import DataLoader


class MLP(pl.LightningModule):
    def __init__(self,hidden_size=16,learning_rate=1e-5):
        super().__init__()
        self.hidden_size=hidden_size
        self.hidden=nn.Linear(2,self.hidden_size)
        self.predict=nn.Linear(self.hidden_size,1)
        self.activation=nn.Sigmoid()
    
    def forward(self,x):
        x=self.hidden(x)
        x=F.relu(x)
        x=self.predict(x)
        return self.activation(x)
    
    def training_step(self,batch,batch_idx):
        weights,propagated_masks,true_mask=batch
        #weights=(batch_size=1,2) => x = (1, [w_up,w_down] )
        #propagated_masks=((batch_size=1),2,1,H,W) => x = (1, [Y_up,Y_down],1,H,W ) 
        #true_mask=((batch_size=1),2,H,W) => x = (1,C,H,W )
        # w_up = 0.6 w_down = 0.4 => w_up=0.8
        prediction=self.forward(weights) # => pred_w_up
        loss= Dice().loss(prediction*weights[:,0]+(1-prediction)*weights[:,1],true_mask)
        return {'loss':loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr=self.learning_rate)

class WeightsDataset(torch.data.Dataset):
    def __init__(self,weights,Y_up,Y_down,Y_true):
        self.weights=weights[:1].flatten(0,1) # (slices*lab,2)
        self.Y_up=Y_up[:1].flatten(0,1).unsqueeze(1) # (slices*lab,1,H,W)
        self.Y_down=Y_down[:1].flatten(0,1).unsqueeze(1)
        self.Y_true=Y_true[:1].flatten(0,1).unsqueeze(1)
        
    
    def __getitem__(self,idx):
        w=self.weights[idx]
        y_up=self.Y_up[idx]
        y_down=self.Y_down[idx]
        y_true=self.Y_true[idx]
        return w,y_up,y_down,y_true

    def __len__(self):
        return self.weights.shape[0]

def optimize_weights(weights,Y_up,Y_down,Y_true,ckpt=None,learning_rate=1e-5):
    ds=WeightsDataset(weights,Y_up,Y_down,Y_true)
    #Convert ds to a DataLoader
    dl=DataLoader(ds,batch_size=1,shuffle=False)
    model=MLP(learning_rate=learning_rate)
    if ckpt is not None:
        model.load_state_dict(torch.load(ckpt))
    else:
        trainer=Trainer(accelerator="gpu",max_epochs=1)
        trainer.fit(model,dl)
    for lab in range(weights.shape[0])[1:]:
        for slc in range(weights.shape[1]):
            w=weights[lab,slc].unsqueeze(0)
            weights[lab,slc]=model.forward(w)

    return weights


