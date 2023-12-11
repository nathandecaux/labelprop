import torch
import torch.nn.functional as F
# import pytorch_lightning as pl
import lightning as pl
import kornia
from .voxelmorph2d import VxmDense,NCC,Grad,Dice
from pprint import pprint
from monai.metrics import compute_generalized_dice

from crfseg import CRF


class LabelProp(pl.LightningModule):

    @property
    def automatic_optimization(self):
        return False
    def norm(self, x):
        
        if len(x.shape)==4:
            x = kornia.enhance.normalize_min_max(x)
        elif len(x.shape)==3:
            x= kornia.enhance.normalize_min_max(x[:, None, ...])[:,0, ...]
        else:
            x = kornia.enhance.normalize_min_max(x[None, None, ...])[0, 0, ...]
        return x
    
   


    def __init__(self,n_channels=1,n_classes=2,learning_rate=1e-3,weight_decay=1e-8,way='up',shape=256,selected_slices=None,losses={},by_composition=False):
        super().__init__()
        self.n_classes = n_classes
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        self.selected_slices=selected_slices #Used in validation step 
        if isinstance(shape,int):shape=[shape,shape]
        self.registrator= VxmDense(shape,bidir=False,int_downsize=1,int_steps=7)
        self.way=way #If up, learning only "forward" transitions (phi_i->j with j>i). Other choices : "down", "both". Bet you understood ;)
        self.losses=losses
        self.by_composition=by_composition
        self.delta=1
        self.mean_dice=0
        self.CRF=CRF(2)
        print('Losses',losses)
        self.save_hyperparameters()

    def apply_deform(self,x,field,ismask=False):
        """Apply deformation to x from flow field
        Args:
            x (Tensor): Image or mask to deform (BxCxHxW)
            field (Tensor): Deformation field (Bx2xHxW)
        Returns:
            Tensor: Transformed image
        """        
        x_hat = self.registrator.transformer(x,field)
        # if ismask:
        #     x_hat=self.CRF(x_hat)
        return x_hat
    
    def compose_list(self,flows):
        flows=list(flows)
        compo=flows[-1]
        if len(flows)>1:
            for flow in reversed(flows[:-1]):
                compo=self.compose_deformation(flow,compo)
        return compo
    def compose_deformation(self,flow_i_k,flow_k_j):
        """ Returns flow_k_j(flow_i_k(.)) flow
        Args:
            flow_i_k 
            flow_k_j
        Returns:
            [Tensor]: Flow field flow_i_j = flow_k_j(flow_i_k(.))
        """        
        flow_i_j= flow_k_j+self.apply_deform(flow_i_k,flow_k_j)
        return flow_i_j

    def apply_successive_transformations(self,moving,flows):
        """
        Args:
            moving (Tensor): Moving image (BxCxHxW)
            flows ([Tensor]): List of deformation fields (Bx2xHxW)
        Returns:
            Tensor: Transformed image
        """
        if len(flows)==0:
            return moving
        else:
            return self.apply_deform(self.apply_successive_transformations(moving,flows[:-1]),flows[-1])
        
    def multi_class_dice(self,pred_with_logits,target):
        """
        Args:
            pred_with_logits: Predicted mask with logits
            target: Target mask
        """
        preds=self.hardmax(pred_with_logits,1)
        preds=torch.stack([preds[:,0],1-preds[:,0]],1)
        target=torch.stack([target[:,0],1-target[:,0]],1)
        dice=-Dice().loss(preds,target)
        return dice
      

    def forward(self, moving,target,registration=True):
        """
        Args:
            moving (Tensor): Moving image (BxCxHxW)
            target ([type]): Fixed image (BxCxHxW)
            registration (bool, optional): If False, also return non-integrated inverse flow field. Else return the integrated one. Defaults to False.
        Returns:
            moved (Tensor): Moved image
            field (Tensor): Deformation field from moving to target
        """                    
        moved,field=self.registrator.forward(moving,target,registration=registration)      
        return moved,field
      
    def compute_loss(self,moved=None,target=None,moved_mask=None,target_mask=None,field=None):
        """
        Args:
            moved : Transformed anatomical image
            target : Target anatomical image 
            moved_mask : Transformed mask  
            target_mask : Target mask 
            field : Deformation field
        """        
        loss_ncc=0
        loss_seg=0
        loss_trans=0
        if moved!=None:
            loss_ncc=NCC().loss(moved,target)
        if moved_mask!=None:
            loss_seg= Dice().loss(moved_mask,target_mask)
        if field!=None:
            loss_trans=Grad().loss(field,field) #Recommanded weight for this loss is 1 (see Voxelmorph paper) 
        return loss_ncc+loss_seg+2*loss_trans

    def blend(self,x,y):
        #For visualization
        x=self.norm(x)
        blended=torch.stack([y,x,x])
        return blended

    def training_step(self, batch, batch_nb):
        X,Y=batch # X : Full scan (1x1xLxHxW) | Y : Ground truth (1xCxLxHxW)
        y_opt=self.optimizers()
        loss=[]
        chunk=[None,None]

        #Identifying chunks (i->j)
        for i in range(X.shape[2]):
            y=Y[:,:,i,...]
            if len(torch.unique(torch.argmax(y,1)))>1:
                if chunk[0]==None:
                    chunk[0]=i
                else:
                    chunk[1]=i

        if self.current_epoch==0:
            print(chunk)
        y_opt.zero_grad()
        #Sequences of flow fields (field_up=forward, field_down=backward)
        fields_up=X.shape[2]*[None]
        print(len(fields_up))
        fields_down=X.shape[2]*[None]
        
        if self.mean_dice>0.98:
            self.delta+=1
        dices=[]

        # loss=torch.stack(loss).sum()
        # self.manual_backward(loss,retain_graph=True)
        loss=[]
        for j in range(chunk[0],chunk[1]):
                #Computing flow fields and loss for each hop from chunk[0] to chunk[1]
                x1=X[:,:,j,...]
                x2=X[:,:,j+1,...]
                moved_x1,field_up=self.forward(x1,x2)
                loss.append(self.compute_loss(moved_x1,x2,field=field_up))
                fields_up[j]=(field_up)
                moved_x2,field_down=self.forward(x2,x1)#
                fields_down[j]=(field_down)
                moved_x2=self.registrator.transformer(x2,field_down)
                loss.append(self.compute_loss(moved_x2,x1,field=field_down))

        for i in range(chunk[0],chunk[1]+1):
            up_idx=i+self.delta
            down_idx=i-self.delta
            moving_x=X[:,:,i,...]
            moving_y=Y[:,:,i,...]
            if up_idx>chunk[1] : 
                up_idx=chunk[1]
            if i<chunk[1]:
                composed_field_up=self.compose_list(fields_up[i:up_idx])
                prop_y_up=self.apply_deform(moving_y,composed_field_up,True)
                prop_x_up=self.apply_deform(moving_x,composed_field_up)
                # prop_y_up=self.apply_successive_transformations(moving_y,fields_up[i:up_idx])
                # prop_x_up=self.apply_successive_transformations(moving_x,fields_up[i:up_idx])
                loss.append(self.compute_loss(prop_x_up,X[:,:,up_idx]))
                d_loss_up=[]
                # for lab in range(Y.shape[1]):
                #     if len(torch.unique(Y[:,lab,up_idx]))>1:
                #         d_loss_up.append(self.compute_loss(moved_mask=prop_y_up[:,lab],target_mask=Y[:,lab,up_idx]))
                d_loss_up.append(self.compute_loss(moved_mask=prop_y_up,target_mask=Y[:,:,up_idx]))
                d_loss_up=torch.stack(d_loss_up).sum()*100
                dices.append(self.multi_class_dice(prop_y_up,Y[:,:,up_idx].detach()))
                loss.append(d_loss_up)

            if i>chunk[0]:
                if down_idx<chunk[0] : 
                    down_idx=chunk[0]

                composed_field_down=self.compose_list(fields_down[down_idx:i][::-1])
                prop_y_down=self.apply_deform(moving_y,composed_field_down,True)
                prop_x_down=self.apply_deform(moving_x,composed_field_down)
                # prop_y_down=self.apply_successive_transformations(moving_y,fields_down[down_idx:i][::-1])
                # prop_x_down=self.apply_successive_transformations(moving_x,fields_down[down_idx:i][::-1])
                loss.append(self.compute_loss(prop_x_down,X[:,:,down_idx]))
                d_loss_down=[]
                # for lab in range(Y.shape[1]):
                #     if len(torch.unique(Y[:,lab,down_idx]))>1:
                #         d_loss_down.append(self.compute_loss(moved_mask=prop_y_down[:,lab],target_mask=Y[:,lab,down_idx]))
                d_loss_down.append(self.compute_loss(moved_mask=prop_y_down,target_mask=Y[:,:,down_idx]))
                d_loss_down=torch.stack(d_loss_down).sum()*100
                dices.append(self.multi_class_dice(prop_y_down,Y[:,:,down_idx].detach()))
                loss.append(d_loss_down)

        loss=torch.stack(loss).sum()
        self.manual_backward(loss,retain_graph=True)
        y_opt.step()
        loss=[]



            #Additionnal loss to ensure sequences (images and masks) generated from "positive" and "negative" flows are equal
            # if self.way=='both':
            #     #This helps
            #     if self.losses['bidir-cons-dice']:
            #         loss+=self.compute_loss(moved_mask=prop_y_down,target_mask=prop_y_up)
            #     #This breaks stuff
            #     if self.losses['bidir-cons-reg']:
            #         loss+=self.compute_loss(prop_x_up,prop_x_down)

            # self.logger.experiment.add_image('x_true',X[0,:,chunk[0],...])
            # self.logger.experiment.add_image('prop_x_down',prop_x_down[0,:,0,...])
            # self.logger.experiment.add_image('x_true_f',X[0,:,chunk[1],...])
            # self.logger.experiment.add_image('prop_x_up',prop_x_up[0,:,-1,...])
        self.mean_dice=torch.stack(dices).mean()
        print('Dices : ',self.mean_dice)
        print('Delta :',self.delta)
        self.log('val_accuracy',self.mean_dice*self.delta)
        # if len(dices_prop)>0:
        #     dices_prop=-torch.stack(dices_prop).mean()
        #     self.log('val_accuracy',dices_prop)
        #     print(dices_prop)
        # else:
        #     self.log('val_accuracy',self.current_epoch)
        return loss

    def register_images(self,moving,target,moving_mask):
        moved,field=self.forward(moving,target,registration=True)
        return moved,self.apply_deform(moving_mask,field),field

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,amsgrad=True)

    def hardmax(self,Y,dim):
        return torch.moveaxis(F.one_hot(torch.argmax(Y,dim),self.n_classes), -1, dim)

