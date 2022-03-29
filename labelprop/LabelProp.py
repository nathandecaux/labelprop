import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import kornia
from .voxelmorph2d import VxmDense,NCC,Grad,Dice

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
    

    def __init__(self,n_channels=1,n_classes=2,learning_rate=5e-3,weight_decay=1e-8,way='up',shape=256,selected_slices=None,losses={},by_composition=False):
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
        print('Losses',losses)
        self.save_hyperparameters()

    def apply_deform(self,x,field):
        """Apply deformation to x from flow field
        Args:
            x (Tensor): Image or mask to deform (BxCxHxW)
            field (Tensor): Deformation field (Bx2xHxW)
        Returns:
            Tensor: Transformed image
        """        
        return self.registrator.transformer(x,field)
    
    def compose_list(self,flows):
        flows=list(flows)
        compo=flows[-1]
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
        return loss_ncc+loss_seg+loss_trans

    def blend(self,x,y):
        #For visualization
        x=self.norm(x)
        blended=torch.stack([y,x,x])
        return blended

    def training_step(self, batch, batch_nb):
        X,Y=batch # X : Full scan (1x1xLxHxW) | Y : Ground truth (1xCxLxHxW)
        y_opt=self.optimizers()
        loss=[]
        chunks=[]
        chunk=[]
        loss_up=[]
        loss_down=[]
        dices_prop=[]
        #Identifying chunks (i->j)
        for i in range(X.shape[2]):
            y=Y[:,:,i,...]
            if len(torch.unique(torch.argmax(y,1)))>1:
                chunk.append(i)
            if len(chunk)==2:
                chunks.append(chunk)
                chunk=[i]
        if self.current_epoch==0:
            print(chunks)
        
        for chunk in chunks:
            y_opt.zero_grad()
            #Sequences of flow fields (field_up=forward, field_down=backward)
            fields_up=[]
            fields_down=[]
            for i in range(chunk[0],chunk[1]):
                #Computing flow fields and loss for each hop from chunk[0] to chunk[1]
                x1=X[:,:,i,...]
                x2=X[:,:,i+1,...]
                if not self.way=='down':
                    moved_x1,field_up=self.forward(x1,x2)
                    loss_up.append(self.compute_loss(moved_x1,x2,field=field_up))
                    fields_up.append(field_up)

                if not self.way=='up':
                    moved_x2,field_down=self.forward(x2,x1)#
                    fields_down.append(field_down)
                    moved_x2=self.registrator.transformer(x2,field_down)
                    loss_down.append(self.compute_loss(moved_x2,x1,field=field_down))
    
            #Better with mean
            if self.way=='up':
                loss=torch.stack(loss_up).mean()
            elif self.way=='down':
                loss=torch.stack(loss_down).mean()
            else:
                loss_up=torch.stack(loss_up).mean()
                loss_down=torch.stack(loss_down).mean()
                loss=(loss_up+loss_down)
            
            # Computing registration from the sequence of flow fields
            if not self.way=='down':
                prop_x_up=X[:,:,chunk[0],...]
                prop_y_up=Y[:,:,chunk[0],...]
                if self.by_composition:
                    composed_fields_up=self.compose_list(fields_up)
                    prop_x_up=self.apply_deform(prop_x_up,composed_fields_up)
                    prop_y_up=self.apply_deform(prop_y_up,composed_fields_up)
                else:
                    for field_up in fields_up:
                        prop_x_up=self.apply_deform(prop_x_up,field_up)
                        prop_y_up=self.apply_deform(prop_y_up,field_up)
                
                if self.losses['compo-reg-up']:
                    loss+=self.compute_loss(prop_x_up,X[:,:,chunk[1],...])
                if self.losses['compo-dice-up']:
                    dice_loss=self.compute_loss(moved_mask=prop_y_up,target_mask=Y[:,:,chunk[1],...])
                    loss+=dice_loss
                    dices_prop.append(dice_loss)

            if not self.way=='up':
                prop_x_down=X[:,:,chunk[1],...]
                prop_y_down=Y[:,:,chunk[1],...]
                if self.by_composition:
                    composed_fields_down=self.compose_list(fields_down[::-1])
                    prop_x_down=self.apply_deform(prop_x_down,composed_fields_down)
                    prop_y_down=self.apply_deform(prop_y_down,composed_fields_down)
                else:
                    for field_down in reversed(fields_down):
                        prop_x_down=self.apply_deform(prop_x_down,field_down)
                        prop_y_down=self.apply_deform(prop_y_down,field_down)

                if self.losses['compo-reg-down']:
                    loss+=self.compute_loss(prop_x_down,X[:,:,chunk[0],...])
                if self.losses['compo-dice-down']:
                    dice_loss=self.compute_loss(moved_mask=prop_y_down,target_mask=Y[:,:,chunk[0],...])
                    loss+=dice_loss
                    dices_prop.append(dice_loss)


            #Additionnal loss to ensure sequences (images and masks) generated from "positive" and "negative" flows are equal
            # if self.way=='both':
            #     #This helps
            #     if self.losses['bidir-cons-dice']:
            #         loss+=self.compute_loss(moved_mask=prop_y_down,target_mask=prop_y_up)
            #     #This breaks stuff
            #     if self.losses['bidir-cons-reg']:
            #         loss+=self.compute_loss(prop_x_up,prop_x_down)

            self.log_dict({'loss':loss},prog_bar=True)
            self.manual_backward(loss)
            y_opt.step()
            loss_up=[]
            loss_down=[]
            # self.logger.experiment.add_image('x_true',X[0,:,chunk[0],...])
            # self.logger.experiment.add_image('prop_x_down',prop_x_down[0,:,0,...])
            # self.logger.experiment.add_image('x_true_f',X[0,:,chunk[1],...])
            # self.logger.experiment.add_image('prop_x_up',prop_x_up[0,:,-1,...])
        
        if len(dices_prop)>0:
            dices_prop=-torch.stack(dices_prop).mean()
            self.log('val_accuracy',dices_prop)
            print(dices_prop)
        else:
            self.log('val_accuracy',self.current_epoch)
        return loss

    def register_images(self,moving,target,moving_mask):
        moved,field=self.forward(moving,target,registration=True)
        return moved,self.apply_deform(moving_mask,field),field

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,amsgrad=True)

    def hardmax(self,Y,dim):
        return torch.moveaxis(F.one_hot(torch.argmax(Y,dim),self.n_classes), -1, dim)

