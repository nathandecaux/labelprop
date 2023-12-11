import torch
from torch import nn
import torch.nn.functional as F
# import pytorch_lightning as pl
import lightning as pl
import kornia
from .voxelmorph2d import VxmDense, NCC, Grad, Dice
from monai.losses import (
    BendingEnergyLoss,
    GlobalMutualInformationLoss,
    DiceLoss,
    LocalNormalizedCrossCorrelationLoss,
)
from kornia.filters import sobel, gaussian_blur2d, canny, spatial_gradient
from kornia.losses import HausdorffERLoss, SSIMLoss, MS_SSIMLoss
from .utils import *
from copy import deepcopy
from datetime import datetime
import os
import json
from crfseg import CRF


class LabelProp(pl.LightningModule):
    @property
    def automatic_optimization(self):
        return False

    def norm(self, x):
        if len(x.shape) == 4:
            x = kornia.enhance.normalize_min_max(x)
        elif len(x.shape) == 3:
            x = kornia.enhance.normalize_min_max(x[:, None, ...])[:, 0, ...]
        else:
            x = kornia.enhance.normalize_min_max(x[None, None, ...])[0, 0, ...]
        return x

    def __init__(
        self,
        n_channels=1,
        n_classes=2,
        learning_rate=5e-4,
        weight_decay=1e-8,
        way="both",
        shape=256,
        selected_slices=None,
        losses={},
        by_composition=False,
        unsupervised=False,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.w_dice = 10
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.selected_slices = selected_slices  # Used in validation step
        if isinstance(shape, int):
            shape = [shape, shape]
        self.registrator = VxmDense(shape, bidir=False, int_downsize=1, int_steps=7)
        self.way = way  # If up, learning only "forward" transitions (phi_i->j with j>i). Other choices : "down", "both". Bet you understood ;)
        self.by_composition = by_composition
        self.unsupervised = unsupervised
        self.CRF=CRF(2)
        # self.loss_model = MTL_loss(['sim','seg','comp','smooth'])
        self.losses = losses
        if self.by_composition:
            print("Using composition for training")
        # Get datetime for saving
        time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.val_by_epoch = time + "_val_by_epoch.json"
        print("Losses", losses)
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
        
    def compose_list(self, flows):
        flows = list(flows)
        compo = flows[-1]
        for flow in reversed(flows[:-1]):
            compo = self.compose_deformation(flow, compo)
        return compo

    def compose_deformation(self, flow_i_k, flow_k_j):
        """Returns flow_k_j(flow_i_k(.)) flow
        Args:
            flow_i_k
            flow_k_j
        Returns:
            [Tensor]: Flow field flow_i_j = flow_k_j(flow_i_k(.))
        """
        flow_i_j = flow_k_j + self.apply_deform(flow_i_k, flow_k_j)
        return flow_i_j

    def forward(self, moving, target, registration=True):
        """
        Args:
            moving (Tensor): Moving image (BxCxHxW)
            target ([type]): Fixed image (BxCxHxW)
            registration (bool, optional): If False, also return non-integrated inverse flow field. Else return the integrated one. Defaults to False.
        Returns:
            moved (Tensor): Moved image
            field (Tensor): Deformation field from moving to target
        """
        return self.registrator.forward(moving, target, registration=registration)

    # def multi_level_training(self,moving,target,level=3):
    #     """
    #     Args:
    #         moving (Tensor): Moving image (BxCxHxW)
    #         target ([type]): Fixed image (BxCxHxW)
    #         registration (bool, optional): If False, also return non-integrated inverse flow field. Else return the integrated one. Defaults to False.
    #     Returns:
    #         moved (Tensor): Moved image
    #         field (Tensor): Deformation field from moving to target
    #     """
    #     stack_moved=[]
    #     stack_field=[]
    #     stack_preint=[]
    #     resampling=torch.nn.Upsample(size=self.shape,mode='bilinear',align_corners=True)
    #     for i in range(level):
    #         downsampling=nn.Upsample(scale_factor=1/(i+1), mode='bilinear',align_corners=True)
    #         downsampled_moving=downsampling(moving)
    #         downsampled_target=downsampling(target)
    #         moved,field,preint_field=self.forward(downsampled_moving,downsampled_target)
    #         self.compute_loss(moved,target,field=field)
    #         stack_moved.append(moved)
    #         stack_field.append(field)
    #         stack_preint.append(preint_field)
    #     return torch.stack(stack_moved,0).mean(0),torch.stack(stack_field,0).mean(0),torch.stack(stack_preint,0).mean(0)

    def compute_loss(
        self, moved=None, target=None, moved_mask=None, target_mask=None, field=None
    ):
        """
        Args:
            moved : Transformed anatomical image
            target : Target anatomical image
            moved_mask : Transformed mask
            target_mask : Target mask
            field : Velocity field (=non integrated)
        """
        losses = {}
        if moved != None:
            # max_peak=F.conv2d(target,target).sum()
            # loss_ncc=-F.conv2d(moved,target).sum()/max_peak#+NCC().loss(moved,target)
            loss_ncc = NCC().loss(moved, target)
            # loss_ncc=GlobalMutualInformationLoss()(moved,target)#MONAI
            # loss_ncc=LocalNormalizedCrossCorrelationLoss(spatial_dims=2, kernel_size=99)(moved,target) #MONAI
            # loss_ncc=nn.MSELoss()(moved,target)
            # loss_ncc=0.05*self.mssim(moved,target)
            losses["sim"] = loss_ncc
        if moved_mask != None:
            loss_seg= Dice().loss(moved_mask,target_mask)

            # loss_seg = (
            #     DiceLoss(include_background=False, softmax=True)(
            #         moved_mask, target_mask
            #     )
            #     - 1
            # )
            losses["seg"] = loss_seg * self.w_dice
            # losses['seg']-=0.005*HausdorffERLoss()(moved_mask[:,1:],target_mask[:,1:].long())
        if field != None:
            # loss_trans=BendingEnergyLoss()(field) #MONAI
            loss_trans = Grad().loss(field, field)
            losses["smooth"] = 1*loss_trans
        # Return dict of losses
        return losses  # {'sim': loss_ncc,'seg':loss_seg,'smooth':loss_trans}

    def compute_contour_loss(self, img, moved_mask):
        # Compute contour loss
        mag, mask_contour = canny(moved_mask[:, 1:2])
        # edges,mag=canny(img)
        return BendingEnergyLoss()(mag)

    def weighting_loss(self, losses):
        """
        Args:
            losses (dict): Dictionary of losses
        Returns:
            loss (Tensor): Weighted loss
        """

    def blend(self, x, y):
        # For visualization
        x = self.norm(x)
        blended = torch.stack([y, x, x])
        return blended

    def training_step(self, batch, batch_nb):
        if self.unsupervised:
            X = batch  # X : Full scan (1x1xLxHxW)
            opt = self.optimizers()
            losses = 0

            for i in range(X.shape[2] - 1):
                # Computing flow fields and loss for each hop from chunk[0] to chunk[1]
                x1 = X[:, :, i, ...]
                x2 = X[:, :, i + 1, ...]
                if not self.way == "down":
                    opt.zero_grad()
                    moved_x1, field_up, preint_field = self.forward(
                        x1, x2, registration=False
                    )
                    cur_loss = self.compute_loss(moved_x1, x2, field=preint_field)
                    loss = cur_loss["sim"] + cur_loss["smooth"]

                    field_down = self.registrator.integrate(-preint_field)
                    moved_x2 = self.registrator.transformer(x2, field_down)
                    loss += self.compute_loss(moved_x2, x1)["sim"]
                    self.manual_backward(loss)
                    opt.step()
                    losses += loss

                if not self.way == "up":
                    opt.zero_grad()
                    moved_x2, field_down, preint_field = self.forward(
                        x2, x1, registration=False
                    )  #
                    cur_loss = self.compute_loss(moved_x2, x1, field=preint_field)
                    loss = cur_loss["sim"] + cur_loss["smooth"]
                    field_up = self.registrator.integrate(-preint_field)
                    moved_x1 = self.registrator.transformer(x1, field_up)
                    loss += self.compute_loss(moved_x1, x2)["sim"]
                    self.manual_backward(loss)
                    opt.step()
                    losses += loss

            self.log_dict({"loss": losses}, prog_bar=True)
            self.log("val_accuracy", -losses)
            return losses
        else:
            hints_multi_lab = None
            with_hints = False
            if len(batch) == 3:
                X, Y, hints_multi_lab = batch
                with_hints = True
            else:
                X, Y = batch  # X : Full scan (1x1xLxHxW) | Y : Ground truth (1xCxLxHxW)
            opt = self.optimizers()
            dices_prop = []
            Y_multi_lab = torch.clone(Y)
            for lab in list(range(Y_multi_lab.shape[1]))[1:]:
                chunks = []
                chunk = []
                # Binarize ground truth according to the label
                Y = torch.stack([1 - Y_multi_lab[:, lab], Y_multi_lab[:, lab]], dim=1)
                if with_hints:
                    hints = torch.stack(
                        [hints_multi_lab[:, -1], hints_multi_lab[:, lab]], dim=1
                    ).float()
                # Identifying chunks (i->j)
                for i in range(X.shape[2]):
                    y = Y[:, :, i, ...]
                    if len(torch.unique(torch.argmax(y, 1))) > 1:
                        chunk.append(i)
                    if len(chunk) == 2:
                        chunks.append(chunk)
                        chunk = [i]
                if self.current_epoch == 0:
                    print(lab, chunks)
                if len(chunks) > 0:
                    for chunk in chunks:
                        opt.zero_grad()
                        # Sequences of flow fields (field_up=forward, field_down=backward)
                        fields_up = []
                        fields_down = []
                        loss_up_sim = []
                        loss_up_smooth = []
                        loss_down_sim = []
                        loss_down_smooth = []
                        loss = 0
                        losses = {
                            "sim": 0,
                            "seg": 0,
                            "comp": 0,
                            "smooth": 0,
                            "hints": [],
                        }

                        for i in range(chunk[0], chunk[1]):
                            # Computing flow fields and loss for each hop from chunk[0] to chunk[1]
                            x1 = X[:, :, i, ...]
                            x2 = X[:, :, i + 1, ...]
                            if not self.way == "down":
                                moved_x1, field_up, preint_field = self.forward(
                                    x1, x2, registration=False
                                )
                                cur_loss = self.compute_loss(
                                    moved_x1, x2, field=preint_field
                                )
                                loss_up_sim.append(cur_loss["sim"])
                                loss_up_smooth.append(cur_loss["smooth"])
                                fields_up.append(field_up)

                                field_down = self.registrator.integrate(-preint_field)
                                moved_x2 = self.registrator.transformer(x2, field_down)
                                loss_up_sim.append(
                                    self.compute_loss(moved_x2, x1)["sim"]
                                )
                                # if len(fields_up)>0:
                                #     field_up_2=self.compose_deformation(fields_up[-1],field_up)
                                #     loss_up.append(self.compute_loss(self.apply_deform(X[:,:,i-1],field_up_2),x2))

                            if not self.way == "up":
                                moved_x2, field_down, preint_field = self.forward(
                                    x2, x1, registration=False
                                )  #
                                fields_down.append(field_down)
                                moved_x2 = self.registrator.transformer(x2, field_down)
                                cur_loss = self.compute_loss(
                                    moved_x2, x1, field=preint_field
                                )
                                loss_down_sim.append(cur_loss["sim"])
                                loss_down_smooth.append(cur_loss["smooth"])
                                field_up = self.registrator.integrate(-preint_field)
                                moved_x1 = self.registrator.transformer(x1, field_up)
                                loss_down_sim.append(
                                    self.compute_loss(moved_x1, x2)["sim"]
                                )

                                # if len(fields_down)>0:
                                #     field_down_2=self.compose_deformation(fields_down[-1],field_down)
                                #     loss_down.append(self.compute_loss(self.apply_deform(X[:,:,i+1],field_down_2),x1))

                        # Better with mean
                        if self.way == "up":
                            losses["sim"] = torch.stack(loss_up_sim).mean()
                            losses["smooth"] = torch.stack(loss_up_smooth).mean()
                        elif self.way == "down":
                            losses["sim"] = torch.stack(loss_down_sim).mean()
                            losses["smooth"] = torch.stack(loss_down_smooth).mean()
                        else:
                            losses["sim"] = (
                                torch.stack(loss_up_sim).mean()
                                + torch.stack(loss_down_sim).mean()
                            )
                            losses["smooth"] = (
                                torch.stack(loss_up_smooth).mean()
                                + torch.stack(loss_down_smooth).mean()
                            )
                            # loss=(loss_up+loss_down)

                        # Computing registration from the sequence of flow fields
                        if not self.way == "down":
                            prop_x_up = X[:, :, chunk[0], ...]
                            prop_y_up = Y[:, :, chunk[0], ...]
                            composed_fields_up = self.compose_list(fields_up)
                            # moved_y1,mask_fields_up=self.forward(prop_y_up[:,1:],Y[:,1:,chunk[1],...],registration=True)
                            # losses['mask_prop']=self.compute_loss(moved_mask=moved_y1,target_mask=Y[:,1:,chunk[1],...])['seg']+nn.L1Loss()(composed_fields_up*prop_y_up,mask_fields_up*prop_y_up)
                            # losses['bending']=BendingEnergyLoss()(composed_fields_up)

                            if self.by_composition:
                                prop_x_up = self.apply_deform(
                                    prop_x_up, composed_fields_up
                                )
                                prop_y_up = self.apply_deform(
                                    prop_y_up, composed_fields_up
                                )
                            else:
                                for i, field_up in enumerate(fields_up):
                                    prop_x_up = self.apply_deform(prop_x_up, field_up)
                                    prop_y_up = self.apply_deform(prop_y_up, field_up, ismask=True)
                                    if with_hints:
                                        if hints[:, 0, chunk[0] + i + 1].sum() > 0:
                                            tp_bkg = (
                                                prop_y_up[:, 0]
                                                * hints[:, 0, chunk[0] + i + 1]
                                            ).sum() / hints[
                                                :, 0, chunk[0] + i + 1
                                            ].sum()
                                            losses["hints"].append(-tp_bkg)
                                        if hints[:, 1, chunk[0] + i + 1].sum() > 0:
                                            tp_obj = (
                                                prop_y_up[:, 1]
                                                * hints[:, 1, chunk[0] + i + 1]
                                            ).sum() / hints[
                                                :, 1, chunk[0] + i + 1
                                            ].sum()
                                            losses["hints"].append(-tp_obj)
                                        # Compute background and foreground true positive
                                        # prop_y_up[:,0][hints[:,0,chunk[0]+i+1]==1]=1
                                        # prop_y_up[:,1][hints[:,1,chunk[0]+i+1]==1]=1

                                    # losses['contours']=self.compute_contour_loss(X[:,:,chunk[0]+i+1],prop_y_up)

                            if self.losses["compo-reg-up"]:
                                losses["comp"] = self.compute_loss(
                                    prop_x_up, X[:, :, chunk[1], ...]
                                )["sim"]
                            if self.losses["compo-dice-up"]:
                                dice_loss = self.compute_loss(
                                    moved_mask=prop_y_up,
                                    target_mask=Y[:, :, chunk[1], ...],
                                )["seg"]
                                losses["seg"] = dice_loss
                                dices_prop.append(dice_loss)

                        if not self.way == "up":
                            prop_x_down = X[:, :, chunk[1], ...]
                            prop_y_down = Y[:, :, chunk[1], ...]
                            composed_fields_down = self.compose_list(fields_down[::-1])
                            # moved_y2,mask_fields_down=self.forward(prop_y_down[:,1:],Y[:,1:,chunk[0],...],registration=True)
                            # losses['mask_prop']+=self.compute_loss(moved_mask=moved_y2,target_mask=Y[:,1:,chunk[0],...])['seg']+nn.L1Loss()(composed_fields_down*prop_y_down,mask_fields_down*prop_y_down)
                            # losses['bending']+=BendingEnergyLoss()(composed_fields_down)
                            if self.by_composition:
                                prop_x_down = self.apply_deform(
                                    prop_x_down, composed_fields_down
                                )
                                prop_y_down = self.apply_deform(
                                    prop_y_down, composed_fields_down
                                )
                            else:
                                i = 1
                                for field_down in reversed(fields_down):
                                    prop_x_down = self.apply_deform(
                                        prop_x_down, field_down
                                    )
                                    prop_y_down = self.apply_deform(
                                        prop_y_down, field_down, ismask=True)
                                    # losses['contours']+=self.compute_contour_loss(X[:,:,chunk[1]-i],prop_y_down)
                                    if with_hints:
                                        if hints[:, 0, chunk[1] - i].sum() > 0:
                                            # tp_bkg=1-prop_y_down[:,0][hints[:,0,chunk[1]-i]==1].mean()
                                            # tp_obj=1-prop_y_down[:,1][hints[:,1,chunk[1]-i]==1].mean()
                                            tp_bkg = (
                                                prop_y_down[:, 0]
                                                * hints[:, 0, chunk[1] - i]
                                            ).sum() / hints[:, 0, chunk[1] - i].sum()
                                            losses["hints"].append(-tp_bkg)
                                        if hints[:, 1, chunk[1] - i].sum() > 0:
                                            tp_obj = (
                                                prop_y_down[:, 1]
                                                * hints[:, 1, chunk[1] - i]
                                            ).sum() / hints[:, 1, chunk[1] - i].sum()
                                            losses["hints"].append(-tp_obj)
                                        # prop_y_down[:,0][hints[:,0,chunk[1]-i]==1]=1
                                        # prop_y_down[:,1][hints[:,1,chunk[1]-i]==1]=1
                                    i += 1

                            if self.losses["compo-reg-down"]:
                                losses["comp"] += self.compute_loss(
                                    prop_x_down, X[:, :, chunk[0], ...]
                                )["sim"]
                            if self.losses["compo-dice-down"]:
                                dice_loss = self.compute_loss(
                                    moved_mask=prop_y_down,
                                    target_mask=Y[:, :, chunk[0], ...],
                                )["seg"]
                                losses["seg"] += dice_loss
                                dices_prop.append(dice_loss)

                        loss = (
                            losses["sim"] + losses["smooth"]
                        )  # +losses['mask_prop']#+losses['bending']#+losses['contours']##torch.stack([v for v in losses.values()]).mean()
                        if (
                            self.losses["compo-dice-up"]
                            or self.losses["compo-dice-down"]
                        ):
                            loss += losses["seg"]
                        if self.losses["compo-reg-up"] or self.losses["compo-reg-down"]:
                            loss += losses["comp"]
                        if "hints" in losses and len(losses["hints"]) > 0:
                            loss += torch.stack(losses["hints"]).mean()
                        # loss=self.loss_model(losses)
                        self.log_dict({"loss": loss}, prog_bar=True)
                        self.manual_backward(loss)
                        opt.step()
                # else:
                #     opt=self.optimizers()
                #     losses=0

                #     for i in range(X.shape[2]-1):
                #         #Computing flow fields and loss for each hop from chunk[0] to chunk[1]
                #         x1=X[:,:,i,...]
                #         x2=X[:,:,i+1,...]
                #         if not self.way=='down':
                #             opt.zero_grad()
                #             moved_x1,field_up,preint_field=self.forward(x1,x2,registration=False)
                #             cur_loss=self.compute_loss(moved_x1,x2,field=preint_field)
                #             loss=cur_loss['sim']+cur_loss['smooth']

                #             field_down=self.registrator.integrate(-preint_field)
                #             moved_x2=self.registrator.transformer(x2,field_down)
                #             loss+=self.compute_loss(moved_x2,x1)['sim']
                #             self.manual_backward(loss)
                #             opt.step()
                #             losses+=loss

                #         if not self.way=='up':
                #             opt.zero_grad()
                #             moved_x2,field_down,preint_field=self.forward(x2,x1,registration=False)#
                #             cur_loss=self.compute_loss(moved_x2,x1,field=preint_field)
                #             loss=cur_loss['sim']+cur_loss['smooth']
                #             field_up=self.registrator.integrate(-preint_field)
                #             moved_x1=self.registrator.transformer(x1,field_up)
                #             loss+=self.compute_loss(moved_x1,x2)['sim']
                #             self.manual_backward(loss)
                #             opt.step()
                #             losses+=loss

                #     self.log_dict({'loss':losses},prog_bar=True)
                #     self.log('val_accuracy',-losses)
                #     return losses
                if with_hints:
                    # Sequences of flow fields (field_up=forward, field_down=backward)
                    fields_down = []
                    loss_down_sim = []
                    loss_down_smooth = []
                    fields_up = []
                    loss_up_sim = []
                    loss_up_smooth = []
                    losses = {"sim": 0, "seg": 0, "comp": 0, "smooth": 0}

                    # Get first and last slice index of hints
                    hints_idx = [
                        idx
                        for idx in range(hints.shape[2])
                        if hints[:, :, idx].sum() > 0
                    ]
                    if len(hints_idx) > 0:
                        low_hint = hints_idx[0]
                        high_hint = hints_idx[-1]
                        if len(chunks) == 0:
                            low_annotation = chunk[0]
                            high_annotation = chunk[0]
                        else:
                            low_annotation = chunks[0][0]
                            high_annotation = chunks[-1][1]
                        loss_hint_up = 0
                        loss_hint_down = 0

                        if low_annotation > low_hint:
                            loss = 0
                            opt.zero_grad()
                            losses = {
                                "sim": 0.0,
                                "comp": 0.0,
                                "smooth": 0.0,
                                "hints": [],
                            }
                            for i in list(range(low_annotation, low_hint - 1, -1))[1:]:
                                x1 = X[:, :, i, ...]
                                x2 = X[:, :, i + 1, ...]
                                moved_x2, field_down, preint_field = self.forward(
                                    x2, x1, registration=False
                                )  #
                                fields_down.append(field_down)
                                moved_x2 = self.registrator.transformer(x2, field_down)
                                cur_loss = self.compute_loss(
                                    moved_x2, x1, field=preint_field
                                )
                                loss_down_sim.append(cur_loss["sim"])
                                loss_down_smooth.append(cur_loss["smooth"])
                                field_up = self.registrator.integrate(-preint_field)
                                moved_x1 = self.registrator.transformer(x1, field_up)
                                loss_down_sim.append(
                                    self.compute_loss(moved_x1, x2)["sim"]
                                )
                            prop_y_down = Y[:, :, low_annotation, ...]
                            prop_x_down = X[:, :, low_annotation, ...]
                            i = low_annotation - 1
                            for field_down in fields_down:
                                prop_y_down = self.apply_deform(prop_y_down, field_down, ismask=True)
                                prop_x_down = self.apply_deform(prop_x_down, field_down)
                                if hints[:, 0, i].sum() > 0:
                                    tp_bkg = (
                                        prop_y_down[:, 0] * hints[:, 0, i]
                                    ).sum() / hints[:, 0, i].sum()
                                    losses["hints"].append(-tp_bkg)
                                if hints[:, 1, i].sum() > 0:
                                    tp_obj = (
                                        prop_y_down[:, 1] * hints[:, 1, i]
                                    ).sum() / hints[:, 1, i].sum()
                                    losses["hints"].append(-tp_obj)
                                i = i - 1
                            losses["hints"] = torch.stack(losses["hints"], 0).mean(0)
                            losses["sim"] += torch.stack(loss_down_sim).mean()
                            losses["smooth"] += torch.stack(loss_down_smooth).mean()
                            losses["comp"] += self.compute_loss(
                                prop_x_down, X[:, :, low_hint, ...]
                            )["sim"]
                            loss_hint_down = (
                                losses["sim"]
                                + losses["smooth"]
                                + losses["comp"]
                                + losses["hints"]
                            )
                            # self.log_dict({'loss':loss_hint_down},prog_bar=True)
                            self.manual_backward(loss_hint_down)
                            opt.step()
                            if "hints" in losses:
                                loss += losses["hints"].detach()
                                dices_prop.append(loss)

                        if high_annotation < high_hint:
                            opt.zero_grad()
                            loss = 0
                            losses = {
                                "sim": 0.0,
                                "comp": 0.0,
                                "smooth": 0.0,
                                "hints": [],
                            }
                            for i in range(high_annotation, high_hint):
                                x1 = X[:, :, i, ...]
                                x2 = X[:, :, i + 1, ...]
                                moved_x1, field_up, preint_field = self.forward(
                                    x1, x2, registration=False
                                )
                                cur_loss = self.compute_loss(
                                    moved_x1, x2, field=preint_field
                                )
                                loss_up_sim.append(cur_loss["sim"])
                                loss_up_smooth.append(cur_loss["smooth"])
                                fields_up.append(field_up)
                                field_down = self.registrator.integrate(-preint_field)
                                moved_x2 = self.registrator.transformer(x2, field_down)
                                loss_up_sim.append(
                                    self.compute_loss(moved_x2, x1)["sim"]
                                )

                            losses["sim"] += torch.stack(loss_up_sim).mean()
                            losses["smooth"] += torch.stack(loss_up_smooth).mean()
                            prop_y_up = Y[:, :, high_annotation, ...]
                            prop_x_up = X[:, :, high_annotation, ...]
                            i = high_annotation
                            for field_up in fields_up:
                                prop_y_up = self.apply_deform(prop_y_up, field_up, ismask=True)
                                if hints[:, 0, i + 1].sum() > 0:
                                    tp_bkg = (
                                        prop_y_up[:, 0] * hints[:, 0, i + 1]
                                    ).sum() / hints[:, 0, i + 1].sum()
                                    losses["hints"].append(-tp_bkg)
                                if hints[:, 1, i + 1].sum() > 0:
                                    tp_obj = (
                                        prop_y_up[:, 1] * hints[:, 1, i + 1]
                                    ).sum() / hints[:, 1, i + 1].sum()
                                    losses["hints"].append(-tp_obj)
                                prop_x_up = self.apply_deform(prop_x_up, field_up)
                                i = i + 1

                            losses["comp"] += self.compute_loss(
                                prop_x_up, X[:, :, high_hint, ...]
                            )["sim"]
                            losses["hints"] = torch.stack(losses["hints"], 0).mean(0)
                            loss_hint_up = (
                                losses["sim"]
                                + losses["smooth"]
                                + losses["comp"]
                                + losses["hints"]
                            )
                            # self.log_dict({'loss':loss_hint_up},prog_bar=True)
                            self.manual_backward(loss_hint_up)
                            opt.step()
                            loss += losses["hints"].detach()
                            dices_prop.append(loss)

                # self.logger.experiment.add_image('x_true',X[0,:,chunk[0],...])
                # self.logger.experiment.add_image('prop_x_down',prop_x_down[0,:,0,...])
                # self.logger.experiment.add_image('x_true_f',X[0,:,chunk[1],...])
                # self.logger.experiment.add_image('prop_x_up',prop_x_up[0,:,-1,...])

            if len(dices_prop) > 0:
                dices_prop = -torch.stack(dices_prop).mean()
                self.log("val_accuracy", dices_prop * 100 / self.w_dice)
                print("Propagated label mean DICE", dices_prop.detach().cpu().numpy())
            else:
                self.log("val_accuracy", self.current_epoch)
            return loss

    def validation_step(self, batch, batch_idx):
        X, Y_dense = batch
        Y = Y_dense.clone()
        # Check if file self.val_by_epoch (json file) exists, otherwise create it
        if not os.path.exists(self.val_by_epoch):
            with open(self.val_by_epoch, "w") as f:
                json.dump({}, f, cls=NumpyEncoder)
        # Load the file
        with open(self.val_by_epoch, "r") as f:
            val_by_epoch = json.load(f)

        for i in range(Y.shape[2]):
            for lab in range(1, Y.shape[1]):
                if i not in self.selected_slices[lab]:
                    Y[:, lab, i, ...] *= 0
        Y_up, Y_down, Y_fused = propagate_by_composition(X[0], Y[0], self)

        metrics = {}
        for lab in range(1, Y.shape[1]):
            metrics[lab] = {
                "dice": {"up": [], "down": [], "fused": []},
                "haus": {"up": [], "down": [], "fused": []},
                "asd": {"up": [], "down": [], "fused": []},
            }
            for i in range(Y.shape[2]):
                if i not in self.selected_slices[lab] and Y_dense[0, lab, i].sum() > 0:
                    for k, v in zip(["up", "down", "fused"], [Y_up, Y_down, Y_fused]):
                        dice, haus, asd = compute_metrics(
                            v[lab, i].to(self.device), Y_dense[0, lab, i]
                        )
                        metrics[lab]["dice"][k].append(dice.cpu().numpy())
                        metrics[lab]["haus"][k].append(haus.cpu().numpy())
                        metrics[lab]["asd"][k].append(asd.cpu().numpy())
            # for k in ['up','down','fused']:
            #     metrics[lab]['mean_dice_'+k]=np.stack(metrics[lab]['dice']).mean()
            #     metrics[lab]['mean_haus_'+k]=np.stack(metrics[lab]['haus']).mean()
            #     metrics[lab]['mean_asd_'+k]=np.stack(metrics[lab]['asd']).mean()

        self.train()
        val_by_epoch[self.current_epoch] = deepcopy(metrics)
        with open(self.val_by_epoch, "w") as f:
            json.dump(val_by_epoch, f, cls=NumpyEncoder)

        return metrics

    def register_images(self, moving, target, moving_mask):
        moved, field = self.forward(moving, target, registration=True)
        return moved, self.apply_deform(moving_mask, field), field

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            amsgrad=True,
        )

    def hardmax(self, Y, dim):
        return torch.moveaxis(F.one_hot(torch.argmax(Y, dim), self.n_classes), -1, dim)


class MTL_loss(torch.nn.Module):
    def __init__(self, losses):
        super().__init__()
        start = 1.0
        self.lw = {}
        self.sigmas = nn.ParameterDict()

        for k in losses:
            self.lw[k] = start
        self.set_dict(self.lw)

    def set_dict(self, dic):
        self.lw = dic
        for k in dic.keys():
            if dic[k] > 0:
                self.sigmas[k] = nn.Parameter(torch.ones(1) / len(dic.keys()))

    def forward(self, loss_dict):
        loss = 0
        with torch.set_grad_enabled(True):
            for k in loss_dict.keys():
                if k in self.lw.keys():
                    print(
                        k,
                        torch.exp(self.sigmas[k])
                        / torch.stack(
                            [torch.exp(v) for v in self.sigmas.values()]
                        ).sum(),
                    )
                    loss += (
                        loss_dict[k]
                        * torch.exp(self.sigmas[k])
                        / torch.stack(
                            [torch.exp(v) for v in self.sigmas.values()]
                        ).sum()
                    )
        return loss