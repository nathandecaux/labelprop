from re import S

import numpy
from labelprop.train import train_and_eval, get_successive_fields,inference
from labelprop.DataLoading import LabelPropDataModule
from labelprop.napari_entry import propagate_from_ckpt
from labelprop.lightning_model import LabelProp
from os.path import join
import plotext as plt
import torch
#Init vars
checkpoint='labelprop-epoch=99-val_accuracy=99.00-20042022-100334.ckpt'
if checkpoint!=None:
    checkpoint=f'checkpoints/bench/{checkpoint}'
max_epochs=100
data_dir="/opt/Sync/PLEX/bids/norm/sub-000"
selected_slices=[107,120,153,199]
way='both'
shape=(256,256)
by_composition=True
n_classes=2
losses={'compo-reg-up':False,'compo-reg-down':False,'compo-dice-up':False,'compo-dice-down':False,'bidir-cons-reg':False,'bidir-cons-dice':False}
model_PARAMS={'n_classes':n_classes,'way':way,'shape':shape,'selected_slices':selected_slices,'losses':losses,'by_composition':by_composition}

#Dataloading
dm=LabelPropDataModule(img_path=join(data_dir,"img.nii.gz"),mask_path=join(data_dir,"mask.nii.gz"),lab='all',shape=shape,selected_slices=selected_slices,z_axis=0)

#Training and testing
# trained_model,Y_up,Y_down,results=train_and_eval(datamodule=dm,model_PARAMS=model_PARAMS,max_epochs=max_epochs,ckpt=checkpoint)

#Dans results, il y a des scores (DSC, ASD, et HD) pour Y_up, Y_down et Y_fused + le checkpoint_path

#Fusing Y_up and Y_down
dm.setup()
X,Y_sparse=dm.test_dataset[0]
# trained_model=LabelProp(**model_PARAMS).load_from_checkpoint(checkpoint)
# fields_up,fields_down=get_successive_fields(X.to('cuda'),trained_model.to('cuda'))

# torch.save(fields_up,'fields_up.pt')
# torch.save(fields_down,'fields_down.pt')
# torch.save(X,'img.pt')
# torch.save(Y_sparse,'mask.pt')
import numpy as np
# print(line.shape)
plt.clc()

for i in range(30):
    plt.clt()
    # plt.cld()
    plt.scatter([i],[i])
plt.show()
#Il y a mieux qu'une boucle à faire ici, mais il faudrait bouger les dimensions
# for i,w in enumerate(weights):
#     Y_up[:,:,i]*=1-w
#     Y_down[:,:,i]*=w
# Y_fused=Y_up+Y_down
# print(results)
#up to you now