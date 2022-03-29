from re import S
import train
from .DataLoading import LabelPropDataModule
from os.path import join

#Init vars
checkpoint=None
if checkpoint!=None:
    checkpoint=f'checkpoints/bench/{checkpoint}'
max_epochs=100
data_dir="/home/nathan/PLEX/norm/sub-000"
selected_slices=[107,153,199]
way='both'
shape=(256,256)
by_composition=True
n_classes=2
losses={'compo-reg-up':True,'compo-reg-down':True,'compo-dice-up':True,'compo-dice-down':True,'bidir-cons-reg':False,'bidir-cons-dice':False}
model_PARAMS={'n_classes':n_classes,'way':way,'shape':shape,'selected_slices':selected_slices,'losses':losses,'by_composition':by_composition}

#Dataloading
dm=LabelPropDataModule(img_path=join(data_dir,"img.nii.gz"),mask_path=join(data_dir,"mask.nii.gz"),lab='all',shape=shape,selected_slices=selected_slices,z_axis=0)

#Training and testing
trained_model,Y_up,Y_down,results=train.train_and_eval(datamodule=dm,model_PARAMS=model_PARAMS,max_epochs=max_epochs,ckpt=checkpoint)
#Dans results, il y a des scores (DSC, ASD, et HD) pour Y_up, Y_down et Y_fused + le checkpoint_path

#Fusing Y_up and Y_down
X,Y_sparse=dm.test_dataset[0]
weights=train.get_weights(Y_sparse)
#Il y a mieux qu'une boucle à faire ici, mais il faudrait bouger les dimensions
for i,w in enumerate(weights):
    Y_up[:,:,i]*=1-w
    Y_down[:,:,i]*=w
Y_fused=Y_up+Y_down
print(results)
#up to you now