import sys
import os
import click
import nibabel as ni
from labelprop.napari_entry import propagate_from_ckpt,train_and_infer
from labelprop.napari_entry import pretrain as pretraining
from labelprop.napari_entry import train_dataset as train_dataset_entry


# @click.option('--debug/--no-debug', default=False)
@click.group()
def cli():
    pass    # click.echo('Debug mode is %s' % ('on' if debug else 'off'))



# # def train(img,mask,pretrained_ckpt,shape,max_epochs,z_axis=2,output_dir='~/label_prop_checkpoints',name='',pretraining=False):
# #     Y_up,Y_down,Y_fused=train_and_infer(img,mask,pretrained_ckpt,shape,max_epochs,z_axis=z_axis,output_dir=output_dir,name=name,pretraining=pretraining)

@cli.command()
@click.argument('img_path',type=click.Path(exists=True,dir_okay=False))#,help='Path to the greyscale image (.nii.gz)')
@click.argument('mask_path',type=click.Path(exists=True,dir_okay=False))#,help='Path to the mask image (.nii.gz)')
@click.option('--shape','-s', default=256, help='Image size (default: 256)')
@click.option('--pretrained_ckpt','-c',type=click.Path(exists=True,dir_okay=False), default=None, help='Path to the pretrained checkpoint (.ckpt)')
@click.option('--max_epochs','-e', default=100)
@click.option('--z_axis','-z', default=2, help='Axis along which to propagate (default: 2)')
@click.option('--output_dir','-o', type=click.Path(exists=True,file_okay=False),default='~/label_prop_checkpoints',help='Output directory for checkpoint and predicted masks')
@click.option('--name','-n', default='',help='Prefix for the output files (checkpoint and masks)')
def train(img_path,mask_path,pretrained_ckpt,shape,max_epochs,z_axis,output_dir,name):
    """
    Train a model and save the checkpoint and predicted masks.
    IMG_PATH is a greyscale nifti (.nii.gz or .nii) image, while MASKPATH is it related sparse segmentation.
    """
    affine=ni.load(img_path).affine
    Y_up,Y_down,Y_fused=train_and_infer(img_path,mask_path,pretrained_ckpt,shape,max_epochs,z_axis=z_axis,output_dir=output_dir,name=name,pretraining=False)
    ni.save(ni.Nifti1Image(Y_up.astype('uint8'),affine),os.path.join(output_dir,name+'_up.nii.gz'))
    ni.save(ni.Nifti1Image(Y_down.astype('uint8'),affine),os.path.join(output_dir,name+'_down.nii.gz'))
    ni.save(ni.Nifti1Image(Y_fused.astype('uint8'),affine),os.path.join(output_dir,name+'_fused.nii.gz'))

# def propagate(img,mask,checkpoint,shape=304,z_axis=2,label='all',**kwargs):
#     Y_up,Y_down,Y_fused=propagate_from_ckpt(img,mask,checkpoint,shape,z_axis,label,**kwargs)

@cli.command()
@click.argument('img_path',type=click.Path(exists=True,dir_okay=False))#,help='Path to the greyscale image (.nii.gz)')
@click.argument('mask_path',type=click.Path(exists=True,dir_okay=False))#,help='Path to the mask image (.nii.gz)')
@click.argument('checkpoint',type=click.Path(exists=True,dir_okay=False))#,help='Path to the checkpoint (.ckpt)')
@click.option('--shape','-s', default=256, help='Image size (default: 256)')
@click.option('--z_axis','-z', default=2, help='Axis along which to propagate (default: 2)')
@click.option('--label','-l', default=0, help='Label to propagate (default: 0 = all)')
@click.option('--output_dir','-o', type=click.Path(exists=True,file_okay=False),default='~/label_prop_checkpoints',help='Output directory for predicted masks (up, down and fused)')
@click.option('--name','-n', default='',help='Prefix for the output files (masks)')
def propagate(img_path,mask_path,checkpoint,shape,z_axis,label,output_dir,name):
    """
        Propagate labels from sparse segmentation. 
        IMG_PATH is a greyscale nifti (.nii.gz or .nii) image, while MASKPATH is it related sparse segmentation.
        CHECKPOINT is the path to the checkpoint (.ckpt) file.
    """
    affine=ni.load(img_path).affine
    Y_up,Y_down,Y_fused=propagate_from_ckpt(img_path,mask_path,checkpoint,shape,z_axis,label)
    ni.save(ni.Nifti1Image(Y_up.astype('uint8'),affine),os.path.join(output_dir,name+'_up.nii.gz'))
    ni.save(ni.Nifti1Image(Y_down.astype('uint8'),affine),os.path.join(output_dir,name+'_down.nii.gz'))
    ni.save(ni.Nifti1Image(Y_fused.astype('uint8'),affine),os.path.join(output_dir,name+'_fused.nii.gz'))

# def pretrain(img_list,shape,z_axis=2,output_dir='~/label_prop_checkpoints',name='',max_epochs=100):
#     pretrain(img_list,shape,z_axis=z_axis,output_dir=output_dir,name=name,max_epochs=max_epochs)

@cli.command()
@click.argument('img_list',type=click.File('r'))#,help='Text file containing line-separated paths to greyscale images (.nii.gz)')
@click.option('--shape','-s', default=256, help='Image size (default: 256)')
@click.option('--z_axis','-z', default=2, help='Axis along which to propagate (default: 2)')
@click.option('--output_dir','-o', type=click.Path(exists=True,file_okay=False),default='~/label_prop_checkpoints',help='Output directory for checkpoint')
@click.option('--name','-n', default='',help='Checkpoint name (default : datetime')
@click.option('--max_epochs','-e', default=100)
def pretrain(img_list,shape,z_axis,output_dir,name,max_epochs):
    """
    Pretrain the model on a list of images. The images are assumed to be greyscale nifti files. IMG_LIST is a text file containing line-separated paths to the images.
    """
    #Convert csv to list of paths
    img_list=img_list.read().splitlines()
    #Check if files in list exist
    for img_path in img_list:
        if not os.path.exists(img_path):
            raise ValueError('File %s does not exist' % img_path)
    pretraining(img_list,shape,z_axis=z_axis,output_dir=output_dir,name=name,max_epochs=max_epochs)


#train_dataset(img_list,mask_list,pretrained_ckpt,shape,max_epochs,z_axis=2,output_dir='~/label_prop_checkpoints',name='',**kwargs):
@cli.command()
@click.argument('img_mask_list',type=click.File('r')if hints!='':
            arrays['hints']=hints.data.astype('uint8')if hints!='':
            arrays['hints']=hints.data.astype('uint8'))#,help='Text file containing line-separated paths to greyscale images (.nii.gz) and comma separated mask paths (.nii.gz)')
@click.option('pretrained_ckpt','-c',type=click.Path(exists=True,dir_okay=False))#,help='Path to the checkpoint (.ckpt)')
@click.option('--shape','-s', default=256, help='Image size (default: 256)')
@click.option('--z_axis','-z', default=2, help='Axis along which to propagate (default: 2)')
@click.option('--output_dir','-o', type=click.Path(exists=True,file_okay=False),default='~/label_prop_checkpoints',help='Output directory for checkpoint')
@click.option('--name','-n', default='',help='Checkpoint name (default : datetime')
@click.option('--max_epochs','-e', default=100)
def train_dataset(img_mask_list,pretrained_ckpt,shape,z_axis,output_dir,name,max_epochs):
    """
    Train the model on a full dataset. The images are assumed to be greyscale nifti files. Text file containing line-separated paths to greyscale images and comma separated associated mask paths
    """
    #Convert csv to list of paths
    img_mask_list=img_mask_list.read().splitlines()
    img_list=[x.split(',')[0] for x in img_mask_list]
    mask_list=[x.split(',')[1] for x in img_mask_list]
    #Check if files in list exist
    for img_path in img_list:
        if not os.path.exists(img_path):
            raise ValueError('File %s does not exist' % img_path)
    for mask_path in mask_list:
        if not os.path.exists(mask_path):
            raise ValueError('File %s does not exist' % mask_path)
    train_dataset_entry(img_list,mask_list,pretrained_ckpt,shape,max_epochs,z_axis=z_axis,output_dir=output_dir,name=name)
if __name__ == '__main__':
    cli()