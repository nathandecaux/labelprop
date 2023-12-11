import sys
import os
import click
import nibabel as ni
from .napari_entry import propagate_from_ckpt,train_and_infer
from .napari_entry import pretrain as pretraining
from .napari_entry import train_dataset as train_dataset_entry
from subprocess import Popen
import pandas as pd

@click.group()
def cli():
    pass    # click.echo('Debug mode is %s' % ('on' if debug else 'off'))

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
    img_list_df=pd.read_csv(img_list,header=None,names=['path'])
    img_list=img_list_df['path'].tolist()

    #Check if files in list exist
    for img_path in img_list:
        img_path=img_path.strip()
        print(img_path.split(),os.path.exists(img_path))
        if not os.path.exists(img_path):
            raise ValueError('File %s does not exist' % img_path)
    pretraining(img_list,shape,z_axis=z_axis,output_dir=output_dir,name=name,max_epochs=max_epochs)


@cli.command()
@click.argument('img_mask_list',type=click.File('r'))#,help='Text file containing line-separated paths to greyscale images (.nii.gz) and comma separated mask paths (.nii.gz)')
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

@cli.command()
@click.option('--addr','-a',default='0.0.0.0')
@click.option('--port','-p',default=6000)
def launch_server(addr,port):
    #Export FLASK_APP=api and FLASK_ENV=development before running
    os.environ['FLASK_APP']='api'
    os.environ['FLASK_ENV']='development'
    os.environ['MKL_SERVICE_FORCE_INTEL']='1'
    #Get package path
    package_path=os.path.dirname(os.path.abspath(__file__))
    Popen('cd %s && flask run --host=%s --port=%d' % (package_path,addr,port),shell=True).wait()


    

if __name__ == '__main__':
    cli()