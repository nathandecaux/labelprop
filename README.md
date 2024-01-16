# LabelProp - CLI and Server


[![License](https://img.shields.io/pypi/l/deep-labelprop.svg?color=green)](https://github.com/nathandecaux/labelprop/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/deep-labelprop.svg?color=green)](https://pypi.org/project/deep-labelprop)
[![Python Version](https://img.shields.io/pypi/pyversions/deep-labelprop.svg?color=green)](https://python.org)


3D semi-automatic segmentation using deep registration-based 2D label propagation
---------------------------------------------------------------------------------
---

Check [napari-labelprop](https://github.com/nathandecaux/napari-labelprop) plugin for use in the napari viewer. 
See also the [napari-labelprop-remote](https://github.com/nathandecaux/napari-labelprop-remote) plugin for remote computing.

---------------------------------------------------------------------------------
---
## About

See "Semi-automatic muscle segmentation in MR images using deep registration-based label propagation" paper : 

[[Paper]![Paper](https://www.integrad.nl/assets/uploads/2016/02/cta-elsevier_logo-no_bg.png)](https://www.sciencedirect.com/science/article/pii/S0031320323002297?casa_token=r5FPBVXYXX4AAAAA:mStyUXb0i4lGqBmfF1j5fV1T9FuCMrpYfwh3lwQve2XAnzUBPZviAiFgMtH7lv6hdcWsA7yM) [[PDF]![PDF](https://www.ouvrirlascience.fr/wp-content/uploads/2018/12/HAL-3.png)](https://hal.science/hal-03945559/document)
<p>
  <img src="https://github.com/nathandecaux/labelprop.github.io/raw/main/demo_cut.gif" width="600">
</p>

## Installation

Using pip

    pip install deep-labelprop

or to get the development version :

    git clone https://github.com/nathandecaux/labelprop
    cd labelprop
    pip install -e .

## Usage

### CLI

Basic operations can be done using the command-line interface provided in labelprop.py at the root of the project.

#### Pretraining

    $ labelprop pretrain --help
    Usage: labelprop.py pretrain [OPTIONS] IMG_LIST

    Pretrain the model on a list of images. The images are assumed to be
      greyscale nifti files. IMG_LIST is a text file containing line-separated
      paths to the images.

    Options:
      -s, --shape INTEGER         Image size (default: 256)
      -z, --z_axis INTEGER        Axis along which to propagate (default: 2)
      -o, --output_dir DIRECTORY  Output directory for checkpoint
      -n, --name TEXT             Checkpoint name (default : datetime)
      -e, --max_epochs INTEGER    

#### Training

    $ labelprop train --help
    Usage: labelprop.py train [OPTIONS] IMG_PATH MASK_PATH

    Train a model and save the checkpoint and predicted masks. IMG_PATH is a
      greyscale nifti (.nii.gz or .nii) image, while MASKPATH is it related sparse
      segmentation.

    Options:
      -s, --shape INTEGER         Image size (default: 256)
      -c, --pretrained_ckpt FILE  Path to the pretrained checkpoint (.ckpt)
      -e, --max_epochs INTEGER
      -z, --z_axis INTEGER        Axis along which to propagate (default: 2)
      -o, --output_dir DIRECTORY  Output directory for checkpoint and predicted
                                  masks
      -n, --name TEXT             Prefix for the output files (checkpoint and
                                  masks)

#### Propagating (inference)

    $ labelprop propagate --help
    Usage: labelprop.py propagate [OPTIONS] IMG_PATH MASK_PATH CHECKPOINT

    Propagate labels from sparse segmentation.  IMG_PATH is a greyscale nifti
      (.nii.gz or .nii) image, while MASKPATH is it related sparse segmentation.
      CHECKPOINT is the path to the checkpoint (.ckpt) file.

    Options:
      -s, --shape INTEGER         Image size (default: 256)
      -z, --z_axis INTEGER        Axis along which to propagate (default: 2)
      -l, --label INTEGER         Label to propagate (default: 0 = all)
      -o, --output_dir DIRECTORY  Output directory for predicted masks (up, down
                                  and fused)
      -n, --name TEXT             Prefix for the output files (masks)

### GUI

See this [repo](https://github.com/nathandecaux/napari-labelprop-remote)

<p align="center">
  <img src="https://github.com/nathandecaux/labelprop.github.io/raw/main/client_server.drawio.svg" width="600">
</p>

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
