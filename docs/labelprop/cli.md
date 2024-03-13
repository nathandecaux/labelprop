# Cli

[Labelprop Index](../README.md#labelprop-index) / [Labelprop](./index.md#labelprop) / Cli

> Auto-generated documentation for [labelprop.cli](https://github.com/nathandecaux/labelprop/blob/main/labelprop/cli.py) module.

- [Cli](#cli)
  - [cli](#cli)
  - [launch_server](#launch_server)
  - [pretrain](#pretrain)
  - [propagate](#propagate)
  - [train](#train)
  - [train_dataset](#train_dataset)

## cli

[Show source in cli.py:11](https://github.com/nathandecaux/labelprop/blob/main/labelprop/cli.py#L11)

#### Signature

```python
@click.group()
def cli(): ...
```



## launch_server

[Show source in cli.py:109](https://github.com/nathandecaux/labelprop/blob/main/labelprop/cli.py#L109)

#### Signature

```python
@cli.command()
@click.option("--addr", "-a", default="0.0.0.0")
@click.option("--port", "-p", default=6000)
def launch_server(addr, port): ...
```



## pretrain

[Show source in cli.py:60](https://github.com/nathandecaux/labelprop/blob/main/labelprop/cli.py#L60)

Pretrain the model on a list of images. The images are assumed to be greyscale nifti files. IMG_LIST is a text file containing line-separated paths to the images.

#### Signature

```python
@cli.command()
@click.argument("img_list", type=click.File("r"))
@click.option("--shape", "-s", default=256, help="Image size (default: 256)")
@click.option(
    "--z_axis", "-z", default=2, help="Axis along which to propagate (default: 2)"
)
@click.option(
    "--output_dir",
    "-o",
    type=click.Path(exists=True, file_okay=False),
    default="~/label_prop_checkpoints",
    help="Output directory for checkpoint",
)
@click.option("--name", "-n", default="", help="Checkpoint name (default : datetime")
@click.option("--max_epochs", "-e", default=100)
def pretrain(img_list, shape, z_axis, output_dir, name, max_epochs): ...
```



## propagate

[Show source in cli.py:37](https://github.com/nathandecaux/labelprop/blob/main/labelprop/cli.py#L37)

Propagate labels from sparse segmentation.
IMG_PATH is a greyscale nifti (.nii.gz or .nii) image, while MASKPATH is it related sparse segmentation.
CHECKPOINT is the path to the checkpoint (.ckpt) file.

#### Signature

```python
@cli.command()
@click.argument("img_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("mask_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("checkpoint", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--hints",
    "-h",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to the hints image (.nii.gz)",
)
@click.option("--shape", "-s", default=256, help="Image size (default: 256)")
@click.option(
    "--z_axis", "-z", default=2, help="Axis along which to propagate (default: 2)"
)
@click.option("--label", "-l", default=0, help="Label to propagate (default: 0 = all)")
@click.option(
    "--output_dir",
    "-o",
    type=click.Path(exists=True, file_okay=False),
    default="~/label_prop_checkpoints",
    help="Output directory for predicted masks (up, down and fused)",
)
@click.option("--name", "-n", default="", help="Prefix for the output files (masks)")
def propagate(
    img_path, mask_path, checkpoint, hints, shape, z_axis, label, output_dir, name
): ...
```



## train

[Show source in cli.py:15](https://github.com/nathandecaux/labelprop/blob/main/labelprop/cli.py#L15)

Train a model and save the checkpoint and predicted masks.
IMG_PATH is a greyscale nifti (.nii.gz or .nii) image, while MASKPATH is it related sparse segmentation.

#### Signature

```python
@cli.command()
@click.argument("img_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("mask_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--hints",
    "-h",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to the hints image (.nii.gz)",
)
@click.option("--shape", "-s", default=256, help="Image size (default: 256)")
@click.option(
    "--pretrained_ckpt",
    "-c",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to the pretrained checkpoint (.ckpt)",
)
@click.option("--max_epochs", "-e", default=100)
@click.option(
    "--z_axis", "-z", default=2, help="Axis along which to propagate (default: 2)"
)
@click.option(
    "--output_dir",
    "-o",
    type=click.Path(exists=True, file_okay=False),
    default="~/label_prop_checkpoints",
    help="Output directory for checkpoint and predicted masks",
)
@click.option(
    "--name", "-n", default="", help="Prefix for the output files (checkpoint and masks)"
)
def train(
    img_path,
    mask_path,
    hints,
    pretrained_ckpt,
    shape,
    max_epochs,
    z_axis,
    output_dir,
    name,
): ...
```



## train_dataset

[Show source in cli.py:84](https://github.com/nathandecaux/labelprop/blob/main/labelprop/cli.py#L84)

Train the model on a full dataset. The images are assumed to be greyscale nifti files. Text file containing line-separated paths to greyscale images and comma separated associated mask paths

#### Signature

```python
@cli.command()
@click.argument("img_mask_list", type=click.File("r"))
@click.option(
    "pretrained_ckpt",
    "-c",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the pretrained checkpoint (.ckpt)",
)
@click.option("--shape", "-s", default=256, help="Image size (default: 256)")
@click.option(
    "--z_axis", "-z", default=2, help="Axis along which to propagate (default: 2)"
)
@click.option(
    "--output_dir",
    "-o",
    type=click.Path(exists=True, file_okay=False),
    default="~/label_prop_checkpoints",
    help="Output directory for checkpoint",
)
@click.option("--name", "-n", default="", help="Checkpoint name (default : datetime")
@click.option("--max_epochs", "-e", default=100)
def train_dataset(
    img_mask_list, pretrained_ckpt, shape, z_axis, output_dir, name, max_epochs
): ...
```