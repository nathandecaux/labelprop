# Napari Entry

[Labelprop Index](../README.md#labelprop-index) / [Labelprop](./index.md#labelprop) / Napari Entry

> Auto-generated documentation for [labelprop.napari_entry](https://github.com/nathandecaux/labelprop/blob/main/labelprop/napari_entry.py) module.

#### Attributes

- `package_dir` - Get package directory: pathlib.Path(__file__).parent.absolute()

- `home` - Get user home directory: pathlib.Path.home()


- [Napari Entry](#napari-entry)
  - [get_ckpt_dir](#get_ckpt_dir)
  - [get_fields](#get_fields)
  - [pretrain](#pretrain)
  - [propagate_from_ckpt](#propagate_from_ckpt)
  - [propagate_from_fields](#propagate_from_fields)
  - [resample](#resample)
  - [train_and_infer](#train_and_infer)
  - [train_dataset](#train_dataset)

## get_ckpt_dir

[Show source in napari_entry.py:24](https://github.com/nathandecaux/labelprop/blob/main/labelprop/napari_entry.py#L24)

#### Signature

```python
def get_ckpt_dir(): ...
```



## get_fields

[Show source in napari_entry.py:227](https://github.com/nathandecaux/labelprop/blob/main/labelprop/napari_entry.py#L227)

Get successive fields from the LabelProp model.

#### Arguments

- `img` *Tensor* - The input image.
- `ckpt` *str* - The path to the checkpoint file.
- `mask` *Tensor, optional* - The mask image. Defaults to None.
- `z_axis` *int, optional* - The axis representing the z-dimension. Defaults to 2.
- `selected_slices` *list, optional* - The list of selected slices. Defaults to None.

#### Returns

- `Tensor` - The concatenated fields_up.
- `Tensor` - The concatenated fields_down.
- `Tensor` - The input image (X).
- `Tensor` - The target image (Y).

#### Signature

```python
def get_fields(img, ckpt, mask=None, z_axis=2, selected_slices=None): ...
```



## pretrain

[Show source in napari_entry.py:153](https://github.com/nathandecaux/labelprop/blob/main/labelprop/napari_entry.py#L153)

Pretrains a model using unsupervised learning on a list of images.

#### Arguments

- `img_list` *list* - List of image paths.
- `shape` *int* - Desired shape of the images after resizing.
- `z_axis` *int, optional* - Axis along which to slice the images. Defaults to 2.
- `output_dir` *str, optional* - Directory to save the trained model checkpoint. Defaults to '~/label_prop_checkpoints'.
- `name` *str, optional* - Name of the trained model checkpoint. Defaults to an empty string.
- `max_epochs` *int, optional* - Maximum number of training epochs. Defaults to 100.

#### Signature

```python
def pretrain(
    img_list,
    shape,
    z_axis=2,
    output_dir="~/label_prop_checkpoints",
    name="",
    max_epochs=100,
): ...
```



## propagate_from_ckpt

[Show source in napari_entry.py:42](https://github.com/nathandecaux/labelprop/blob/main/labelprop/napari_entry.py#L42)

Propagates labels from a given checkpoint using the LabelPropDataModule and inference.

#### Arguments

img (ndarray or str): Input image data or path to the image file.
mask (ndarray or str): Input mask data or path to the mask file.
- `checkpoint` *str* - Path to the checkpoint file.
- `shape` *int, optional* - Shape of the input image and mask. Defaults to 304.
- `z_axis` *int, optional* - Axis along which the slices are selected. Defaults to 2.
- `label` *str, optional* - Label to propagate. Defaults to 'all'.
hints (ndarray or str, optional): Input hints data or path to the hints file. Defaults to None.
- `**kwargs` - Additional keyword arguments to be passed to the inference function.

#### Returns

- `tuple` - A tuple containing the propagated labels for the up direction, down direction, and fused direction.

#### Signature

```python
def propagate_from_ckpt(
    img, mask, checkpoint, shape=304, z_axis=2, label="all", hints=None, **kwargs
): ...
```



## propagate_from_fields

[Show source in napari_entry.py:179](https://github.com/nathandecaux/labelprop/blob/main/labelprop/napari_entry.py#L179)

Propagates labels from given fields to the input image.

#### Arguments

img (str or ndarray): Path to the input image or the image array itself.
mask (str or ndarray): Path to the mask image or the mask array itself.
- `fields_up` *ndarray* - Array of fields for upward propagation.
- `fields_down` *ndarray* - Array of fields for downward propagation.
- `shape` *int* - Desired shape of the propagated labels.
- `z_axis` *int, optional* - Axis along which the slices are selected. Defaults to 2.
- `selected_slices` *list, optional* - List of selected slices. Defaults to None.
- `kwargs` *dict, optional* - Additional keyword arguments for propagation.

#### Returns

tuple or tuple of ndarrays: Tuple containing the propagated labels for upward propagation,
    downward propagation, and fused propagation. If 'return_weights' is True in kwargs,
    the tuple also contains the weights used for propagation.

#### Signature

```python
def propagate_from_fields(
    img, mask, fields_up, fields_down, shape, z_axis=2, selected_slices=None, kwargs={}
): ...
```



## resample

[Show source in napari_entry.py:32](https://github.com/nathandecaux/labelprop/blob/main/labelprop/napari_entry.py#L32)

Resample a label tensor to the given size

#### Signature

```python
def resample(Y, size): ...
```



## train_and_infer

[Show source in napari_entry.py:88](https://github.com/nathandecaux/labelprop/blob/main/labelprop/napari_entry.py#L88)

Trains a model and performs inference on the given image and mask.

#### Arguments

img (ndarray or str): The input image data or path to the image file.
mask (ndarray or str): The input mask data or path to the mask file.
- `pretrained_ckpt` *str* - The path to the pretrained checkpoint file.
- `shape` *int* - The desired shape of the input image and mask.
- `max_epochs` *int* - The maximum number of training epochs.
- `z_axis` *int, optional* - The axis along which the slices are selected. Defaults to 2.
- `output_dir` *str, optional* - The directory to save the output checkpoint file. Defaults to '~/label_prop_checkpoints'.
- `name` *str, optional* - The name of the output checkpoint file. Defaults to an empty string.
- `pretraining` *bool, optional* - Whether to perform pretraining. Defaults to False.
hints (ndarray or str, optional): The input hints data or path to the hints file. Defaults to None.
- `**kwargs` - Additional keyword arguments for training and inference.

#### Returns

- `tuple` - A tuple containing the upsampled, downsampled, and fused predictions as numpy arrays.

#### Signature

```python
def train_and_infer(
    img,
    mask,
    pretrained_ckpt,
    shape,
    max_epochs,
    z_axis=2,
    output_dir="~/label_prop_checkpoints",
    name="",
    pretraining=False,
    hints=None,
    **kwargs
): ...
```



## train_dataset

[Show source in napari_entry.py:259](https://github.com/nathandecaux/labelprop/blob/main/labelprop/napari_entry.py#L259)

Trains a dataset using label propagation.

#### Arguments

- `img_list` *list* - List of image file paths.
- `mask_list` *list* - List of mask file paths.
- `pretrained_ckpt` *str* - Path to the pretrained checkpoint.
- `shape` *int* - Shape of the input images.
- `max_epochs` *int* - Maximum number of training epochs.
- `z_axis` *int, optional* - Z-axis index. Defaults to 2.
- `output_dir` *str, optional* - Output directory for saving checkpoints. Defaults to '~/label_prop_checkpoints'.
- `name` *str, optional* - Name of the saved checkpoint. Defaults to ''.
- `**kwargs` - Additional keyword arguments.

#### Returns

- `str` - Path to the best checkpoint.

#### Signature

```python
def train_dataset(
    img_list,
    mask_list,
    pretrained_ckpt,
    shape,
    max_epochs,
    z_axis=2,
    output_dir="~/label_prop_checkpoints",
    name="",
    **kwargs
): ...
```