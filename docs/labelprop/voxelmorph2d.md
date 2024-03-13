# Voxelmorph2d

[Labelprop Index](../README.md#labelprop-index) / [Labelprop](./index.md#labelprop) / Voxelmorph2d

> Auto-generated documentation for [labelprop.voxelmorph2d](https://github.com/nathandecaux/labelprop/blob/main/labelprop/voxelmorph2d.py) module.

- [Voxelmorph2d](#voxelmorph2d)
  - [AffineGenerator](#affinegenerator)
    - [AffineGenerator().forward](#affinegenerator()forward)
  - [AffineGenerator3D](#affinegenerator3d)
    - [AffineGenerator3D().forward](#affinegenerator3d()forward)
  - [Dice](#dice)
    - [Dice().loss](#dice()loss)
  - [FeaturesToAffine](#featurestoaffine)
    - [FeaturesToAffine().forward](#featurestoaffine()forward)
  - [Grad](#grad)
    - [Grad().loss](#grad()loss)
  - [MSE](#mse)
    - [MSE().loss](#mse()loss)
  - [MultiLevelNet](#multilevelnet)
    - [MultiLevelNet().compose_deformation](#multilevelnet()compose_deformation)
    - [MultiLevelNet().compose_list](#multilevelnet()compose_list)
    - [MultiLevelNet().forward](#multilevelnet()forward)
    - [MultiLevelNet().get_conv_blocks](#multilevelnet()get_conv_blocks)
    - [MultiLevelNet().get_downsample_blocks](#multilevelnet()get_downsample_blocks)
    - [MultiLevelNet().get_transformer_list](#multilevelnet()get_transformer_list)
  - [NCC](#ncc)
    - [NCC().loss](#ncc()loss)
  - [ResizeTransform](#resizetransform)
    - [ResizeTransform().forward](#resizetransform()forward)
  - [SingleLevelNet](#singlelevelnet)
    - [SingleLevelNet().forward](#singlelevelnet()forward)
    - [SingleLevelNet().get_conv_blocks](#singlelevelnet()get_conv_blocks)
  - [SpatialTransformer](#spatialtransformer)
    - [SpatialTransformer().forward](#spatialtransformer()forward)
  - [VecInt](#vecint)
    - [VecInt().forward](#vecint()forward)
  - [VxmDense](#vxmdense)
    - [VxmDense().forward](#vxmdense()forward)

## AffineGenerator

[Show source in voxelmorph2d.py:178](https://github.com/nathandecaux/labelprop/blob/main/labelprop/voxelmorph2d.py#L178)

Dense network that takes affine matrix and generate affine transformation

#### Signature

```python
class AffineGenerator(nn.Module):
    def __init__(self, inshape): ...
```

### AffineGenerator().forward

[Show source in voxelmorph2d.py:189](https://github.com/nathandecaux/labelprop/blob/main/labelprop/voxelmorph2d.py#L189)

#### Signature

```python
def forward(self, x1, x2): ...
```



## AffineGenerator3D

[Show source in voxelmorph2d.py:197](https://github.com/nathandecaux/labelprop/blob/main/labelprop/voxelmorph2d.py#L197)

Dense network that takes affine matrix and generate affine transformation

#### Signature

```python
class AffineGenerator3D(nn.Module):
    def __init__(self, inshape): ...
```

### AffineGenerator3D().forward

[Show source in voxelmorph2d.py:208](https://github.com/nathandecaux/labelprop/blob/main/labelprop/voxelmorph2d.py#L208)

#### Signature

```python
def forward(self, x1, x2): ...
```



## Dice

[Show source in voxelmorph2d.py:518](https://github.com/nathandecaux/labelprop/blob/main/labelprop/voxelmorph2d.py#L518)

N-D dice for segmentation

#### Signature

```python
class Dice: ...
```

### Dice().loss

[Show source in voxelmorph2d.py:523](https://github.com/nathandecaux/labelprop/blob/main/labelprop/voxelmorph2d.py#L523)

#### Signature

```python
def loss(self, y_true, y_pred): ...
```



## FeaturesToAffine

[Show source in voxelmorph2d.py:158](https://github.com/nathandecaux/labelprop/blob/main/labelprop/voxelmorph2d.py#L158)

Dense network that takes pixels of features map and convert it to affine matrix

#### Signature

```python
class FeaturesToAffine(nn.Module):
    def __init__(self, inshape): ...
```

### FeaturesToAffine().forward

[Show source in voxelmorph2d.py:172](https://github.com/nathandecaux/labelprop/blob/main/labelprop/voxelmorph2d.py#L172)

#### Signature

```python
def forward(self, x): ...
```



## Grad

[Show source in voxelmorph2d.py:531](https://github.com/nathandecaux/labelprop/blob/main/labelprop/voxelmorph2d.py#L531)

N-D gradient loss.

#### Signature

```python
class Grad:
    def __init__(self, penalty="l1", loss_mult=None): ...
```

### Grad().loss

[Show source in voxelmorph2d.py:540](https://github.com/nathandecaux/labelprop/blob/main/labelprop/voxelmorph2d.py#L540)

#### Signature

```python
def loss(self, _, y_pred): ...
```



## MSE

[Show source in voxelmorph2d.py:509](https://github.com/nathandecaux/labelprop/blob/main/labelprop/voxelmorph2d.py#L509)

Mean squared error loss.

#### Signature

```python
class MSE: ...
```

### MSE().loss

[Show source in voxelmorph2d.py:514](https://github.com/nathandecaux/labelprop/blob/main/labelprop/voxelmorph2d.py#L514)

#### Signature

```python
def loss(self, y_true, y_pred): ...
```



## MultiLevelNet

[Show source in voxelmorph2d.py:257](https://github.com/nathandecaux/labelprop/blob/main/labelprop/voxelmorph2d.py#L257)

Convolutional network generating deformation field with different scales.

#### Signature

```python
class MultiLevelNet(nn.Module):
    def __init__(self, inshape, in_channels=2, levels=3, features=16): ...
```

### MultiLevelNet().compose_deformation

[Show source in voxelmorph2d.py:309](https://github.com/nathandecaux/labelprop/blob/main/labelprop/voxelmorph2d.py#L309)

Returns flow_k_j(flow_i_k(.)) flow

#### Arguments

flow_i_k
flow_k_j

#### Returns

- `[Tensor]` - Flow field flow_i_j = flow_k_j(flow_i_k(.))

#### Signature

```python
def compose_deformation(self, flow_i_k, flow_k_j): ...
```

### MultiLevelNet().compose_list

[Show source in voxelmorph2d.py:303](https://github.com/nathandecaux/labelprop/blob/main/labelprop/voxelmorph2d.py#L303)

#### Signature

```python
def compose_list(self, flows): ...
```

### MultiLevelNet().forward

[Show source in voxelmorph2d.py:320](https://github.com/nathandecaux/labelprop/blob/main/labelprop/voxelmorph2d.py#L320)

For each levels, downsample the input and apply the convolutional block.

#### Signature

```python
def forward(self, x, registration=False): ...
```

### MultiLevelNet().get_conv_blocks

[Show source in voxelmorph2d.py:279](https://github.com/nathandecaux/labelprop/blob/main/labelprop/voxelmorph2d.py#L279)

For each levels, create a convolutional block with two Conv Tanh BatchNorm layers

#### Signature

```python
def get_conv_blocks(self, in_channels, levels, intermediate_features): ...
```

### MultiLevelNet().get_downsample_blocks

[Show source in voxelmorph2d.py:273](https://github.com/nathandecaux/labelprop/blob/main/labelprop/voxelmorph2d.py#L273)

#### Signature

```python
def get_downsample_blocks(self, in_channels, levels): ...
```

### MultiLevelNet().get_transformer_list

[Show source in voxelmorph2d.py:294](https://github.com/nathandecaux/labelprop/blob/main/labelprop/voxelmorph2d.py#L294)

Create a list of spatial transformer for each level.

#### Signature

```python
def get_transformer_list(self, levels, inshape): ...
```



## NCC

[Show source in voxelmorph2d.py:444](https://github.com/nathandecaux/labelprop/blob/main/labelprop/voxelmorph2d.py#L444)

Local (over window) normalized cross correlation loss.

#### Signature

```python
class NCC:
    def __init__(self, win=None): ...
```

### NCC().loss

[Show source in voxelmorph2d.py:452](https://github.com/nathandecaux/labelprop/blob/main/labelprop/voxelmorph2d.py#L452)

#### Signature

```python
def loss(self, y_true, y_pred, mean=True): ...
```



## ResizeTransform

[Show source in voxelmorph2d.py:410](https://github.com/nathandecaux/labelprop/blob/main/labelprop/voxelmorph2d.py#L410)

Resize a transform, which involves resizing the vector field *and* rescaling it.

#### Signature

```python
class ResizeTransform(nn.Module):
    def __init__(self, vel_resize, ndims): ...
```

### ResizeTransform().forward

[Show source in voxelmorph2d.py:424](https://github.com/nathandecaux/labelprop/blob/main/labelprop/voxelmorph2d.py#L424)

#### Signature

```python
def forward(self, x): ...
```



## SingleLevelNet

[Show source in voxelmorph2d.py:215](https://github.com/nathandecaux/labelprop/blob/main/labelprop/voxelmorph2d.py#L215)

Convolutional network generating deformation field

#### Signature

```python
class SingleLevelNet(nn.Module):
    def __init__(self, inshape, in_channels=2, features=16): ...
```

### SingleLevelNet().forward

[Show source in voxelmorph2d.py:245](https://github.com/nathandecaux/labelprop/blob/main/labelprop/voxelmorph2d.py#L245)

Forward pass of the network

#### Arguments

- `x` *[Tensor]* - Tensor of shape (B,C,H,W)

#### Returns

- `[Tensor]` - Tensor of shape (B,C,H,W)

#### Signature

```python
def forward(self, x): ...
```

### SingleLevelNet().get_conv_blocks

[Show source in voxelmorph2d.py:227](https://github.com/nathandecaux/labelprop/blob/main/labelprop/voxelmorph2d.py#L227)

For each levels, create a convolutional block with two Conv Tanh BatchNorm layers

#### Signature

```python
def get_conv_blocks(self, in_channels, intermediate_features): ...
```



## SpatialTransformer

[Show source in voxelmorph2d.py:343](https://github.com/nathandecaux/labelprop/blob/main/labelprop/voxelmorph2d.py#L343)

N-D Spatial Transformer

#### Signature

```python
class SpatialTransformer(nn.Module):
    def __init__(self, size, mode="bilinear", levels=4): ...
```

### SpatialTransformer().forward

[Show source in voxelmorph2d.py:368](https://github.com/nathandecaux/labelprop/blob/main/labelprop/voxelmorph2d.py#L368)

#### Signature

```python
def forward(self, src, flow): ...
```



## VecInt

[Show source in voxelmorph2d.py:390](https://github.com/nathandecaux/labelprop/blob/main/labelprop/voxelmorph2d.py#L390)

Integrates a vector field via scaling and squaring.

#### Signature

```python
class VecInt(nn.Module):
    def __init__(self, inshape, nsteps): ...
```

### VecInt().forward

[Show source in voxelmorph2d.py:403](https://github.com/nathandecaux/labelprop/blob/main/labelprop/voxelmorph2d.py#L403)

#### Signature

```python
def forward(self, vec): ...
```



## VxmDense

[Show source in voxelmorph2d.py:10](https://github.com/nathandecaux/labelprop/blob/main/labelprop/voxelmorph2d.py#L10)

VoxelMorph network for (unsupervised) nonlinear registration between two images.

#### Signature

```python
class VxmDense(nn.Module):
    def __init__(
        self,
        inshape,
        int_steps=7,
        int_downsize=2,
        bidir=False,
        use_probs=False,
        src_feats=1,
        trg_feats=1,
        unet_half_res=False,
        sub_levels=3,
    ): ...
```

### VxmDense().forward

[Show source in voxelmorph2d.py:111](https://github.com/nathandecaux/labelprop/blob/main/labelprop/voxelmorph2d.py#L111)

#### Arguments

- `source` - Source image tensor.
- `target` - Target image tensor.
- `registration` - Return transformed image and flow. Default is False.

#### Signature

```python
def forward(self, source, target, registration=False): ...
```