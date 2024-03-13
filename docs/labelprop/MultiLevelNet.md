# Multilevelnet

[Labelprop Index](../README.md#labelprop-index) / [Labelprop](./index.md#labelprop) / Multilevelnet

> Auto-generated documentation for [labelprop.MultiLevelNet](https://github.com/nathandecaux/labelprop/blob/main/labelprop/MultiLevelNet.py) module.

- [Multilevelnet](#multilevelnet)
  - [MultiLevelNet](#multilevelnet)
    - [MultiLevelNet().forward](#multilevelnet()forward)
    - [MultiLevelNet().get_conv_blocks](#multilevelnet()get_conv_blocks)
    - [MultiLevelNet().get_downsample_blocks](#multilevelnet()get_downsample_blocks)

## MultiLevelNet

[Show source in MultiLevelNet.py:5](https://github.com/nathandecaux/labelprop/blob/main/labelprop/MultiLevelNet.py#L5)

Convolutional network generating deformation field with different scales.

#### Signature

```python
class MultiLevelNet(nn.Module):
    def __init__(self, inshape, in_channels=2, levels=3, features=16): ...
```

### MultiLevelNet().forward

[Show source in MultiLevelNet.py:41](https://github.com/nathandecaux/labelprop/blob/main/labelprop/MultiLevelNet.py#L41)

For each levels, downsample the input and apply the convolutional block.

#### Signature

```python
def forward(self, x, registration=False): ...
```

### MultiLevelNet().get_conv_blocks

[Show source in MultiLevelNet.py:26](https://github.com/nathandecaux/labelprop/blob/main/labelprop/MultiLevelNet.py#L26)

For each levels, create a convolutional block with two Conv Tanh BatchNorm layers

#### Signature

```python
def get_conv_blocks(self, in_channels, levels, intermediate_features): ...
```

### MultiLevelNet().get_downsample_blocks

[Show source in MultiLevelNet.py:20](https://github.com/nathandecaux/labelprop/blob/main/labelprop/MultiLevelNet.py#L20)

#### Signature

```python
def get_downsample_blocks(self, in_channels, levels): ...
```