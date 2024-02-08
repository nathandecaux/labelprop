# Dataloading

[Labelprop Index](../README.md#labelprop-index) / [Labelprop](./index.md#labelprop) / Dataloading

> Auto-generated documentation for [labelprop.DataLoading](https://github.com/nathandecaux/labelprop/blob/main/labelprop/DataLoading.py) module.

- [Dataloading](#dataloading)
  - [BatchLabelPropDataModule](#batchlabelpropdatamodule)
    - [BatchLabelPropDataModule().setup](#batchlabelpropdatamodule()setup)
    - [BatchLabelPropDataModule().train_dataloader](#batchlabelpropdatamodule()train_dataloader)
  - [FullScan](#fullscan)
    - [FullScan().norm](#fullscan()norm)
    - [FullScan().resample](#fullscan()resample)
  - [LabelPropDataModule](#labelpropdatamodule)
    - [LabelPropDataModule().setup](#labelpropdatamodule()setup)
    - [LabelPropDataModule().test_dataloader](#labelpropdatamodule()test_dataloader)
    - [LabelPropDataModule().train_dataloader](#labelpropdatamodule()train_dataloader)
  - [PreTrainingDataModule](#pretrainingdatamodule)
    - [PreTrainingDataModule().setup](#pretrainingdatamodule()setup)
    - [PreTrainingDataModule().train_dataloader](#pretrainingdatamodule()train_dataloader)
  - [UnsupervisedScan](#unsupervisedscan)
    - [UnsupervisedScan().norm](#unsupervisedscan()norm)
  - [to_one_hot](#to_one_hot)

## BatchLabelPropDataModule

[Show source in DataLoading.py:299](https://github.com/nathandecaux/labelprop/blob/main/labelprop/DataLoading.py#L299)

Equivalent to LabelPropDataModule, but for multiple images

#### Signature

```python
class BatchLabelPropDataModule(pl.LightningDataModule):
    def __init__(
        self, img_path_list, mask_path_list, lab="all", shape=(288, 288), z_axis=0
    ): ...
```

### BatchLabelPropDataModule().setup

[Show source in DataLoading.py:314](https://github.com/nathandecaux/labelprop/blob/main/labelprop/DataLoading.py#L314)

#### Signature

```python
def setup(self, stage=None): ...
```

### BatchLabelPropDataModule().train_dataloader

[Show source in DataLoading.py:332](https://github.com/nathandecaux/labelprop/blob/main/labelprop/DataLoading.py#L332)

#### Signature

```python
def train_dataloader(self, batch_size=1): ...
```



## FullScan

[Show source in DataLoading.py:25](https://github.com/nathandecaux/labelprop/blob/main/labelprop/DataLoading.py#L25)

#### Signature

```python
class FullScan(data.Dataset):
    def __init__(
        self,
        X,
        Y,
        lab="all",
        shape=256,
        selected_slices=None,
        z_axis=-1,
        hints=None,
        isotropic=True,
    ): ...
```

### FullScan().norm

[Show source in DataLoading.py:137](https://github.com/nathandecaux/labelprop/blob/main/labelprop/DataLoading.py#L137)

#### Signature

```python
def norm(self, x, z_axis): ...
```

### FullScan().resample

[Show source in DataLoading.py:127](https://github.com/nathandecaux/labelprop/blob/main/labelprop/DataLoading.py#L127)

#### Signature

```python
def resample(self, X, Y, size): ...
```



## LabelPropDataModule

[Show source in DataLoading.py:195](https://github.com/nathandecaux/labelprop/blob/main/labelprop/DataLoading.py#L195)

#### Signature

```python
class LabelPropDataModule(pl.LightningDataModule):
    def __init__(
        self,
        img_path,
        mask_path,
        lab="all",
        shape=(288, 288),
        selected_slices=None,
        z_axis=0,
        hints=None,
    ): ...
```

### LabelPropDataModule().setup

[Show source in DataLoading.py:228](https://github.com/nathandecaux/labelprop/blob/main/labelprop/DataLoading.py#L228)

#### Signature

```python
def setup(self, stage=None): ...
```

### LabelPropDataModule().test_dataloader

[Show source in DataLoading.py:259](https://github.com/nathandecaux/labelprop/blob/main/labelprop/DataLoading.py#L259)

#### Signature

```python
def test_dataloader(self): ...
```

### LabelPropDataModule().train_dataloader

[Show source in DataLoading.py:256](https://github.com/nathandecaux/labelprop/blob/main/labelprop/DataLoading.py#L256)

#### Signature

```python
def train_dataloader(self, batch_size=1): ...
```



## PreTrainingDataModule

[Show source in DataLoading.py:263](https://github.com/nathandecaux/labelprop/blob/main/labelprop/DataLoading.py#L263)

#### Signature

```python
class PreTrainingDataModule(pl.LightningDataModule):
    def __init__(self, img_list, shape=(288, 288), z_axis=0): ...
```

### PreTrainingDataModule().setup

[Show source in DataLoading.py:278](https://github.com/nathandecaux/labelprop/blob/main/labelprop/DataLoading.py#L278)

#### Signature

```python
def setup(self, stage=None): ...
```

### PreTrainingDataModule().train_dataloader

[Show source in DataLoading.py:293](https://github.com/nathandecaux/labelprop/blob/main/labelprop/DataLoading.py#L293)

#### Signature

```python
def train_dataloader(self, batch_size=None): ...
```



## UnsupervisedScan

[Show source in DataLoading.py:148](https://github.com/nathandecaux/labelprop/blob/main/labelprop/DataLoading.py#L148)

#### Signature

```python
class UnsupervisedScan(data.Dataset):
    def __init__(self, X, shape=256, z_axis=-1, name=""): ...
```

### UnsupervisedScan().norm

[Show source in DataLoading.py:184](https://github.com/nathandecaux/labelprop/blob/main/labelprop/DataLoading.py#L184)

#### Signature

```python
def norm(self, x): ...
```



## to_one_hot

[Show source in DataLoading.py:12](https://github.com/nathandecaux/labelprop/blob/main/labelprop/DataLoading.py#L12)

One hot encoding of a label tensor
Input:
    Y: label tensor
    dim: dimension where to apply one hot encoding

#### Signature

```python
def to_one_hot(Y, n_labels, dim=1): ...
```