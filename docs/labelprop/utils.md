# Utils

[Labelprop Index](../README.md#labelprop-index) / [Labelprop](./index.md#labelprop) / Utils

> Auto-generated documentation for [labelprop.utils](https://github.com/nathandecaux/labelprop/blob/main/labelprop/utils.py) module.

- [Utils](#utils)
  - [Normalize](#normalize)
    - [Normalize().forward](#normalize()forward)
  - [NumpyEncoder](#numpyencoder)
    - [NumpyEncoder().default](#numpyencoder()default)
  - [SuperST](#superst)
    - [SuperST().apply_deform](#superst()apply_deform)
    - [SuperST().compose_deformation](#superst()compose_deformation)
    - [SuperST().compose_list](#superst()compose_list)
    - [SuperST().forward](#superst()forward)
  - [binarize](#binarize)
  - [complex_propagation](#complex_propagation)
  - [compute_metrics](#compute_metrics)
  - [create_dict](#create_dict)
  - [fuse_up_and_down](#fuse_up_and_down)
  - [get_chunks](#get_chunks)
  - [get_dices](#get_dices)
  - [get_successive_fields](#get_successive_fields)
  - [get_weights](#get_weights)
  - [hardmax](#hardmax)
  - [propagate_by_composition](#propagate_by_composition)
  - [propagate_labels](#propagate_labels)
  - [remove_annotations](#remove_annotations)
  - [to_batch](#to_batch)
  - [to_one_hot](#to_one_hot)

## Normalize

[Show source in utils.py:228](https://github.com/nathandecaux/labelprop/blob/main/labelprop/utils.py#L228)

#### Signature

```python
class Normalize(torch.nn.Module):
    def __init__(self, dim=2): ...
```

### Normalize().forward

[Show source in utils.py:233](https://github.com/nathandecaux/labelprop/blob/main/labelprop/utils.py#L233)

#### Signature

```python
def forward(self, x): ...
```



## NumpyEncoder

[Show source in utils.py:18](https://github.com/nathandecaux/labelprop/blob/main/labelprop/utils.py#L18)

Special json encoder for numpy types

#### Signature

```python
class NumpyEncoder(json.JSONEncoder): ...
```

### NumpyEncoder().default

[Show source in utils.py:20](https://github.com/nathandecaux/labelprop/blob/main/labelprop/utils.py#L20)

#### Signature

```python
def default(self, obj): ...
```



## SuperST

[Show source in utils.py:29](https://github.com/nathandecaux/labelprop/blob/main/labelprop/utils.py#L29)

Inherit from SpatialTransformer

#### Signature

```python
class SuperST(torch.nn.Module):
    def __init__(self, *args, **kwargs): ...
```

### SuperST().apply_deform

[Show source in utils.py:56](https://github.com/nathandecaux/labelprop/blob/main/labelprop/utils.py#L56)

#### Signature

```python
def apply_deform(self, x, flow): ...
```

### SuperST().compose_deformation

[Show source in utils.py:35](https://github.com/nathandecaux/labelprop/blob/main/labelprop/utils.py#L35)

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

### SuperST().compose_list

[Show source in utils.py:46](https://github.com/nathandecaux/labelprop/blob/main/labelprop/utils.py#L46)

#### Signature

```python
def compose_list(self, flows): ...
```

### SuperST().forward

[Show source in utils.py:53](https://github.com/nathandecaux/labelprop/blob/main/labelprop/utils.py#L53)

#### Signature

```python
def forward(self, x, flow): ...
```



## binarize

[Show source in utils.py:86](https://github.com/nathandecaux/labelprop/blob/main/labelprop/utils.py#L86)

#### Signature

```python
def binarize(Y, lab): ...
```



## complex_propagation

[Show source in utils.py:375](https://github.com/nathandecaux/labelprop/blob/main/labelprop/utils.py#L375)

Deprecated

#### Signature

```python
def complex_propagation(X, Y, model): ...
```



## compute_metrics

[Show source in utils.py:399](https://github.com/nathandecaux/labelprop/blob/main/labelprop/utils.py#L399)

#### Signature

```python
def compute_metrics(y_pred, y): ...
```



## create_dict

[Show source in utils.py:96](https://github.com/nathandecaux/labelprop/blob/main/labelprop/utils.py#L96)

#### Signature

```python
def create_dict(keys, values): ...
```



## fuse_up_and_down

[Show source in utils.py:128](https://github.com/nathandecaux/labelprop/blob/main/labelprop/utils.py#L128)

Fuse up and down predictions using weights

#### Arguments

- `Y_up` *[Tensor]* - Up predictions
- `Y_down` *[Tensor]* - Down predictions
- `weights` *[Tensor]* - Weights

#### Returns

- `[Tensor]` - Fused predictions

#### Signature

```python
def fuse_up_and_down(Y_up, Y_down, weights): ...
```



## get_chunks

[Show source in utils.py:67](https://github.com/nathandecaux/labelprop/blob/main/labelprop/utils.py#L67)

#### Signature

```python
def get_chunks(Y): ...
```



## get_dices

[Show source in utils.py:417](https://github.com/nathandecaux/labelprop/blob/main/labelprop/utils.py#L417)

Compute dices for each slice of the volume, and for each label.

#### Arguments

- `Y_dense` *torch.Tensor* - Dense ground truth segmentation.
- `Y` *torch.Tensor* - Predicted segmentation in one direction.
- `Y2` *torch.Tensor* - Predicted segmentation in the other direction.
- `selected_slices` *list* - List of index of manually segmented slices that were provided during the inference.

#### Signature

```python
def get_dices(Y_dense, Y, Y2, selected_slices): ...
```



## get_successive_fields

[Show source in utils.py:166](https://github.com/nathandecaux/labelprop/blob/main/labelprop/utils.py#L166)

Generate successive fields between slices, in both directions

#### Arguments

- `X` *[Tensor]* - Volume
- `model` *[type]* - Registration model

#### Returns

- `[list]` - List of fields in one direction
- `[list]` - List of fields in the other direction

#### Signature

```python
def get_successive_fields(X, model): ...
```



## get_weights

[Show source in utils.py:102](https://github.com/nathandecaux/labelprop/blob/main/labelprop/utils.py#L102)

Get weights for each slice based on the distance to the closest annotated slice

#### Arguments

- `Y` *[Tensor]* - Sparsely annotated volume segmentation

#### Returns

- `[Tensor]` - Weights

#### Signature

```python
def get_weights(Y): ...
```



## hardmax

[Show source in utils.py:64](https://github.com/nathandecaux/labelprop/blob/main/labelprop/utils.py#L64)

#### Signature

```python
def hardmax(Y, dim): ...
```



## propagate_by_composition

[Show source in utils.py:236](https://github.com/nathandecaux/labelprop/blob/main/labelprop/utils.py#L236)

Propagate labels by composition of transformations

#### Arguments

- `X` *torch.Tensor* - input volume
- `Y` *torch.Tensor* - input labels (sparse)
- `hints` *torch.Tensor* - input hints
- `model` *torch.nn.Module* - model to use for registration
fields (list) (optionnal): list of fields to use for composition (if None, will be computed)
criteria ('ncc' or 'distance') (optionnal): criteria to use for weighting. Default is ncc
reduction ('none', 'mean' or 'local_mean') (optionnal): reduction method to use for weighting. Default is none
func (torch.nn.Module) (optionnal): function to use for normalization. Default is Normalize(2)
device (str) (optionnal): device to use. Default is 'cuda'
return_weights (bool) (optionnal): whether to return weights. Default is False
patch_size (int) (optionnal): patch size to compute NCC. Default is 31 (31x31 px)
extrapolate (bool) (optionnal): whether to extrapolate propagation to the whole volume. Default is False

#### Returns

- `torch.Tensor` - propagated labels in a single direction
- `torch.Tensor` - propagated labels in the other direction
- `torch.Tensor` - propagated labels in both directions (fused)

if return_weights:
    - `torch.Tensor` - weights used for fusion

#### Signature

```python
def propagate_by_composition(
    X,
    Y,
    hints,
    model,
    fields=None,
    criteria="ncc",
    reduction="none",
    func=Normalize(2),
    device="cuda",
    return_weights=False,
    patch_size=31,
    extrapolate=False,
): ...
```



## propagate_labels

[Show source in utils.py:190](https://github.com/nathandecaux/labelprop/blob/main/labelprop/utils.py#L190)

Deprecated

#### Signature

```python
def propagate_labels(X, Y, model, model_down=None): ...
```



## remove_annotations

[Show source in utils.py:89](https://github.com/nathandecaux/labelprop/blob/main/labelprop/utils.py#L89)

#### Signature

```python
def remove_annotations(Y, selected_slices): ...
```



## to_batch

[Show source in utils.py:61](https://github.com/nathandecaux/labelprop/blob/main/labelprop/utils.py#L61)

#### Signature

```python
def to_batch(x, device="cpu"): ...
```



## to_one_hot

[Show source in utils.py:58](https://github.com/nathandecaux/labelprop/blob/main/labelprop/utils.py#L58)

#### Signature

```python
def to_one_hot(Y, dim=0): ...
```