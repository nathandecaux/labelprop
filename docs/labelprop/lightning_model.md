# Lightning Model

[Labelprop Index](../README.md#labelprop-index) / [Labelprop](./index.md#labelprop) / Lightning Model

> Auto-generated documentation for [labelprop.lightning_model](https://github.com/nathandecaux/labelprop/blob/main/labelprop/lightning_model.py) module.

- [Lightning Model](#lightning-model)
  - [LabelProp](#labelprop)
    - [LabelProp().apply_deform](#labelprop()apply_deform)
    - [LabelProp().automatic_optimization](#labelprop()automatic_optimization)
    - [LabelProp().blend](#labelprop()blend)
    - [LabelProp().compose_deformation](#labelprop()compose_deformation)
    - [LabelProp().compose_list](#labelprop()compose_list)
    - [LabelProp().compute_contour_loss](#labelprop()compute_contour_loss)
    - [LabelProp().compute_loss](#labelprop()compute_loss)
    - [LabelProp().configure_optimizers](#labelprop()configure_optimizers)
    - [LabelProp().forward](#labelprop()forward)
    - [LabelProp().hardmax](#labelprop()hardmax)
    - [LabelProp().norm](#labelprop()norm)
    - [LabelProp().register_images](#labelprop()register_images)
    - [LabelProp().training_step](#labelprop()training_step)
    - [LabelProp().validation_step](#labelprop()validation_step)
    - [LabelProp().weighting_loss](#labelprop()weighting_loss)
  - [MTL_loss](#mtl_loss)
    - [MTL_loss().forward](#mtl_loss()forward)
    - [MTL_loss().set_dict](#mtl_loss()set_dict)

## LabelProp

[Show source in lightning_model.py:23](https://github.com/nathandecaux/labelprop/blob/main/labelprop/lightning_model.py#L23)

#### Signature

```python
class LabelProp(pl.LightningModule):
    def __init__(
        self,
        n_channels=1,
        n_classes=2,
        learning_rate=0.0005,
        weight_decay=1e-08,
        way="both",
        shape=256,
        selected_slices=None,
        losses={},
        by_composition=False,
        unsupervised=False,
    ): ...
```

### LabelProp().apply_deform

[Show source in lightning_model.py:88](https://github.com/nathandecaux/labelprop/blob/main/labelprop/lightning_model.py#L88)

Apply deformation to x from flow field

#### Arguments

- `x` *Tensor* - Image or mask to deform (BxCxHxW)
- `field` *Tensor* - Deformation field (Bx2xHxW)

#### Returns

- `Tensor` - Transformed image

#### Signature

```python
def apply_deform(self, x, field, ismask=False): ...
```

### LabelProp().automatic_optimization

[Show source in lightning_model.py:24](https://github.com/nathandecaux/labelprop/blob/main/labelprop/lightning_model.py#L24)

#### Signature

```python
@property
def automatic_optimization(self): ...
```

### LabelProp().blend

[Show source in lightning_model.py:219](https://github.com/nathandecaux/labelprop/blob/main/labelprop/lightning_model.py#L219)

#### Signature

```python
def blend(self, x, y): ...
```

### LabelProp().compose_deformation

[Show source in lightning_model.py:118](https://github.com/nathandecaux/labelprop/blob/main/labelprop/lightning_model.py#L118)

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

### LabelProp().compose_list

[Show source in lightning_model.py:101](https://github.com/nathandecaux/labelprop/blob/main/labelprop/lightning_model.py#L101)

Composes a list of flows by applying each flow in reverse order to the last flow.

#### Arguments

- `flows` *list* - A list of flows to be composed.

#### Returns

The composed flow.

#### Signature

```python
def compose_list(self, flows): ...
```

### LabelProp().compute_contour_loss

[Show source in lightning_model.py:205](https://github.com/nathandecaux/labelprop/blob/main/labelprop/lightning_model.py#L205)

#### Signature

```python
def compute_contour_loss(self, img, moved_mask): ...
```

### LabelProp().compute_loss

[Show source in lightning_model.py:166](https://github.com/nathandecaux/labelprop/blob/main/labelprop/lightning_model.py#L166)

#### Arguments

moved : Transformed anatomical image
target : Target anatomical image
moved_mask : Transformed mask
target_mask : Target mask
field : Velocity field (=non integrated)

#### Signature

```python
def compute_loss(
    self, moved=None, target=None, moved_mask=None, target_mask=None, field=None
): ...
```

### LabelProp().configure_optimizers

[Show source in lightning_model.py:758](https://github.com/nathandecaux/labelprop/blob/main/labelprop/lightning_model.py#L758)

#### Signature

```python
def configure_optimizers(self): ...
```

### LabelProp().forward

[Show source in lightning_model.py:129](https://github.com/nathandecaux/labelprop/blob/main/labelprop/lightning_model.py#L129)

#### Arguments

- `moving` *Tensor* - Moving image (BxCxHxW)
- `target` *[type]* - Fixed image (BxCxHxW)
- `registration` *bool, optional* - If False, also return non-integrated inverse flow field. Else return the integrated one. Defaults to False.

#### Returns

- `moved` *Tensor* - Moved image
- `field` *Tensor* - Deformation field from moving to target

#### Signature

```python
def forward(self, moving, target, registration=True): ...
```

### LabelProp().hardmax

[Show source in lightning_model.py:766](https://github.com/nathandecaux/labelprop/blob/main/labelprop/lightning_model.py#L766)

#### Signature

```python
def hardmax(self, Y, dim): ...
```

### LabelProp().norm

[Show source in lightning_model.py:28](https://github.com/nathandecaux/labelprop/blob/main/labelprop/lightning_model.py#L28)

#### Signature

```python
def norm(self, x): ...
```

### LabelProp().register_images

[Show source in lightning_model.py:754](https://github.com/nathandecaux/labelprop/blob/main/labelprop/lightning_model.py#L754)

#### Signature

```python
def register_images(self, moving, target, moving_mask): ...
```

### LabelProp().training_step

[Show source in lightning_model.py:225](https://github.com/nathandecaux/labelprop/blob/main/labelprop/lightning_model.py#L225)

Perform a single training step on the given batch of data.

#### Arguments

- `batch` - The input batch of data.
- `batch_nb` - The batch number.

#### Returns

The total loss computed during the training step.

#### Signature

```python
def training_step(self, batch, batch_nb): ...
```

### LabelProp().validation_step

[Show source in lightning_model.py:708](https://github.com/nathandecaux/labelprop/blob/main/labelprop/lightning_model.py#L708)

Deprecated

#### Signature

```python
def validation_step(self, batch, batch_idx): ...
```

### LabelProp().weighting_loss

[Show source in lightning_model.py:211](https://github.com/nathandecaux/labelprop/blob/main/labelprop/lightning_model.py#L211)

#### Arguments

- `losses` *dict* - Dictionary of losses

#### Returns

- `loss` *Tensor* - Weighted loss

#### Signature

```python
def weighting_loss(self, losses): ...
```



## MTL_loss

[Show source in lightning_model.py:770](https://github.com/nathandecaux/labelprop/blob/main/labelprop/lightning_model.py#L770)

Multi-task learning loss. Not used

#### Signature

```python
class MTL_loss(torch.nn.Module):
    def __init__(self, losses): ...
```

### MTL_loss().forward

[Show source in lightning_model.py:788](https://github.com/nathandecaux/labelprop/blob/main/labelprop/lightning_model.py#L788)

#### Signature

```python
def forward(self, loss_dict): ...
```

### MTL_loss().set_dict

[Show source in lightning_model.py:782](https://github.com/nathandecaux/labelprop/blob/main/labelprop/lightning_model.py#L782)

#### Signature

```python
def set_dict(self, dic): ...
```