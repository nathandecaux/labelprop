# Pretraining Model

[Labelprop Index](../README.md#labelprop-index) / [Labelprop](./index.md#labelprop) / Pretraining Model

> Auto-generated documentation for [labelprop.Pretraining_model](https://github.com/nathandecaux/labelprop/blob/main/labelprop/Pretraining_model.py) module.

- [Pretraining Model](#pretraining-model)
  - [LabelProp](#labelprop)
    - [LabelProp().apply_deform](#labelprop()apply_deform)
    - [LabelProp().apply_successive_transformations](#labelprop()apply_successive_transformations)
    - [LabelProp().automatic_optimization](#labelprop()automatic_optimization)
    - [LabelProp().blend](#labelprop()blend)
    - [LabelProp().compose_deformation](#labelprop()compose_deformation)
    - [LabelProp().compose_list](#labelprop()compose_list)
    - [LabelProp().compute_loss](#labelprop()compute_loss)
    - [LabelProp().configure_optimizers](#labelprop()configure_optimizers)
    - [LabelProp().forward](#labelprop()forward)
    - [LabelProp().hardmax](#labelprop()hardmax)
    - [LabelProp().multi_class_dice](#labelprop()multi_class_dice)
    - [LabelProp().norm](#labelprop()norm)
    - [LabelProp().register_images](#labelprop()register_images)
    - [LabelProp().training_step](#labelprop()training_step)

## LabelProp

[Show source in Pretraining_model.py:12](https://github.com/nathandecaux/labelprop/blob/main/labelprop/Pretraining_model.py#L12)

#### Signature

```python
class LabelProp(pl.LightningModule):
    def __init__(
        self,
        n_channels=1,
        n_classes=2,
        learning_rate=0.001,
        weight_decay=1e-08,
        way="up",
        shape=256,
        selected_slices=None,
        losses={},
        by_composition=False,
    ): ...
```

### LabelProp().apply_deform

[Show source in Pretraining_model.py:46](https://github.com/nathandecaux/labelprop/blob/main/labelprop/Pretraining_model.py#L46)

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

### LabelProp().apply_successive_transformations

[Show source in Pretraining_model.py:77](https://github.com/nathandecaux/labelprop/blob/main/labelprop/Pretraining_model.py#L77)

#### Arguments

- `moving` *Tensor* - Moving image (BxCxHxW)
- `flows` *[Tensor]* - List of deformation fields (Bx2xHxW)

#### Returns

- `Tensor` - Transformed image

#### Signature

```python
def apply_successive_transformations(self, moving, flows): ...
```

### LabelProp().automatic_optimization

[Show source in Pretraining_model.py:14](https://github.com/nathandecaux/labelprop/blob/main/labelprop/Pretraining_model.py#L14)

#### Signature

```python
@property
def automatic_optimization(self): ...
```

### LabelProp().blend

[Show source in Pretraining_model.py:136](https://github.com/nathandecaux/labelprop/blob/main/labelprop/Pretraining_model.py#L136)

#### Signature

```python
def blend(self, x, y): ...
```

### LabelProp().compose_deformation

[Show source in Pretraining_model.py:66](https://github.com/nathandecaux/labelprop/blob/main/labelprop/Pretraining_model.py#L66)

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

[Show source in Pretraining_model.py:59](https://github.com/nathandecaux/labelprop/blob/main/labelprop/Pretraining_model.py#L59)

#### Signature

```python
def compose_list(self, flows): ...
```

### LabelProp().compute_loss

[Show source in Pretraining_model.py:116](https://github.com/nathandecaux/labelprop/blob/main/labelprop/Pretraining_model.py#L116)

#### Arguments

moved : Transformed anatomical image
target : Target anatomical image
moved_mask : Transformed mask
target_mask : Target mask
field : Deformation field

#### Signature

```python
def compute_loss(
    self, moved=None, target=None, moved_mask=None, target_mask=None, field=None
): ...
```

### LabelProp().configure_optimizers

[Show source in Pretraining_model.py:262](https://github.com/nathandecaux/labelprop/blob/main/labelprop/Pretraining_model.py#L262)

#### Signature

```python
def configure_optimizers(self): ...
```

### LabelProp().forward

[Show source in Pretraining_model.py:103](https://github.com/nathandecaux/labelprop/blob/main/labelprop/Pretraining_model.py#L103)

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

[Show source in Pretraining_model.py:265](https://github.com/nathandecaux/labelprop/blob/main/labelprop/Pretraining_model.py#L265)

#### Signature

```python
def hardmax(self, Y, dim): ...
```

### LabelProp().multi_class_dice

[Show source in Pretraining_model.py:90](https://github.com/nathandecaux/labelprop/blob/main/labelprop/Pretraining_model.py#L90)

#### Arguments

- `pred_with_logits` - Predicted mask with logits
- `target` - Target mask

#### Signature

```python
def multi_class_dice(self, pred_with_logits, target): ...
```

### LabelProp().norm

[Show source in Pretraining_model.py:17](https://github.com/nathandecaux/labelprop/blob/main/labelprop/Pretraining_model.py#L17)

#### Signature

```python
def norm(self, x): ...
```

### LabelProp().register_images

[Show source in Pretraining_model.py:258](https://github.com/nathandecaux/labelprop/blob/main/labelprop/Pretraining_model.py#L258)

#### Signature

```python
def register_images(self, moving, target, moving_mask): ...
```

### LabelProp().training_step

[Show source in Pretraining_model.py:142](https://github.com/nathandecaux/labelprop/blob/main/labelprop/Pretraining_model.py#L142)

#### Signature

```python
def training_step(self, batch, batch_nb): ...
```