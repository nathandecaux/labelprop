# Weight Optimizer

[Labelprop Index](../README.md#labelprop-index) / [Labelprop](./index.md#labelprop) / Weight Optimizer

> Auto-generated documentation for [labelprop.weight_optimizer](https://github.com/nathandecaux/labelprop/blob/main/labelprop/weight_optimizer.py) module.

- [Weight Optimizer](#weight-optimizer)
  - [MLP](#mlp)
    - [MLP().configure_optimizers](#mlp()configure_optimizers)
    - [MLP().forward](#mlp()forward)
    - [MLP().training_step](#mlp()training_step)
  - [WeightsDataset](#weightsdataset)
  - [optimize_weights](#optimize_weights)

## MLP

[Show source in weight_optimizer.py:14](https://github.com/nathandecaux/labelprop/blob/main/labelprop/weight_optimizer.py#L14)

#### Signature

```python
class MLP(pl.LightningModule):
    def __init__(self, hidden_size=16, learning_rate=1e-05): ...
```

### MLP().configure_optimizers

[Show source in weight_optimizer.py:38](https://github.com/nathandecaux/labelprop/blob/main/labelprop/weight_optimizer.py#L38)

#### Signature

```python
def configure_optimizers(self): ...
```

### MLP().forward

[Show source in weight_optimizer.py:22](https://github.com/nathandecaux/labelprop/blob/main/labelprop/weight_optimizer.py#L22)

#### Signature

```python
def forward(self, x): ...
```

### MLP().training_step

[Show source in weight_optimizer.py:28](https://github.com/nathandecaux/labelprop/blob/main/labelprop/weight_optimizer.py#L28)

#### Signature

```python
def training_step(self, batch, batch_idx): ...
```



## WeightsDataset

[Show source in weight_optimizer.py:41](https://github.com/nathandecaux/labelprop/blob/main/labelprop/weight_optimizer.py#L41)

#### Signature

```python
class WeightsDataset(torch.data.Dataset):
    def __init__(self, weights, Y_up, Y_down, Y_true): ...
```



## optimize_weights

[Show source in weight_optimizer.py:59](https://github.com/nathandecaux/labelprop/blob/main/labelprop/weight_optimizer.py#L59)

#### Signature

```python
def optimize_weights(weights, Y_up, Y_down, Y_true, ckpt=None, learning_rate=1e-05): ...
```