# Train

[Labelprop Index](../README.md#labelprop-index) / [Labelprop](./index.md#labelprop) / Train

> Auto-generated documentation for [labelprop.train](https://github.com/nathandecaux/labelprop/blob/main/labelprop/train.py) module.

- [Train](#train)
  - [inference](#inference)
  - [train](#train)
  - [train_and_eval](#train_and_eval)

## inference

[Show source in train.py:121](https://github.com/nathandecaux/labelprop/blob/main/labelprop/train.py#L121)

Perform inference using a trained LabelProp model.

#### Arguments

- `datamodule` *DataModule* - The data module used for training.
- `model_PARAMS` *dict* - The parameters used to initialize the LabelProp model.
- `ckpt` *str* - The path to the checkpoint file of the trained model.
- `**kwargs` - Additional keyword arguments.

#### Returns

- `Tuple` - A tuple containing the predicted labels for the up, down, and fused directions.

#### Signature

```python
def inference(datamodule, model_PARAMS, ckpt, **kwargs): ...
```



## train

[Show source in train.py:71](https://github.com/nathandecaux/labelprop/blob/main/labelprop/train.py#L71)

Train the model using the given data module and parameters.

#### Arguments

- `datamodule` - The data module used for training.
- `model_PARAMS` - The parameters for the model.
- `max_epochs` - The maximum number of epochs for training.
- `ckpt` - The checkpoint path for resuming training (optional).
- `pretraining` - Whether to perform pretraining (default: False).
- `**kwargs` - Additional keyword arguments.

#### Returns

- `model` - The trained model.
- `best_ckpt` - The path to the best model checkpoint.

#### Signature

```python
def train(
    datamodule, model_PARAMS, max_epochs, ckpt=None, pretraining=False, **kwargs
): ...
```



## train_and_eval

[Show source in train.py:23](https://github.com/nathandecaux/labelprop/blob/main/labelprop/train.py#L23)

Train and evaluate a LabelProp model.

#### Arguments

- `datamodule` *DataModule* - The data module containing the dataset.
- `model_PARAMS` *dict* - The parameters for the LabelProp model.
- `max_epochs` *int* - The maximum number of epochs to train the model.
- `ckpt` *str, optional* - The path to a checkpoint file to load the model from. Defaults to None.

#### Returns

- `tuple` - A tuple containing the trained model, the propagated labels (Y_up), the inverse propagated labels (Y_down),
and the evaluation results.

#### Signature

```python
def train_and_eval(datamodule, model_PARAMS, max_epochs, ckpt=None): ...
```