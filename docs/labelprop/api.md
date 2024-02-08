# Api

[Labelprop Index](../README.md#labelprop-index) / [Labelprop](./index.md#labelprop) / Api

> Auto-generated documentation for [labelprop.api](https://github.com/nathandecaux/labelprop/blob/main/labelprop/api.py) module.

- [Api](#api)
  - [check](#check)
  - [check_tmp](#check_tmp)
  - [create_buf_npz](#create_buf_npz)
  - [download_inference](#download_inference)
  - [get_ckpt_dir](#get_ckpt_dir)
  - [get_session_info](#get_session_info)
  - [get_session_list](#get_session_list)
  - [get_tmp_hashed_img](#get_tmp_hashed_img)
  - [hash_array](#hash_array)
  - [inference](#inference)
  - [list_ckpts](#list_ckpts)
  - [list_hashes](#list_hashes)
  - [send_ckpt](#send_ckpt)
  - [set_ckpt_dir](#set_ckpt_dir)
  - [timer](#timer)
  - [training](#training)

## check

[Show source in api.py:30](https://github.com/nathandecaux/labelprop/blob/main/labelprop/api.py#L30)

#### Signature

```python
@app.route("/check", methods=["GET"])
def check(): ...
```



## check_tmp

[Show source in api.py:25](https://github.com/nathandecaux/labelprop/blob/main/labelprop/api.py#L25)

Check if /tmp directory exists, if not create it (watch out for permissions in Windows)

#### Signature

```python
def check_tmp(): ...
```



## create_buf_npz

[Show source in api.py:59](https://github.com/nathandecaux/labelprop/blob/main/labelprop/api.py#L59)

Create a compressed NumPy .npz file from a dictionary of arrays.

#### Arguments

- `array_dict` *dict* - A dictionary containing arrays to be saved in the .npz file.

#### Returns

- `io.BytesIO` - A BytesIO object containing the compressed .npz file.

#### Signature

```python
def create_buf_npz(array_dict): ...
```



## download_inference

[Show source in api.py:261](https://github.com/nathandecaux/labelprop/blob/main/labelprop/api.py#L261)

Return the inference results as a zip file.

#### Signature

```python
@app.route("/download_inference", methods=["GET", "POST"])
def download_inference(): ...
```



## get_ckpt_dir

[Show source in api.py:34](https://github.com/nathandecaux/labelprop/blob/main/labelprop/api.py#L34)

#### Signature

```python
@app.route("/get_ckpt_dir", methods=["GET"])
def get_ckpt_dir(): ...
```



## get_session_info

[Show source in api.py:279](https://github.com/nathandecaux/labelprop/blob/main/labelprop/api.py#L279)

Return the session info for a given token.

#### Signature

```python
@app.route("/get_session_info", methods=["GET"])
def get_session_info(): ...
```



## get_session_list

[Show source in api.py:287](https://github.com/nathandecaux/labelprop/blob/main/labelprop/api.py#L287)

Return a list of all tokens.

#### Signature

```python
@app.route("/get_session_list", methods=["GET"])
def get_session_list(): ...
```



## get_tmp_hashed_img

[Show source in api.py:97](https://github.com/nathandecaux/labelprop/blob/main/labelprop/api.py#L97)

List hash from file of /tmp directory
Hased images are named as "<hash>_img.npy"

#### Signature

```python
def get_tmp_hashed_img(): ...
```



## hash_array

[Show source in api.py:93](https://github.com/nathandecaux/labelprop/blob/main/labelprop/api.py#L93)

#### Signature

```python
@timer
def hash_array(array): ...
```

#### See also

- [timer](#timer)



## inference

[Show source in api.py:109](https://github.com/nathandecaux/labelprop/blob/main/labelprop/api.py#L109)

Perform inference using the LabelProp model.

This function loads the necessary input data, processes it, and returns the inference results.
It saves the input image and mask to the temporary folder, and then calls the `propagate_from_ckpt` function
to perform the inference. The results are stored in a session dictionary.

#### Returns

- `response` *Response* - The response object containing the session token.

#### Signature

```python
@app.route("/inference", methods=["POST"])
def inference(): ...
```



## list_ckpts

[Show source in api.py:238](https://github.com/nathandecaux/labelprop/blob/main/labelprop/api.py#L238)

Return a list of all checkpoints in the checkpoint_dir.

#### Signature

```python
@app.route("/list_ckpts", methods=["GET"])
def list_ckpts(): ...
```



## list_hashes

[Show source in api.py:245](https://github.com/nathandecaux/labelprop/blob/main/labelprop/api.py#L245)

Return a list of all hash in the sessions dictionnary from get_tmp_hashed_img().

#### Signature

```python
@app.route("/list_hash", methods=["GET"])
def list_hashes(): ...
```



## send_ckpt

[Show source in api.py:252](https://github.com/nathandecaux/labelprop/blob/main/labelprop/api.py#L252)

Receive a checkpoint file and save it in the checkpoint_dir.

#### Signature

```python
@app.route("/send_ckpt", methods=["POST"])
def send_ckpt(): ...
```



## set_ckpt_dir

[Show source in api.py:39](https://github.com/nathandecaux/labelprop/blob/main/labelprop/api.py#L39)

#### Signature

```python
@app.route("/set_ckpt_dir", methods=["POST"])
def set_ckpt_dir(): ...
```



## timer

[Show source in api.py:74](https://github.com/nathandecaux/labelprop/blob/main/labelprop/api.py#L74)

Decorator that measures the execution time of a function.

#### Arguments

- `func` - The function to be timed.

#### Returns

The wrapped function.

#### Signature

```python
def timer(func): ...
```



## training

[Show source in api.py:176](https://github.com/nathandecaux/labelprop/blob/main/labelprop/api.py#L176)

This function performs the training process for the labelprop module.

It loads the necessary arrays and parameters from the request files, checks for temporary files,
and prepares the input data for training. It then calls the `train_and_infer` function to perform
the training and inference process. Finally, it saves the results and returns a response token.

#### Returns

- `response` *Response* - The response object containing the token.

#### Raises

- `Exception` - If an error occurs during the training process.

#### Signature

```python
@app.route("/training", methods=["POST"])
def training(): ...
```