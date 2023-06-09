from ntpath import join
from urllib import response
from flask import Flask,request,Response,send_file
from .napari_entry import propagate_from_ckpt,train_and_infer
import os
import numpy as np
import json
import time
import uuid
import zipfile
import io
import hashlib
import functools
import logging
import pathlib
from copy import deepcopy
app=Flask(__name__)
package_path = os.path.dirname(os.path.abspath(__file__))
server=json.load(fp=open(os.path.join(package_path,'conf.json')))
checkpoint_dir=server['checkpoint_dir']
#If checkpoint dir does not exist, create it
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

def check_tmp():
    """Check if /tmp directory exists, if not create it (watch out for permissions in Windows)"""
    if not os.path.exists('/tmp'):
        os.makedirs('/tmp')

@app.route('/get_ckpt_dir',methods=['GET'])

def get_ckpt_dir():
    server=json.load(fp=open(os.path.join(package_path,'conf.json')))
    return server['checkpoint_dir']

@app.route('/set_ckpt_dir',methods=['POST'])
def set_ckpt_dir():
    ckpt_dir=request.args['ckpt_dir']
    global checkpoint_dir
    checkpoint_dir=ckpt_dir
    try:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    except Exception as e:
        return "Unallowed to create checkpoint directory"
    #Save checkpoint dir in conf.json
    server['checkpoint_dir']=ckpt_dir
    with open(os.path.join(package_path,'conf.json'),'w') as f:
        json.dump(server,f)
    return 'ok'
# checkpoint_dir='F:/checkpoints/'
    
global sessions
sessions={}
def create_buf_npz(array_dict):
    buf = io.BytesIO()
    np.savez_compressed(buf, **array_dict)
    buf.seek(0)
    return buf

def timer(func):
    @functools.wraps(func) #optional line if you went the name of the function to be maintained, to be imported
    def wrapper(*args, **kwargs):
        start = time.time()
        #do somehting with the function
        value = func(*args, **kwargs)
        end = time.time()
        print('Time',end-start)
        return value
    return wrapper

@timer
def hash_array(array):
    return hashlib.md5(array.tobytes()).hexdigest()

def get_tmp_hashed_img():
    """
    List hash from file of /tmp directory
    Hased images are named as "<hash>_img.npy"
    """
    check_tmp()
    tmp_dir = '/tmp/'
    files = os.listdir(tmp_dir)
    hash_list = [f.split('_img')[0] for f in files if f.endswith('_img.npy')]
    return hash_list
    

@app.route('/inference',methods=['POST'])
def inference():
    """
    Receive img and mask arrays as string and checkpoint,shape,z_axis,lab parameters and call propagate_from_ckpt. 
    """
    check_tmp()
    arrays=np.load(request.files['arrays'])
    infos=json.loads(request.files['params'].read())
    if 'hash' in infos.keys():
        hash=infos['hash']
        img=np.load('/tmp/'+hash+'_img.npy')
        del infos['hash']
    else:
        img=arrays['img']
        hash=hash_array(img)
    
    token=str(uuid.uuid4())
    mask=arrays['mask']
    hints=None
    if 'hints' in arrays.keys():
        hints=arrays['hints']
    print(token)
    #Save img and mask to /tmp folder
    img_path=join('/tmp/',hash+'_img.npy')
    mask_path=join('/tmp/',hash+'_mask.npy')
    np.save(img_path,img)
    np.save(mask_path,mask)
    # if hints is not None:
    #     hints_path=join('/tmp/',hash+'_hints.npy')
    #     np.save(hints_path,hints)

    response=Response(token)
    @response.call_on_close
    def process_after_request():
        try:
            Y_up,Y_down,Y_fused=propagate_from_ckpt(img,mask,hints=hints,**infos)
            print(np.unique(Y_up))
            sessions[token]={'img':img_path,'mask':mask_path,'infos':infos}
            sessions[token]['hash']=hash
            sessions[token]['time']=time.time()
            sessions[token]['Y_up']=Y_up
            sessions[token]['Y_down']=Y_down
            sessions[token]['Y_fused']=Y_fused
        except Exception as e:
            logging.exception("message")
            sessions[token]={'error':deepcopy(str(e))}

        
    return response
    # return response
    # mask=request.form['mask']
    # checkpoint=request.form['checkpoint']
    # shape=int(request.form['shape'])
    # z_axis=int(request.form['z_axis'])
    # lab=request.form['lab']
    # Y_up,Y_down,Y_fused=propagate_from_ckpt(img,mask,checkpoint,shape,z_axis,lab)
    # return str(Y_up)+'\n'+str(Y_down)+'\n'+str(Y_fused)


@app.route('/training',methods=['POST'])
def training():
    """
    Receive img and mask arrays as string and checkpoint,shape,z_axis,lab parameters and call propagate_from_ckpt. 
    """
    check_tmp()
    arrays=np.load(request.files['arrays'])
    infos=json.loads(request.files['params'].read())
    mask=arrays['mask']
    if 'hash' in infos.keys():
        hash=infos['hash']
        img=np.load('/tmp/'+hash+'_img.npy')
        del infos['hash']
    else:
        img=arrays['img']
        hash=hash_array(img)
    hints=None
    if 'hints' in arrays.keys():
        hints=arrays['hints']
    infos['output_dir']=get_ckpt_dir()
    if infos['pretrained_ckpt']!='':
        infos['pretrained_ckpt']=os.path.join(str(get_ckpt_dir()),str(infos['pretrained_ckpt']))
    else:
        infos['pretrained_ckpt']=None
    token=str(uuid.uuid4())
    print(token)
    #Save img and mask to /tmp folder
    img_path=join('/tmp/',hash+'_img.npy')
    mask_path=join('/tmp/',hash+'_mask.npy')
    print(img_path)
    np.save(img_path,img)
    np.save(mask_path,mask)
    response=Response(token)
    @response.call_on_close
    def process_after_request():
        try:
            Y_up,Y_down,Y_fused=train_and_infer(img,mask,hints=hints,**infos)
            print(np.unique(Y_up))
            sessions[token]={'img':img_path,'mask':mask_path,'infos':infos}
            sessions[token]['time']=time.time()
            sessions[token]['Y_up']=Y_up
            sessions[token]['Y_down']=Y_down
            sessions[token]['Y_fused']=Y_fused
            sessions[token]['hash']=hash
        except Exception as e:
            logging.exception("message")
            sessions[token]={'error':e}
        
    return response


@app.route('/list_ckpts',methods=['GET'])
def list_ckpts():
    """
    Return a list of all checkpoints in the checkpoint_dir.
    """
    return ','.join([x for x in os.listdir(get_ckpt_dir()) if x.endswith('.ckpt')])

@app.route('/list_hash',methods=['GET'])
def list_hashes():
    """
    Return a list of all hash in the sessions dictionnary from get_tmp_hashed_img().
    """
    return ','.join(get_tmp_hashed_img())
 
@app.route('/send_ckpt',methods=['POST'])
def send_ckpt():
    """
    Receive a checkpoint file and save it in the checkpoint_dir.
    """
    ckpt=request.files['ckpt']
    ckpt.save(join(get_ckpt_dir(),ckpt.filename))
    return 'ok'

@app.route('/download_inference',methods=['GET','POST'])
def download_inference():
    """
    Return the inference results as a zip file.
    """
    token=request.args['token']
    if 'error' in sessions[token].keys():
        print('HELLO',sessions[token]['error'])
        return str(sessions[token]['error'])
    else:
        Y_up=sessions[token]['Y_up']
        Y_down=sessions[token]['Y_down']
        Y_fused=sessions[token]['Y_fused']
        #Compress arrays with np.savez_compressed
        arrays={'Y_up':Y_up.astype('uint8'),'Y_down':Y_down.astype('uint8'),'Y_fused':Y_fused.astype('uint8')}
        buf=create_buf_npz(arrays)
        return Response(buf)#send_file(buf,mimetype='application/x-zip-compressed',as_attachment=False,attachment_filename='inference_results.npz')

@app.route('/get_session_info',methods=['GET'])
def get_session_info():
    """
    Return the session info for a given token.
    """
    token=request.args['token']
    return json.dumps(sessions[token]['infos'])

@app.route('/get_session_list',methods=['GET'])
def get_session_list():
    """
    Return a list of all tokens.
    """
    return ','.join(sessions.keys())

# docs=FlaskApiSpec(app)

# docs.register(training)
# docs.register(list_ckpts)
# docs.register(list_hashes)
# docs.register(send_ckpt)
# docs.register(download_inference)
# docs.register(get_session_info)
# docs.register(get_session_list)
# docs.register(inference)

# app.config.update({
#     'APISPEC_SPEC': APISpec(
#         title='DeepFluorescence',
#         version='v1',
#         plugins=[MarshmallowPlugin()],
#         openapi_version='2.0.0'
#     ),
#     'APISPEC_SWAGGER_URL': '/swagger/',
#     'APISPEC_SWAGGER_UI_URL': '/swagger-ui/'
# })


if __name__=='__main__':
    app.run(host='0.0.0.0',port=5000,debug=True)
