from urllib import response
from flask import Flask,request,Response,send_file
from .napari_entry import propagate_from_ckpt 
import os
import numpy as np
import json
import time
import uuid
import zipfile
import io

app=Flask(__name__)

checkpoint_dir='/tmp/checkpoints/'
    
global sessions
sessions={}
def create_buf_npz(array_dict):
    buf = io.BytesIO()
    np.savez_compressed(buf, **array_dict)
    buf.seek(0)
    return buf

@app.route('/inference',methods=['POST'])
def inference():
    """
    Receive img and mask arrays as string and checkpoint,shape,z_axis,lab parameters and call propagate_from_ckpt. 
    """
    arrays=np.load(request.files['arrays'])
    img=arrays['img']
    mask=arrays['mask']
    infos=json.loads(request.files['params'].read())
    token=str(uuid.uuid4())
    print(token)
    response=Response(token)
    @response.call_on_close
    def process_after_request():
        Y_up,Y_down,Y_fused=propagate_from_ckpt(img,mask,**infos)
        sessions[token]={'img':img,'mask':mask,'infos':infos}
        sessions[token]['time']=time.time()
        sessions[token]['Y_up']=Y_up
        sessions[token]['Y_down']=Y_down
        sessions[token]['Y_fused']=Y_fused
        
    return response
    # return response
    # mask=request.form['mask']
    # checkpoint=request.form['checkpoint']
    # shape=int(request.form['shape'])
    # z_axis=int(request.form['z_axis'])
    # lab=request.form['lab']
    # Y_up,Y_down,Y_fused=propagate_from_ckpt(img,mask,checkpoint,shape,z_axis,lab)
    # return str(Y_up)+'\n'+str(Y_down)+'\n'+str(Y_fused)


@app.route('/list_ckpts',methods=['GET'])
def list_ckpts():
    """
    Return a list of all checkpoints in the checkpoint_dir.
    """
    return ','.join([x for x in os.listdir(checkpoint_dir) if '.ckpt' in x])

@app.route('/download_inference',methods=['GET','POST'])
def download_inference():
    """
    Return the inference results as a zip file.
    """
    token=request.args['token']
    Y_up=sessions[token]['Y_up']
    Y_down=sessions[token]['Y_down']
    Y_fused=sessions[token]['Y_fused']
    #Compress arrays with np.savez_compressed
    arrays={'Y_up':Y_up,'Y_down':Y_down,'Y_fused':Y_fused}
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


if __name__=='__main__':
    app.run(host='0.0.0.0',port=5000)
