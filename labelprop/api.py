from urllib import response
from flask import Flask,request,Response
from .napari_entry import propagate_from_ckpt 
import os
import numpy as np
import json
import time
app=Flask(__name__)

checkpoint_dir='/tmp/checkpoints/'
    



@app.route('/inference',methods=['POST'])
def inference():
    """
    Receive img and mask arrays as string and checkpoint,shape,z_axis,lab parameters and call propagate_from_ckpt. 
    """
    arrays=np.load(request.files['arrays'])
    img=arrays['img']
    mask=arrays['mask']
    infos=json.loads(request.files['params'].read())
    response=Response('Pouet')
    @response.call_on_close
    def process_after_request():
        propagate_from_ckpt(img,mask,**infos)
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

