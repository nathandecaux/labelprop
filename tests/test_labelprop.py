#!/usr/bin/env python

"""Tests for `labelprop` package."""


import unittest

from labelprop.napari_entry import train_and_infer
import numpy as np
import os
import nibabel as ni
import pandas as pd
import subprocess as sp
import time
import json
import requests
import io

def create_buf_npz(array_dict):
    buf = io.BytesIO()
    np.savez_compressed(buf, **array_dict)
    buf.seek(0)
    return buf

def hash_array(array):
    return hashlib.md5(array.tobytes()).hexdigest()

class TestLabelprop(unittest.TestCase):
    """Tests for `labelprop` package."""

    def setUp(self):
        img=np.ones((20,20,20))
        img[:,0:10,0:10]=0.5

        mask=np.zeros((20,20,20))
        #Add a random square to the mask at the first and last slice
        mask[0,0:10,0:10]=1
        mask[-1,0:10,0:10]=1

        hints=np.zeros((20,20,20))
        hints[:,0:5,0:5]=1
        #Make dir /tmp/labelprop_test
        if not os.path.exists('/tmp/labelprop_test'):
            os.makedirs('/tmp/labelprop_test')


        #Save image, mask and hints as nifti
        ni.save(ni.Nifti1Image(img,np.eye(4)),'/tmp/labelprop_test/img.nii.gz')
        ni.save(ni.Nifti1Image(mask,np.eye(4)),'/tmp/labelprop_test/mask.nii.gz')
        ni.save(ni.Nifti1Image(hints,np.eye(4)),'/tmp/labelprop_test/hints.nii.gz')

        #Create a csv for pretraing and training 
        pretraining_csv=pd.DataFrame({'img_path':['/tmp/labelprop_test/img.nii.gz']})
        pretraining_csv.to_csv('/tmp/labelprop_test/pretraining.csv',index=False,header=False)

        #Kill any flask server running
        os.system('pkill -9 flask')
   


    def do_train_and_infer(self):
        """Test train_and_infer with random image and mask"""
        img=np.ones((20,20,20))
        img[:,0:10,0:10]=0.5

        mask=np.zeros((20,20,20))
        #Add a random square to the mask at the first and last slice
        mask[0,0:10,0:10]=1
        mask[-1,0:10,0:10]=1
        
        #Add smaller squares in intermediate slices as hints
        hints=np.zeros((20,20,20))
        hints[:,0:5,0:5]=1
        Y_up,Y_down,Y_fused=train_and_infer(img,mask,None,shape=20,max_epochs=1,z_axis=0,output_dir='/tmp/labelprop_test',name='test_function',pretraining=False,hints=hints)
        return Y_fused
    def test_train_and_infer(self):
        """Test train_and_infer with random image and mask"""
        Y_fused=self.do_train_and_infer()
        self.assertEqual(Y_fused.shape,(20,20,20))
        #Assert test.ckpt exists
        self.assertEqual(os.path.exists('/tmp/labelprop_test/test_function.ckpt'),True)



    def test_cli(self):
        """Test pretraining"""

        os.system('labelprop pretrain /tmp/labelprop_test/pretraining.csv --shape 20 --max_epochs 1 --z_axis 0 --output_dir /tmp/labelprop_test --name test_pretraining')
        #Assert test.ckpt exists
        self.assertEqual(os.path.exists('/tmp/labelprop_test/test_pretraining.ckpt'),True)

        os.system('labelprop train /tmp/labelprop_test/img.nii.gz /tmp/labelprop_test/mask.nii.gz --shape 20 --max_epochs 1 --z_axis 0 --output_dir /tmp/labelprop_test --name test_training --pretrained_ckpt /tmp/labelprop_test/test_pretraining.ckpt')
        self.assertEqual(os.path.exists('/tmp/labelprop_test/test_training.ckpt'),True)

        os.system('labelprop propagate /tmp/labelprop_test/img.nii.gz /tmp/labelprop_test/mask.nii.gz /tmp/labelprop_test/test_training.ckpt --shape 20 --z_axis 0 --output_dir /tmp/labelprop_test --name propagated_mask')
        self.assertEqual(ni.load('/tmp/labelprop_test/propagated_mask_fused.nii.gz').shape,(20,20,20))

    
    def test_server(self):
        #Start server
        sp.Popen('labelprop launch-server --port 5555'.split())
        #Wait for server to start
        time.sleep(10)
        
        #Check if server is running
        request='http://localhost:5555/check'
        response=requests.get(request,timeout=10)
        self.assertEqual(response.text,'Hello World!')

        #Setting checkpoint dir
        request='http://localhost:5555/set_ckpt_dir?ckpt_dir=/tmp/labelprop_test'
        response=requests.post(request,timeout=10)
        request='http://localhost:5555/get_ckpt_dir'
        response=requests.get(request,timeout=10)
        self.assertEqual(response.text,'/tmp/labelprop_test')
    

        img=ni.load('/tmp/labelprop_test/img.nii.gz').get_fdata().astype('float32')
        mask=ni.load('/tmp/labelprop_test/mask.nii.gz').get_fdata().astype('uint8')
        hints=ni.load('/tmp/labelprop_test/hints.nii.gz').get_fdata().astype('uint8')

        #def training_function(image: "napari.layers.Image", labels: "napari.layers.Labels",hints:"napari.layers.Labels", pretrained_checkpoint: "napari.types.Path" = '', shape: int=256, z_axis: int=0, label : int=0, max_epochs: int=10,checkpoint_name='',criteria='ncc',reduction='none',gpu=True) -> "napari.types.LayerDataTuple":

        params={'shape':20,'z_axis':0,'max_epochs':1,'output_dir':'/tmp/labelprop_test','name':'test_api','pretrained_ckpt':''}
        # hash=hash_array(img)
        # params['hash']=hash
        arrays={}
        arrays['img']=img
        arrays['mask']=mask
        arrays['hints']=hints

        buf=create_buf_npz(arrays)
        params=json.dumps(params).encode('utf-8')

        request='http://localhost:5555/training'
        response=requests.post(request,files={'arrays':buf,'params':params})
        token=response.text
        buf.close()

        active_sessions=requests.get('http://localhost:5555/get_session_list').text
        timeout=0
        while token not in active_sessions or timeout>30:
            time.sleep(1)
            active_sessions=requests.get('http://localhost:5555/get_session_list').text

        request='http://localhost:5555/download_inference?token='+token
        response=requests.get(request,timeout=10)
        npz_file=np.load(io.BytesIO(response.content),encoding = 'latin1',allow_pickle=True)

        self.assertEqual(npz_file['Y_fused'].shape,(20,20,20))



        

       




if __name__ == '__main__':
    unittest.main()


