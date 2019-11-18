import scipy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.io import imread
from skimage import data_dir
from skimage.transform import radon, rescale
from skimage.transform import iradon
import h5py
class DataLoader():
    def __init__(self, dataset_name, img_crop=(128, 128)):
        self.dataset_name = dataset_name
        self.img_crop = img_crop
        np.random.seed(3)
    def load_sample_data(self, batch_size=1, is_testing=False,cal_mse=False,mse_path=''):
        imgs_bl = []
        imgs_sh = []
        views=16
        f=h5py.File(mse_path, "r")
        label5=f['label']
        input5=f['input']
        for iiter,(img_bl, img_sh) in enumerate(zip(input5,label5)):
            imgs_bl.append(img_bl)
            imgs_sh.append(img_sh)
        f.close()
        imgs_bl = np.array(imgs_bl) 
        imgs_sh = np.array(imgs_sh) 
        return imgs_bl, imgs_sh 
    
    def load_data(self, batch_size=1, is_testing=False,cal_mse=False,mse_path=''):
        path = '/scratch/gilbreth/li3120/dataset/DIV2K_train_HR/Train/%s.h5' % (self.dataset_name)
        f= h5py.File(path, "r")
        input5=f['input']
        ll=len(input5)
        f.close()
        iteration=1000
        ipath=np.random.permutation(range(ll))
        ipath=ipath[:1000*batch_size]
        for i in range(iteration):
            batch = ipath[i*batch_size:(i+1)*batch_size]
            rd=np.random.randint(4,size=batch_size)
            imgs_bl = []
            imgs_sh = []
            f= h5py.File(path, "r")
            rdind=0
            for img_path in batch:
                
                label5 =f["label"]
                img_sh=label5[img_path]
                img_sh=np.array(img_sh)
                if rd[rdind]==0:
                    img_sh=np.flip(img_sh,0)
                if rd[rdind]==1:
                    img_sh=np.flip(img_sh,1)
                if rd[rdind]==2:
                    img_sh=np.rot90(img_sh,1,(0,1))
                input5=f["input"]
                img_bl=input5[img_path]
                img_bl=np.array(img_bl)
                if rd[rdind]==0:
                    img_bl=np.flip(img_bl,0)
                if rd[rdind]==1:
                    img_bl=np.flip(img_bl,1)
                if rd[rdind]==2:
                    img_bl=np.rot90(img_bl,1,(0,1))
                imgs_bl.append(img_bl)
                imgs_sh.append(img_sh)
                rdind=rdind+1
            f.close()    
            imgs_bl = np.array(imgs_bl) 
            imgs_sh = np.array(imgs_sh) 
            yield imgs_bl, imgs_sh    

    def load_test_data(self, batch_size=1, is_testing=False,cal_mse=False,mse_path=''):
        batch_images=glob('%s/*'%mse_path)
        print(batch_images[0])    
        imgs_bl = []
        imgs_sh = []
        views=16
        for img_path in batch_images:
            f= h5py.File(img_path, "r")
            img_sh =f["gt"]
            img_sh=np.expand_dims(img_sh, axis=-1)
            img_bl=f["mvbp"]
            img_bl=np.array(img_bl)
            imgs_bl.append(img_bl)
            imgs_sh.append(img_sh)
            f.close()
        imgs_bl = np.array(imgs_bl) / 1000.0
        imgs_sh = np.array(imgs_sh) / 1000.0
        return imgs_bl, imgs_sh  
