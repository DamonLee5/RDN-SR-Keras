import numpy as np
import h5py
import os
from glob import glob
import matplotlib.pyplot as plt
from skimage.transform import rescale
from skimage.measure import compare_psnr,compare_ssim

paths=glob('/scratch/gilbreth/li3120/RDB/Test_Result/*')

for path in paths:
    print(path)
    f= h5py.File(path, "r")
    rec=f['rec']
    gt=f['gt']
    Ave_ssim=np.zeros((len(gt),1))
    Ave_psnr=np.zeros((len(gt),1))
    def rgb2ycbcr (img):
        y = 16 + (65.481 * img[:, :, 0]) + (128.553 * img[:, :, 1]) + (24.966 * img[:, :, 2])
        return y / 255
    for i in range(len(gt)):
        img=np.squeeze(rec[i,:,:,:])
        imgg=np.squeeze(gt[i,:,:,:])
        img=np.clip(img,a_min=0,a_max=1)
        img=rgb2ycbcr(img)
        imgg=rgb2ycbcr(imgg)
        Ave_ssim[i]=compare_ssim(img,imgg)
        Ave_psnr[i]=compare_psnr(img,imgg)
    print('average ssim:%f'%(np.mean(Ave_ssim)))
    print('average psnr:%f'%(np.mean(Ave_psnr)))
    f.close()
