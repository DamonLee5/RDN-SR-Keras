import tensorflow as tf
import numpy as np
import time 
import os
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate,GaussianNoise,Lambda,ConvLSTM2D,Bidirectional
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.applications import VGG19
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.models import load_model
from keras.losses import mean_squared_error,mean_absolute_error
from keras.initializers import RandomNormal
import keras.backend as K
import datetime
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import sys
from data_loader import DataLoader
import h5py
from model import RDN
from glob import glob
if __name__ == '__main__':
    test=['test_x3','test_x2','test_x4','test_x3_BN','test_x3_DN']
    for j in range(17,18):
        rdn=RDN(load=1,rfi=j,lfi=j)
        paths=glob('/scratch/gilbreth/li3120/dataset/DIV2K_train_HR/Test/*/%s.h5'%(test[j-16]))
        for path in paths:
            rdn.predict_process(path=path)
        print(path)
        del rdn
        
