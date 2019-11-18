import numpy as np
import h5py
import os
from glob import glob
import cv2
from skimage.transform import rescale
print('__file__={0:<35} | __name__={1:<25} | __package__={2:<25}'.format(__file__,__name__,str(__package__)))
import sys
sys.path.append("/home/li3120/matlab/lib")
import matlab.engine
eng = matlab.engine. start_matlab ()
mdouble = matlab.double
def ski_rescale(image, ratio, mode=3):
    #image_rescale = rescale(image, ratio, order=mode,preserve_range=True)
    image_rescale=np. asarray (eng. imresize ( mdouble (image .tolist ()), ratio , 'bicubic'))
    return image_rescale

def imread(path):
    img = cv2.imread(path)
    return img

def imsave(image, path):
    cv2.imwrite(path,image)

def modcrop(img,scale=3):
    if len(img.shape) ==3:
        h, w, _ = img.shape
        h=h-np.mod(h,scale)
        w=w-np.mod(w,scale)
        img = img[0:h, 0:w, :]
    else:
        h, w = img.shape
        h=h-np.mod(h,scale)
        w=w-np.mod(w,scale)
        img = img[0:h, 0:w]
    return img

def preprocess(path, scale = 3):
    img = imread(path)
    label_ = modcrop(img, scale)
    input_ = ski_rescale(label_, 1.0/scale, mode=3)

    input_ = input_[:, :, ::-1]
    label_ = label_[:, :, ::-1]

    return input_, label_

def make_data_hf(input_, label_,  times,is_test,path):
    scale=3
    img_size=32
    stride=16
    color_dim=3
    if is_test:
        pp=path.split('/')
        os.makedirs('/scratch/gilbreth/li3120/dataset/DIV2K_train_HR/Test/%s'%(pp[-1]),exist_ok=True)
        savepath='/scratch/gilbreth/li3120/dataset/DIV2K_train_HR/Test/%s/test_x%d_DN.h5'%(pp[-1],scale)
    else:
        os.makedirs('/scratch/gilbreth/li3120/dataset/DIV2K_train_HR/Train',exist_ok=True)
        savepath = '/scratch/gilbreth/li3120/dataset/DIV2K_train_HR/Train/val_x%d_DN.h5' % (scale)
    if times == 0:
        hf = h5py.File(savepath, 'w')

        input_h5 = hf.create_dataset("input", (1, img_size, img_size, color_dim),         
                                    maxshape=(None, img_size, img_size, color_dim),         
                                    chunks=(1, img_size, img_size, color_dim), dtype='float32')
        label_h5 = hf.create_dataset("label", (1, img_size*scale, img_size*scale, color_dim),         
                                    maxshape=(None, img_size*scale, img_size*scale, color_dim),                                     
                                    chunks=(1, img_size*scale, img_size*scale, color_dim),dtype='float32')
    else:
        hf = h5py.File(savepath, 'a')
        input_h5 = hf["input"]
        label_h5 = hf["label"]

    input_h5.resize([times + 1, img_size, img_size, color_dim])
    input_h5[times : times+1] = input_
    label_h5.resize([times + 1, img_size*scale, img_size*scale, color_dim])
    label_h5[times : times+1] = label_

    hf.close()
    return True

def make_sub_data(data,is_test,path=''):
    times = 0
    scale=3
    img_size=32 
    stride=16
    color_dim=3
    for i in range(len(data)):
        input_, label_, = preprocess(data[i], scale)
        #generate BN
        #input_ = cv2.GaussianBlur(input_,(7,7),1.6,1.6, cv2.BORDER_DEFAULT)
        if len(input_.shape) == 3:
            h, w, c = input_.shape
        else:
            h, w = input_.shape
        # DIV2K images are very large, about 2000*2000, we can randomly extract 96*96 patchs for HR.
        for x in range(0, h * scale - img_size * scale + 1, stride * scale):
            for y in range(0, w * scale - img_size * scale + 1, stride * scale):
                sub_label = label_[x: x + img_size * scale, y: y + img_size * scale]
                
                sub_label = sub_label.reshape([img_size * scale , img_size * scale, color_dim])
                # some patchs do not have color difference, which are bad sample.
                Y = cv2.cvtColor(sub_label, cv2.COLOR_BGR2YCR_CB)
                Y = Y[:, :, 0]
                diff_x = Y[1:, 0:-1] - Y[0:-1, 0:-1]
                diff_y = Y[0:-1, 1:] - Y[0:-1, 0:-1]
                diff_xy = (diff_x**2 + diff_y**2)**0.5
                r_diffxy = float((diff_xy > 10).sum()) / ((img_size*scale)**2) 
                if r_diffxy < 0.1:
                    continue

                sub_label =  sub_label / 255.0

                x_i = int(x / scale)
                y_i = int(y / scale)
                sub_input = input_[x_i: x_i + img_size, y_i: y_i + img_size]
                sub_input = sub_input.reshape([img_size, img_size, color_dim])
                #generate DN
                sub_input = sub_input+np.random.normal(0,30,(img_size,img_size,color_dim))
                sub_input = np.clip(sub_input,0,255) / 255.0


                save_flag = make_data_hf(sub_input, sub_label, times,is_test,path=path)
                if not save_flag:
                    return
                times += 1

        print("image: [%2d], total: [%2d]"%(i, len(data)))

def prepare_data(is_test,path=''):
    if is_test:
        data=glob(path+'/*')
    else:
        data=glob('/scratch/gilbreth/li3120/dataset/DIV2K_train_HR/Raw/*.png')
    return data

def input_prepare(is_test):
    test_files=glob('/scratch/gilbreth/li3120/dataset/DIV2K_train_HR/TestDataSR/*')
    if is_test:
        for test_path in test_files:
            data=prepare_data(is_test,path=test_path)
            make_sub_data(data,is_test,path=test_path)
    else:
        data=prepare_data(is_test)
        make_sub_data(data,is_test)

if __name__=='__main__':
    input_prepare(is_test=True)
