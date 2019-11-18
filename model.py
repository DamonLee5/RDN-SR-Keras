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

class RDN():
    
    def __init__(self,load=0,rfi=0,lfi=0):
        np.random.seed(0)
        self.channels=3
        self.scale=2
        self.D=20
        self.C=6
        self.G=32
        self.G0=64
        self.bl_height = 32                 # Low resolution image height
        self.bl_width = 32                  # Low resolution image width
        self.bl_shape = (self.bl_height, self.bl_width, self.channels)
        self.sh_height = self.bl_height*self.scale   # High resolution image height
        self.sh_width = self.bl_width *self.scale    # High resolution image width
        self.sh_shape = (self.sh_height, self.sh_width, self.channels)
        self.lr=0.0001
        self.result_file_ind=rfi
        self.load_file_ind=lfi  
        optimizer = Adam(self.lr, 0.5)
        # Configure data loader
        self.resultfile_name ='RDN_x%d_%s'%(self.scale,self.result_file_ind)
        self.dataset_name = 'train_x%d'%(self.scale)
        self.valset_name ='val'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                img_crop=(self.bl_height, self.bl_width))
        if load:
            self.load('/scratch/gilbreth/li3120/RDB/saved_model/RDB_best_%s.h5'%(self.load_file_ind))
        else:
            self.model=self.build_model()
        self.model.save('/scratch/gilbreth/li3120/RDB/saved_model/RDB_best_tt.h5') 
           
        self.model.summary()
        self.model.compile(loss=['mae'],optimizer=optimizer)
        #self.model.save('/scratch/gilbreth/li3120/RDB/saved_model/a.h5')
    def RDBs(self,input_layer):
        rdb_concat=[]
        rdb_in=input_layer
        for i in range(1,self.D+1):
            x=rdb_in
            for j in range(1,self.C+1):
                tmp=Conv2D(self.G,kernel_size=3, strides=1, padding='same',kernel_initializer=RandomNormal(seed=2+i*self.C+j,stddev=0.01),activation='relu')(x)
                x=Concatenate()([x, tmp])
            x=Conv2D(self.G,kernel_size=1, strides=1, padding='same',kernel_initializer=RandomNormal(seed=2+i*self.C+j,stddev=0.01))(x)
            rdb_in=Add()([x,rdb_in]) 
            rdb_concat.append(rdb_in)
        return Concatenate()(rdb_concat)
#     def _phase_shift (I):
#         bsize, a, b, c = I.get_shape().as_list()
        
#         X = tf.reshape(I, [-1, a, b, self.scale, self.scale])
#         X = tf.split(X, a, 1)  # a, [bsize, b, scale, scale]
#         X = tf.concat([tf.squeeze(x,axis=1) for x in X], 2)  # bsize, b, a*scale, scale
#         X = tf.split(X, b, 1)  # b, [bsize, a*scale, scale]
#         X = tf.concat([tf.squeeze(x,axis=1) for x in X], 2)  # bsize, a*r, b*r
#         return tf.reshape(X, (-1, a*self.scale, b*self.scale, 1))
    def UPN(self,input_layer):
        x=Conv2D(self.G0,kernel_size=5,strides=1, padding='same',kernel_initializer=RandomNormal(seed=1,stddev=0.01),activation='relu')(input_layer)
        x=Conv2D(self.G,kernel_size=3,strides=1, padding='same',kernel_initializer=RandomNormal(seed=2,stddev=0.01),activation='relu')(x)
        x=Conv2D(self.channels*self.scale*self.scale,kernel_size=3,strides=1, padding='same',kernel_initializer=RandomNormal(seed=3,stddev=np.sqrt(2/9/32)))(x)
#         def _phase_shift (I,scale):
#             bsize, a, b, c = I.get_shape().as_list()
#             X = tf.reshape(I, [-1, a, b, scale, scale])
#             X = tf.split(X, a, 1)  # a, [bsize, b, scale, scale]
#             X = tf.concat([tf.squeeze(x,axis=1) for x in X], 2)  # bsize, b, a*scale, scale
#             X = tf.split(X, b, 1)  # b, [bsize, a*scale, scale]
#             X = tf.concat([tf.squeeze(x,axis=1) for x in X], 2)  # bsize, a*r, b*r
#             return tf.reshape(X, (-1, a*scale, b*scale, 1))
        def PS(ll,scale):
            import tensorflow as tf
            llc = tf.split(ll, 3, 3)
            ttt=[]
            for I in llc:
                bsize, a, b, c = I.get_shape().as_list()
                X = tf.reshape(I, [-1, a, b, scale, scale])
                X = tf.split(X, a, 1)  # a, [bsize, b, scale, scale]
                X = tf.concat([tf.squeeze(x,axis=1) for x in X], 2)  # bsize, b, a*scale, scale
                X = tf.split(X, b, 1)  # b, [bsize, a*scale, scale]
                X = tf.concat([tf.squeeze(x,axis=1) for x in X], 2)   
                ttt.append(tf.reshape(X, (-1, a*scale, b*scale, 1)))
            #ll=tf.concat([_phase_shift (x,scale) for x in llc], 3)
            ll=tf.concat(ttt,3)
            return ll
        #def PS_shape(input_shape,scale):
        #    return (input_shape[0],input_shape[1]*scale,input_shape[2]*scale,3)
        x=Lambda(PS,arguments={'scale':self.scale})(x)
        return x
    def bm(self):
        d0 = Input(shape=self.bl_shape)
        F_1 = Conv2D(self.G0, kernel_size=3, strides=1, padding='same',kernel_initializer=RandomNormal(seed=1,stddev=0.01))(d0)
        return Model(d0,F_1)
    def build_model(self):
        d0 = Input(shape=self.bl_shape)
        # SFENET
        F_1 = Conv2D(self.G0, kernel_size=3, strides=1, padding='same',kernel_initializer=RandomNormal(seed=1,stddev=0.01))(d0)
        F0  = Conv2D(self.G, kernel_size=3, strides=1, padding='same',kernel_initializer=RandomNormal(seed=2,stddev=0.01))(F_1)
        
        #RDBs
        FD  = self.RDBs(F0)
        
        #DFF
        FGF1=Conv2D(self.G0,kernel_size=1,strides=1, padding='same',kernel_initializer=RandomNormal(seed=1,stddev=0.01))(FD)
        FGF2=Conv2D(self.G0,kernel_size=3,strides=1, padding='same',kernel_initializer=RandomNormal(seed=2,stddev=0.01))(FGF1)
        FDF=Add()([FGF2,F_1])
        #UPN
        FU=self.UPN(FDF)
        IHR=Conv2D(self.channels,kernel_size=3,strides=1, padding='same',kernel_initializer=RandomNormal(seed=1,stddev=np.sqrt(2.0/27)))(FU)
        return Model(d0,IHR)
    def train(self, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()
        gloss_min=10000
        for epoch in range(epochs):
            print('epoch:%d'%(epoch))
            for batch_id, (imgs_bl, imgs_sh) in enumerate(self.data_loader.load_data(batch_size)):
                g_loss = self.model.train_on_batch(imgs_bl, imgs_sh)
                
                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print ("%d time: %s, loss: %f" % (epoch*1000+batch_id, elapsed_time,g_loss))

                # If at save interval => save generated image samples
                if batch_id % sample_interval == 0:
                    print("Test at epoch %d"%(epoch*1000+batch_id));
                    gloss_min=self.sample(epoch*1000+batch_id,path='/scratch/gilbreth/li3120/dataset/DIV2K_train_HR/Train/val_x%d.h5'%(self.scale),gloss_min=gloss_min)
                  

        self.model.save('/scratch/gilbreth/li3120/RDB/saved_model/RDB_final_%s.h5'%(self.result_file_ind)) 
    
    def sample(self, epoch,path,gloss_min):
        os.makedirs('images/%s' % self.resultfile_name, exist_ok=True)
        r, c = 1, 2
        bs=16
        imgs_bl, imgs_sh = self.data_loader.load_sample_data(cal_mse=True,mse_path=path)
        b_size=len(imgs_bl);
        if b_size>2000:
            imgs_bl=imgs_bl[:2000]
            imgs_sh=imgs_sh[:2000]
            b_size=2000
        fake_sh = self.model.predict(imgs_bl,batch_size=16)
        ggloss =self.model.evaluate(imgs_bl, imgs_sh,batch_size=1)      
        print('Loss:%f'%(ggloss))

        # Save generated images and the originals
        titles = ['Generated', 'GroundTrue']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for row in range(r):
            for col, image in enumerate([fake_sh, imgs_sh]):

                pos=axs[col].imshow(np.squeeze(image[row]))
                axs[col].set_title(titles[col])
                axs[col].axis('off')
                fig.colorbar(pos, ax=axs[col])
            cnt += 1
        fig.savefig("images/%s/%d.png" % (self.resultfile_name, epoch))
        plt.close()

        # Save Back Projection images for comparison
        i=np.random.randint(0,b_size);
        fig = plt.figure()
        pos=plt.imshow(np.squeeze(imgs_sh[i]))
        fig.colorbar(pos)
        fig.savefig('images/%s/%d_groundtrue%d.png' % (self.resultfile_name, epoch, i))
        pos=plt.imshow(np.squeeze(fake_sh[i]))
        fig.savefig('images/%s/%d_processed%d.png' % (self.resultfile_name, epoch, i))
        plt.close()
        if ggloss<gloss_min:
            self.model.save('/scratch/gilbreth/li3120/RDB/saved_model/RDB_best_%s.h5'%(self.result_file_ind))          
            return ggloss
        else:
            return gloss_min
    def load(self,path):
        custom_obj = {}
        custom_obj['tf']=tf
        self.model=load_model(path,custom_objects=custom_obj)
        print('loading path:%s'%(path))
        
    def predict_process(self,path):
        ll=path.split("/")
        print(ll)          
        #os.makedirs('/scratch/gilbreth/li3120/RDB/Test_Result/%s_%d'%(ll[-2],self.result_file_ind),exist_ok=True)
        
        imgs_bl,imgs_sh=self.data_loader.load_sample_data(cal_mse=True,mse_path=path)
        fake_sh=self.model.predict(imgs_bl,batch_size=16)

        f= h5py.File('/scratch/gilbreth/li3120/RDB/Test_Result/%s_%d.h5'%(ll[-2],self.result_file_ind), "w")
        Recset=f.create_dataset("rec", data=fake_sh)
        GTset=f.create_dataset("gt", data=imgs_sh)
        f.close()
        
        
    
if __name__=='__main__':
    
    rdn=RDN(load=0,rfi=17,lfi=0)
    rdn.train(epochs=200,batch_size=16, sample_interval=200)
        
