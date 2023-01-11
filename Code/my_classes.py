import numpy as np
import keras
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import os
import sys
import albumentations as albu
#import matplotlib.pyplot as plt

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, paths_X, paths_Y, batch_size=32, dim=(540,960), n_channels=3, n_classes=1, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.paths_X = paths_X
        self.paths_Y = paths_Y
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        #self.boolx=0
        #self.id=0


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        #print(X.shape)
        
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __random_transform(self, img, masks):
        composition = albu.Compose([
            albu.HorizontalFlip(),
            albu.VerticalFlip(),
            albu.ShiftScaleRotate(rotate_limit=45, shift_limit=0.2)
        ])
        composed = composition(image=img, mask=masks)
        aug_img = composed['image']
        aug_masks = composed['mask']
        return aug_img, aug_masks

    def __augment_batch(self, img_batch, masks_batch):
        for i in range(img_batch.shape[0]):
            img_batch[i, ], masks_batch[i, ] = self.__random_transform(
                img_batch[i, ], masks_batch[i, ])

        return img_batch, masks_batch

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim, self.n_channels))
        y = np.empty((self.batch_size, self.dim, 1), dtype=int)
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            img=load_img(os.path.join(self.paths_X[ID]))
            msk=load_img(os.path.join(self.paths_Y[ID]), grayscale=True)
            img=img_to_array(img)
            msk=img_to_array(msk)

            if (np.max(img)>1.1):
                img = img / 255.

            X[i,]=img
            y[i,]=msk
        '''
        if (self.boolx==0):
            plt.imsave('output/org'+str(self.id)+'.png',X[0,]/255.)
            plt.imsave('output/mor'+str(self.id)+'.png',np.squeeze(y[0,]))
            print(np.max(X[0,]))
            print(np.max(y[0,]))
        '''
        X,y = self.__augment_batch(X,y)
        '''
        if (self.boolx==0):
            print(np.max(X[0,]))
            print(np.max(y[0,]))
            plt.imsave('output/img'+str(self.id)+'.png',X[0,]/255.)
            plt.imsave('output/msk'+str(self.id)+'.png',np.squeeze(y[0,]))
            self.id=self.id+1
            if self.id>10:
                self.boolx=1
        
        #imgS = Image.fromarray((X[0,]).astype(np.uint8))
        #imgS.save('training/img0.png')
        #mskFS = Image.fromarray((y[0,]).astype(np.uint8))
        #mskFS.save('training/msk0.png')
        '''
        return X, y
