def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.simplefilter("ignore", category=PendingDeprecationWarning)


import numpy as np
import os
import sys
#import matplotlib.pyplot as plt
import argparse
from glob import glob
import datetime
from PIL import Image
from keras.models import model_from_json

import tensorflow as tf
from keras import backend as K
from keras.preprocessing import image
from keras.models import Model 
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, AveragePooling2D, Conv2DTranspose, BatchNormalization, Activation, Dropout , Add 
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, add, RNN, Flatten 
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

from my_classes import DataGenerator

# Define the parameters of the data to process
K.set_image_data_format('channels_last')

def dice_coef(y_true, y_pred, smooth=1.):
    '''
    y_ture: target 
    y_pred: predicted image from model
    dice_coef =  2*(X n Y) / ( sum(X) + sum(Y))
    '''
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f*y_pred_f)
    return (2.*intersection + smooth ) /  (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    '''
    Compute the cost funciton (loss function) 
     cost = 1-dice_coef 
    '''
    return 1-dice_coef(y_true, y_pred)


def load_data(input_path, output_path, load_model):
	
    print(input_path)
    print(output_path)
    print(load_model)
    print('loading data...')

    testing_img_paths=sorted(glob(os.path.join(input_path,"*/*/*/*/raw.png")))
    testing_msk_paths=sorted(glob(os.path.join(input_path,"*/*/*/*/")))

    print(len(testing_img_paths))
    print(len(testing_msk_paths))

    # Parameters

    # Train model on dataset
    # load json and create model
    json_file = open(load_model+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(load_model+'.h5')
    print("Loaded model from disk")
    #
    print(model.summary())

    # evaluate loaded model on test data
    #model.compile(optimizer = Adam(lr=1e-4), loss = {'output_mask':dice_coef_loss}, metrics = {'output_mask':'accuracy'}) # 'binary_crossentropy', 'accuracy' or mtp.dice_coef_loss, mtp.dice_coef
    
    for i in range(0,len(testing_img_paths)):
        print('Processing Frame: ',i)
        print(testing_img_paths[i])
        img=load_img(os.path.join(testing_img_paths[i]))
        img=img_to_array(img)
        if (np.max(img)>1.1):
            img = img / 255.
        msk=model.predict(img[np.newaxis,:])
        msk=msk[0,:,:,0]
        mskFS = Image.fromarray((msk).astype(np.uint8))
        save_path=testing_msk_paths[i]
        print(save_path[5:])
        if not os.path.exists(output_path+save_path[6:]):
            os.makedirs(output_path+save_path[6:])
        mskFS.save(output_path+save_path[6:]+'output.png')

    print('Saved all output frames')
    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', help='Input file main folder.')
    parser.add_argument('--output_path', help='Output file main folder.')
    parser.add_argument('--load_model', help='load model')
    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        print("Could not find input folder")
        exit()
    
    load_data(args.input_path, args.output_path, args.load_model)

