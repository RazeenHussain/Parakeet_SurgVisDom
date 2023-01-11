def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.simplefilter("ignore", category=PendingDeprecationWarning)


import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import argparse
from glob import glob
import datetime
from PIL import Image

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

batch_size=5
img_rows = 540
img_cols = 960
img_channels = 3

def Rconvolution_block(x, num_filters, size, strides=(1,1), padding='same', activation=True):
    x1 = Conv2D(num_filters, size, strides=strides, kernel_initializer='he_normal', padding=padding)(x)
    x1 = BatchNormalization()(x1)
    if activation == True:
        x1 = Activation('relu')(x1)
    x = Conv2D(num_filters, size, strides=strides, kernel_initializer='he_normal', padding=padding)(x1)
    x = BatchNormalization()(x)
    if activation == True:
        x = Activation('relu')(x)
    x = Add()([x, x1])
    return x

def convolution_block(x_input, num_filters, size, strides=(1,1), padding='same', activation=True):
    x = Conv2D(num_filters, size, strides=strides, kernel_initializer='he_normal', padding=padding)(x_input)
    if activation == True:
        x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(num_filters, size, strides=strides, kernel_initializer='he_normal', padding=padding)(x)
    x = se_block(x, num_filters, ratio=32)
    x_input = Conv2D(num_filters, kernel_size=(1,1), padding='same')(x_input)
    x = Add()([x_input, x])
    if activation == True:
        x = Activation('relu')(x)
    return x

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
def cross_dice_loss(y_true, y_pred, smooth=1.):
	'''
	Compute the cost funciton (loss functions):
	cost = binary_crossentropy + dice 

	'''
	binary_cross = K.binary_crossentropy(y_true, y_pred)
	dice_lss = dice_coef_loss(y_true, y_pred)
	return dice_lss + binary_cross
def dice_metric(y_true, y_pred, smooth=1.):
	'''
	y_ture: target 
	y_pred: predicted image from model
	dice_coef =  2*(X n Y) / ( sum(X) + sum(Y))
	'''
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	y_pred_f = K.cast(K.greater(y_pred_f, 0.5), 'float32')
	intersection = K.sum(y_true_f*y_pred_f)
	return (2.*intersection + smooth ) /  (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def se_block(input, filters, ratio=32):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor
    Returns: a keras tensor
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    # filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // 16, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)
 
    x = multiply([init, se])
    return x

def  Unet(num_filters=32):
    #input images of size [imgs_rows, img_cols, img_channels]
    strides_here =(2,2)
    inputs = Input((img_rows, img_cols,  img_channels), name ='input')

    #First convolution and pooling
    c1 = convolution_block(inputs, num_filters, size=(3,3))
    pool1 = MaxPooling2D(pool_size=(3,4))(c1)
    #pool1 = Conv2D(num_filters, kernel_size=(3,3), strides=(3,4), kernel_initializer='he_normal', padding='same')(c1)

    #Second convolution and pooling
    c2 = convolution_block(pool1, num_filters*2, size=(3,3))
    pool2 = MaxPooling2D(pool_size=(3,4))(c2)
    #pool2 = Conv2D(num_filters*2, kernel_size=(3,3), strides=(3,4), kernel_initializer='he_normal', padding='same')(c2)

    c3 = convolution_block(pool2, num_filters*4, size=(3,3))
    pool3 = MaxPooling2D(pool_size=(2,2))(c3)
    #pool3 = Conv2D(num_filters*4, kernel_size=(3,3), strides=strides_here, kernel_initializer='he_normal', padding='same')(c3)

    c4 = convolution_block(pool3, num_filters*8, size=(3,3))
    pool4 = MaxPooling2D(pool_size=(2,2))(c4)
    #pool4 = Conv2D(num_filters*8, kernel_size=(3,3), strides=strides_here, kernel_initializer='he_normal', padding='same')(c4)

    #Firth convolution and pooling
    c6 = convolution_block(pool4, num_filters*16, size=(3,3))
    c6 = convolution_block(c6, num_filters*32, size=(3,3))
    c6=Dropout(0.5)(c6)

    u7 = Conv2DTranspose(num_filters*8, kernel_size=(2,2), strides=(2,2), padding='same')(c6)
    u7 = concatenate([u7, c4])
    c7 = convolution_block(u7, num_filters*8, size=(3,3))

    u8 = Conv2DTranspose(num_filters*4, kernel_size=(2,2), strides=(2,2), padding='same')(c7)
    u8 = concatenate([u8, c3])
    c8 = convolution_block(u8, num_filters*4, size=(3,3))

    u9 = Conv2DTranspose(num_filters*2, kernel_size=(2,2), strides=(3,4), padding='same')(c8)
    u9 = concatenate([u9, c2])
    c9 = convolution_block(u9, num_filters*2, size=(3,3))

    # upsamplying with concatination
    u10 = Conv2DTranspose(num_filters, kernel_size=(2,2), strides=(3,4), padding='same')(c9)
    u10 = concatenate([u10, c1])
    c11 = convolution_block(u10, num_filters, size=(3,3))

    # Mask_output
    output_mask = Conv2D(1, (1, 1), activation='sigmoid', name='output_mask')(c11)

    model = Model(inputs =[inputs], outputs=[output_mask])
    model.compile(optimizer = Adam(lr=1e-4), loss = {'output_mask':cross_dice_loss}, metrics = {'output_mask':[dice_coef]}) # 'binary_crossentropy', 'accuracy' or mtp.dice_coef_loss, mtp.dice_coef
    print(model.summary())
    return model

def load_data(input_path, output_path):
	
    print(input_path)
    print(output_path)
    print('loading data...')

    training_img_paths=sorted(glob(os.path.join(input_path,"images/*.png")))
    training_msk_paths=sorted(glob(os.path.join(input_path,"fore/*.png")))
    print(len(training_img_paths))
    print(len(training_msk_paths))

    # Parameters
    params = {'dim': (540,960),
            'batch_size': 5,
            'n_classes': 1,
            'n_channels': 3,
            'shuffle': True}

    # Datasets
    partition = list(range(5983))
    
    partition_train = partition[500:5600]

    partition_validation = partition[0:500]
    partition_validation.extend(partition[5600:])

    print(len(partition_train))
    print(len(partition_validation))

    # Generators
    training_generator = DataGenerator(partition_train, training_img_paths, training_msk_paths, **params)
    validation_generator = DataGenerator(partition_validation, training_img_paths, training_msk_paths, **params)

    # Design model
    model = Unet()
    model.load_weights('model_Unet.h5')
    print(model.summary())
    '''    
    # Train model on dataset
    #model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        epochs=150)
   
    # serialize model to JSON
    model_json = model.to_json()
    with open("model_Unet.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_Unet.h5")
    print("Saved model to disk")
    '''

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', help='Input file main folder.')
    parser.add_argument('--output_path', help='Output file main folder.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--task', help='task to perform.', 
                        choices=['test', 'train'], default='train')
    args = parser.parse_args()

    if args.task == 'train':
    	assert args.load is None
    	if not os.path.exists(args.input_path):
    		print("Could not find input folder")
    		exit()
    	if not os.path.exists(args.output_path):
    		print("Could not find output folder")
    		exit()
    	load_data(args.input_path, args.output_path)


    	#start_training(args.input_path, args.output_path)

