'''
This is the keras version of http://arxiv.org/abs/1610.07629, a paper about multi-style transfer.
Comparing to J.Johnson et al., this paper replaces transpose convolution layer with nearest-neighbor 
interpolation.
'''

import numpy as np
from keras.applications import vgg16
from scipy.misc import imsave
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, add, Input
from keras.models import Sequential, Model
from keras.engine.topology import Layer
from keras.preprocessing.image import img_to_array, load_img
from keras import backend as K


def preprocess_img(img_path):
    '''Change image tensor to fit VGG-network.

    Args: 
        img_path: a string. location of unprocessed image.

    Return:
        a 4-D tensor. Input of VGG-network.
    '''
    img = load_img(img_path, target_size=(256, 256))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return vgg16.preprocess_input(img)


def deprocess_img(img):
    '''Change VGG tensor to image matrix.

    Args: 
        img: a 4-D tensor. a processed 4-D tensor.

    Return:
        a 3-D tensor. The tensor of generated image.
    '''
    img = img.reshape((256, 256, 3))
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astype('uint8')
    return img


def _residual_block(input_shape):
    '''a residual block.

    Args: 
        input_shape: a list. indication of tensor's size.

    Return:
        a 4-D tensor. The output after operation.
    '''
    input = Input(shape=input_shape)
    output = Conv2D(128, (3, 3))(input)
    output = BatchNormalization(axis=-1)(output)
    output = Activation('relu')(output)
    output = Conv2D(128, (3, 3))(output)
    output = BatchNormalization(axis=-1)(output)
    output = add([input, output])
    return Model(inputs=input, outputs=output)


def _conv_block(input_shape, filters, kernel_size, strides=1):
    '''a convolution block.

    Args: 
        input_shape: a list. indication of tensor's size.
        filters: number of filters.
        kernel_size: the size of kernals.
        strides: stride of filters. default is strides=1

    Return:
        a 4-D tensor. The output after operation.
    '''
    input = Input(shape=input_shape)
    output = Conv2D(filters=filters, kernel_size=kernel_size,
                    strides=strides, padding='same')(input)
    output = BatchNormalization(axis=-1)(output)
    output = Activation('relu')(output)
    return Model(inputs=input, outputs=output)


def _conv_trans_block(input_shape, filters, kernel_size, strides=1):
    '''a transpose convolution block.

    Args: 
        input_shape: a list. indication of tensor's size.
        filters: number of filters.
        kernel_size: the size of kernals.
        strides: stride of filters. default is strides=1

    Return:
        a 4-D tensor. The output after operation.
    '''
    input = Input(shape=input_shape)
    output = Conv2DTranspose(filters=filters, kernel_size=kernel_size,
                             strides=strides, padding='same')(input)
    output = BatchNormalization(axis=-1)(output)
    output = Activation('relu')(output)
    return Model(inputs=input, outputs=output)


model = Sequential()
model.add(_conv_block((1, 256, 256, 3), filters=32, kernel_size=9))
model.add(_conv_block((1, None, None, None),
                      filters=64, kernel_size=3, strides=2))
model.add(_conv_block((1, None, None, None),
                      filters=128, kernel_size=3, strides=2))
model.add(_residual_block(None))
model.add(_residual_block(None))
model.add(_residual_block(None))
model.add(_residual_block(None))
model.add(_residual_block(None))
model.add(_conv_trans_block(None, filters=64, kernel_size=3, strides=2))
model.add(_conv_trans_block(None, filters=32, kernel_size=3, strides=2))
model.add(_conv_block(None, filters=32, kernel_size=9))


model.compile(optimizer='adam',)
