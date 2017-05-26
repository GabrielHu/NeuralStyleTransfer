'''
This is the keras version of http://arxiv.org/abs/1610.07629, a paper about multi-style transfer.
Comparing to J.Johnson et al., this paper replaces transpose convolution layer with nearest-neighbor 
interpolation.
'''

import numpy as np
from keras.applications import vgg16
from scipy.misc import imsave
from keras.layers.convolutional import Conv2D, UpSampling2D
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


def gram_matrix(img_matrix):
    '''Calculate gram matrix.

    Args: 
        img_matrix: a 4-D tensor. a processed 4-D tensor.

    Return:
        a 2-D matrix. The gram matrix of 4-D tensor.
    '''
    assert K.ndim(img_matrix) == 3
    features = K.batch_flatten(K.permute_dimensions(img_matrix, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


def style_loss(style, combination):
    '''Calculate style loss.

    Args: 
        style: a 4-D tensor. a processed 4-D tensor of style picture.
        combination: a 4-D tensor. a processed 4-D tensor of combination picture.

    Return:
        a scalar. The discrepancy of style.
    '''
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = 256 * 256
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))


def content_loss(base, combination):
    '''Calculate style loss. Designed to transfer style. Designed to maintain the "content" of the base image in the generated image

    Args: 
        base: a 4-D tensor. a processed 4-D tensor of content picture.
        combination: a 4-D tensor. a processed 4-D tensor of combination picture.

    Return:
        a scalar. The discrepancy of content.
    '''
    return K.sum(K.square(combination - base))


def total_variation_loss(x):
    '''Calculate style loss. Designed to keep the generated image locally coherent

    Args: 
        x: a 4-D tensor. a processed 4-D tensor of combination picture.

    Return:
        a scalar. The penalty of local incoherence
    '''
    assert K.ndim(x) == 4
    a = K.square(x[:, :255, :255, :] -
                 x[:, 1:, :255, :])
    b = K.square(x[:, :255, :255, :] -
                 x[:, :255, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


def loss(combination_image):
    '''The loss function to minimize

    Args:
        input_tensor: a 4-D tensor. concatenation of images.

    Return:
        a scalar. 
    '''

    outputs_dict = dict([(layer.name, layer.output) for layer in pre_model.layers])
    loss = K.variable(0.)
    layer_features = outputs_dict['block4_conv2']

    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss += content_loss(base_image_features,
                         combination_features)

    feature_layers = ['block1_conv1', 'block2_conv1',
                      'block3_conv1', 'block4_conv1',
                      'block5_conv1']
    for layer_name in feature_layers:
        layer_features = outputs_dict[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features)
        loss += (0.1 / len(feature_layers)) * sl
    loss += 0.05 * total_variation_loss(combination_image)
    return loss


def _residual_block(input_shape):
    '''a residual block.

    Args: 
        input_shape: a list. indication of tensor's size.

    Return:
        a 4-D tensor. The output after operation.
    '''
    input = Input(shape=input_shape)
    output = Conv2D(filters=128, kernel_size=(3, 3),
                    padding='same', activation='relu')(input)
    output = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(input)
    output = add([input, output])
    return Model(inputs=input, outputs=output)


def _upsampling_block(input_shape, filters):
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
    output = UpSampling2D(input)
    output = Conv2D(filters=filters, kernel_size=(
        3, 3), padding='same', activation='relu')
    return Model(inputs=input, outputs=output)



base_img = K.concatenate()
combination_img = K.variable(preprocess_image(np.random.uniform(0,255,K.shape(base_img)))


pre_model = vgg16.VGG16(input_tensor=input_tensor,
                    weights='imagenet', include_top=False)






model = Sequential()
model.add(Conv2D(32, (9, 9), padding='same', activation='relu',
                 input_shape=(None, 256, 256, None)))
model.add(Conv2D(64, (3, 3), stride=2, padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), stride=2, padding='same', activation='relu'))
model.add(_residual_block((None, 256, 256, None)))
model.add(_residual_block((None, 256, 256, None)))
model.add(_residual_block((None, 256, 256, None)))
model.add(_residual_block((None, 256, 256, None)))
model.add(_residual_block((None, 256, 256, None)))
model.add(_upsampling_block((None, 256, 256, None), 64))
model.add(_upsampling_block((None, 256, 256, None), 32))
model.add(Conv2D(3, (9, 9), padding='same', activation='sigmoid'))


model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit()
