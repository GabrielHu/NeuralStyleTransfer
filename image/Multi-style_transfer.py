'''
This is the keras version of http://arxiv.org/abs/1610.07629, a paper about multi-style transfer.
Comparing to J.Johnson et al., this paper replaces transpose convolution layer with nearest-neighbor
interpolation.
'''
import numpy as np
import utils
from keras.applications import vgg16
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers import add, Input
from keras.models import Sequential, Model
from keras import backend as K


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
    output = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(output)
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
    output = UpSampling2D(size=(2, 2))(input)
    output = Conv2D(filters=filters, kernel_size=(
        3, 3), padding='same', activation='relu')(output)
    return Model(inputs=input, outputs=output)


def loss(content_img, combination_img):
    '''The loss function to minimize

    Args:
        input_tensor: a 4-D tensor. concatenation of images.

    Return:
        a scalar.
    '''
    style_path = r'C:\Users\Renjiu Hu\Pictures\training data\style\\'
    files = utils.list_files(style_path)
    style_label = [utils.get_img(
        style_path + img).astype('float64') for img in files]
    style_list = [K.variable(vgg16.preprocess_input(np.expand_dims(
        utils.get_img(style_path + img).astype('float64'), axis=0))) for img in files]
    style_img = K.concatenate(style_list, axis=0)

    content_model = vgg16.VGG16(input_tensor=content_img,
                                weights='imagenet', include_top=False)
    outputs_dict_content = dict([(layer.name, layer.output)
                                 for layer in content_model.layers])
    style_model = vgg16.VGG16(input_tensor=style_img,
                              weights='imagenet', include_top=False)
    outputs_dict_style = dict([(layer.name, layer.output)
                               for layer in style_model.layers])
    combination_model = vgg16.VGG16(input_tensor=combination_img,
                                    weights='imagenet', include_top=False)
    outputs_dict_combination = dict([(layer.name, layer.output)
                                     for layer in combination_model.layers])

    loss = []
    layer_features_content = outputs_dict_content['block4_conv2']
    layer_features_combination = outputs_dict_combination['block4_conv2']
    for i in range(np.shape(K.eval(combination_img))[0]):
        loss.append(content_loss(layer_features_content[i, :, :, :],
                                 layer_features_combination[i, :, :, :]))

    feature_layers = ['block1_conv1', 'block2_conv1',
                      'block3_conv1', 'block4_conv1',
                      'block5_conv1']
    for layer_name in feature_layers:
        layer_features_style = outputs_dict_style[layer_name]
        layer_features_combination = outputs_dict_combination[layer_name]
        style_reference_features = layer_features_style[3, :, :, :]
        for i in range(np.shape(K.eval(combination_img))[0]):
            sl = style_loss(style_reference_features,
                            layer_features_combination[i, :, :, :])
            loss[i] += (0.1 / len(feature_layers)) * sl
    loss += 0.05 * \
        total_variation_loss(combination_img) * np.ones(np.shape(loss))
    return K.variable(loss)


content_path = r'C:\Users\Renjiu Hu\Pictures\training data\content\\'
files = utils.list_files(content_path)
x_label = [utils.get_img(content_path + img).astype('float64')
           for img in files]
content_list = [K.variable(vgg16.preprocess_input(np.expand_dims(
    img, axis=0))) for img in x_label]
content_img = K.concatenate(content_list, axis=0)

y_label = [np.random.uniform(0, 255, np.shape(x_label[0]))
           for i in range(len(x_label))]
#combination_img = K.placeholder((1, 256, 256, 3))
combination_img = K.concatenate([K.variable(vgg16.preprocess_input(
    np.expand_dims(img, axis=0))) for img in y_label], axis=0)

model = Sequential()
model.add(Conv2D(32, (9, 9), padding='same', activation='relu',
                 input_shape=(256, 256, 3)))
model.add(Conv2D(64, (3, 3), strides=2, padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), strides=2, padding='same',
                 activation='relu', name='conv3'))
model.add(_residual_block((256, 256, 128)))
model.add(_residual_block((256, 256, 128)))
model.add(_residual_block((256, 256, 128)))
model.add(_residual_block((256, 256, 128)))
model.add(_residual_block((256, 256, 128)))
model.add(_upsampling_block((256, 256, 128), 64))
model.add(_upsampling_block((256, 256, 64), 32))
model.add(Conv2D(3, (9, 9), padding='same', activation='sigmoid'))


for i in range(10):
    model.compile(optimizer='adam', loss='loss',
                  metrics=['accuracy'])
    model.fit(content_img, combination_img, epoch=2, batch=2)
    print("2 epochs finish")
    combination_img = model.predict(combination_img, batch_size=2)
    print("next turn starts")

model.save('styleTrans.h5')