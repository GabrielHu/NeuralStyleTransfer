'''
This is the file for functions of image manipulation(import, export, scale).
'''
import os
import scipy.misc
import numpy as np


def save_img(out_path, img):
    '''
    function for saving image.

    Args:
        out_path: a string. Place for image saving.
        img: a 3-D tensor. The image for saving.
    '''
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(out_path, img)


def scale_img(path, scale):
    '''
    function for scaling image.

    Args:
        path: a string. Place for importing unscaled images.
        scale: a scalar. Multiple for the image to expand.

    Return:
        target: a 3-D tensor. A scaled image.
    '''
    scale = float(scale)
    x, y, channel = scipy.misc.imread(path, mode='RGB').shape
    scale = float(scale)
    new_shape = (int(x * scale), int(y * scale), channel)
    target = get_img(path, img_size=new_shape)
    return target


def get_img(src, img_size=False):
    '''
    function for getting image.

    Args:
        src: a string. Place for importing images.
        img_size: a tuple. Setting the size of imported image. Default is False

    Return:
        img: a 3-D tensor. A tensor of image.
    '''
    img = scipy.misc.imread(src, mode='RGB')
    if not (len(img.shape) == 3 and img.shape[2] == 3):
        img = np.dstack((img, img, img))
    if img_size != False:
        img = scipy.misc.imresize(img, img_size)
    return img


def list_files(in_path):
    '''
    function for listing files in the path.

    Args:
        in_path: a string. The place for scanning files.

    Return:
        files: a list. A list of files in the path.
    '''
    files = []
    for (_, _, filenames) in os.walk(in_path):
        files.extend(filenames)
        break
    return files
