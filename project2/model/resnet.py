""" This module implements the ResNet34 CNN archiecture. The implementation is vase don the implementation
provided within examples of Keras library: https://github.com/fchollet/keras/blob/master/keras/applications/resnet50.py.
"""

# Keras
from keras.layers import Convolution2D, Activation, Dense, Flatten, Input, merge, BatchNormalization, AveragePooling2D
from keras.models import Model
from keras import backend as K

def identity_block_original(input_tensor, kernel_size, filters, stage, block):
    '''The identity_block is the block that has no conv layer at shortcut

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    nb_filter1, nb_filter2 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, kernel_size, kernel_size, name=conv_name_base + '2a', border_mode='same')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size,
                      border_mode='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = merge([x, input_tensor], mode='sum')
    x = Activation('relu')(x)
    return x


def conv_block_original(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    '''conv_block is the block that has a conv layer at shortcut

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    '''
    nb_filter1, nb_filter2 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, kernel_size, kernel_size, subsample=strides,
                      name=conv_name_base + '2a', border_mode="same")(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size, border_mode='same',
                      name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    shortcut = Convolution2D(nb_filter2, 1, 1, subsample=strides,border_mode='same',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = merge([x, shortcut], mode='sum')
    x = Activation('relu')(x)
    return x

def resnet34orig(inputShape, nb_class=10):
    """ ResNet for input images (patches) of size 16x16. The output
    layer consists of 1 neuron with logistic activation function (1 = signal, 0 = background).

    Parameters
    ----------
    inputShape : 3-tuple of int32
        Shape of input data disregarding the batch axis (e.g. (channels, height, width) for
        theano backend).

    nb_class : int32
        Number of output classes.

    Returns
    -------
    model : keras.Model
        Resulting ResNet model.

    """

    # Check for correct input size.
    if len(inputShape) != 3:
        raise('Input shape must be 3-tuple.')

    img_input = Input(shape=inputShape)

    bn_axis = 1

    x = Convolution2D(16, 3, 3, name='conv1', border_mode='same')(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)

    x = conv_block_original(x, 3, [16, 16], stage=2, block='a', strides=(1, 1))
    x = identity_block_original(x, 3, [16, 16], stage=2, block='b')
    x = identity_block_original(x, 3, [16, 16], stage=2, block='c')

    x = conv_block_original(x, 3, [32, 32], stage=3, block='a', strides=(2, 2))
    x = identity_block_original(x, 3, [32, 32], stage=3, block='b')
    x = identity_block_original(x, 3, [32, 32], stage=3, block='c')
    x = identity_block_original(x, 3, [32, 32], stage=3, block='d')

    x = conv_block_original(x, 3, [64, 64], stage=4, block='a', strides=(2, 2))
    x = identity_block_original(x, 3, [64, 64], stage=4, block='b')
    x = identity_block_original(x, 3, [64, 64], stage=4, block='c')
    x = identity_block_original(x, 3, [64, 64], stage=4, block='d')
    x = identity_block_original(x, 3, [64, 64], stage=4, block='e')
    x = identity_block_original(x, 3, [64, 64], stage=4, block='f')

    x = conv_block_original(x, 3, [128, 128], stage=5, block='a', strides=(2, 2))
    x = identity_block_original(x, 3, [128, 128], stage=5, block='b')
    x = identity_block_original(x, 3, [128, 128], stage=5, block='c')

    x = AveragePooling2D((2, 2), name='avg_pool')(x)
    x = Flatten()(x)
    x = Dense(nb_class, activation='softmax', name='fc')(x)

    model = Model(img_input, x, name='resnet')

    return model
