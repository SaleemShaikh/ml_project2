"""
Translated the implementation mentioned in the paper

Ref:
    ETH Master thesis

    FCN paper PAMI 2016


"""

from keras.models import Model
from keras.layers import Convolution2D, Deconvolution2D
from keras.layers import MaxPooling2D, Dropout, Dense, Activation


def conv_block2(input_tensor, nb_filters, rows, cols, stage):
    name_conv = 'Conv_' + str(stage)
    name_acti = 'ReLU_' + str(stage)
    name_pool = 'Pool_' + str(stage)
    x = Convolution2D(nb_filter=nb_filters,
                      nb_row=rows, nb_col=cols,
                      border_mode='same',
                      name=name_conv + '_1')(input_tensor)
    x = Activation('relu', name=name_acti + '_1')(x)
    x = Convolution2D(nb_filter=nb_filters,
                      nb_row=rows, nb_col=cols,
                      border_mode='same',
                      name=name_conv + '_2')(x)
    x = Activation('relu', name=name_acti + '_2')(x)
    x = MaxPooling2D(pool_size=(2,2), name=name_pool)(x)
    return x


def conv_block3(input_tensor, nb_filters, rows, cols, stage):
    name_conv = 'Conv_' + str(stage)
    name_acti = 'ReLU_' + str(stage)
    name_pool = 'Pool_' + str(stage)
    x = Convolution2D(nb_filter=nb_filters,
                      nb_row=rows, nb_col=cols,
                      border_mode='same',
                      name=name_conv + '_1')(input_tensor)
    x = Activation('relu', name=name_acti + '_1')(x)
    x = Convolution2D(nb_filter=nb_filters,
                      nb_row=rows, nb_col=cols,
                      border_mode='same',
                      name=name_conv + '_2')(x)
    x = Activation('relu', name=name_acti + '_2')(x)
    x = Convolution2D(nb_filter=nb_filters,
                      nb_row=rows, nb_col=cols,
                      border_mode='same',
                      name=name_conv + '_3')(x)
    x = Activation('relu', name=name_acti + '_3')(x)
    x = MaxPooling2D(pool_size=(2,2), name=name_pool)(x)
    return x

