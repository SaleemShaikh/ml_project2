"""
Translated the implementation mentioned in the paper

Ref:
    ETH Master thesis

    FCN paper PAMI 2016


"""
from build.lib.keras.engine import Input
from keras.models import Model
from keras.layers import Convolution2D, Deconvolution2D, Flatten
from keras.layers import MaxPooling2D, Dropout, Dense, Activation



# Definition of blocks
def conv_block2(input_tensor, nb_filters, rows, cols, stage):
    """
    Convolutional block with 2 layer
    :param input_tensor: input tensor as Keras.tensor, such as Input, output from previous layer
    :param nb_filters: number of feature maps
    :param rows: conv_kernal row size
    :param cols: cov_kernal col size
    :param stage: indicator for model names
    :return:
    """
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


def leNet(input_shape, nb_class=2, mode=0, boarder_mode='same', activation='relu'):
    """
    Modified version of LeNet5, change the kernal size from 5,5 to 3,3,
    :param input_shape:
    :param mode:
    :param boarder_mode:
    :return:
    """
    img_input = Input(input_shape)
    if mode == 0:
        x = Convolution2D(6, 3, 3, boarder_mode=boarder_mode)(img_input)
        x = MaxPooling2D(pool_size=(2,2))(x)
        x = Activation(activation)(x)

        x = Convolution2D(16, 3, 3, boarder_mode=boarder_mode)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Activation(activation)(x)
        x = Dropout(0.5)(x)

        x = Convolution2D(120, 1, 1, border_mode=boarder_mode)(x)
        x = Flatten()(x)
        x = Dense(84)(x)
        x = Activation(activation)(x)
        x = Dense(nb_class)(x)

        model = Model(img_input, x, name='LeNet5')

        return model


