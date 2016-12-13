

import keras.backend as K
from keras.engine import Input
from keras.engine import Model


from keras.layers import Convolution2D, MaxPooling2D, Deconvolution2D, ZeroPadding2D, UpSampling2D, Reshape, Activation, \
    Dropout, SpatialDropout2D


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


def fcn_32s(input_shape, nb_class):
    """
    Construct Fully Convolutional Network based on paper

    This is the first row of Figure3

    This would use keras.models.Graph to construct the graph directly

    References:
     Fully Convolutional Networks for Semantic Segmentation
     https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf

    Parameters
    ----------
    input_shape : tuple     (img_channel, img_width, img_height)
    nb_class : int          number of class to be classified per pixel

    Returns
    -------
    keras Model, without compiling
    """
    #
    # model = Graph()
    #
    # model.add_input(name='input', input_shape=(None,)+input_shape)
    #
    # model.add_node(ZeroPadding2D(2), name='input_padding', input='input')
    # model.add_node(Convolution2D(nb_filter=16, nb_row=3, nb_col=3, border_mode='same',
    #                              activation='relu'),
    #                name='conv1', input='input_padding')
    # model.add_node(MaxPooling2D(pool_size=(2,2), strides=(1,1,)),
    #                name='pool1', input='conv1')


    img_input = Input(input_shape)
    # x = ZeroPadding2D()(img_input)
    x = conv_block2(img_input, 16, 3, 3, stage=1)
    x = conv_block2(x, 32, 3, 3, stage=2)
    x = conv_block3(x, 48, 3, 3, stage=3)
    x = conv_block3(x, 48, 3, 3, stage=4)
    x = conv_block3(x, 48, 3, 3, stage=5)

    x = Convolution2D(64, 3, 3, activation='relu', name='conv_stage_6')(x)
    x = SpatialDropout2D(0.5)(x)

    x = Convolution2D(128, 3, 3, activation='relu', name='conv_stage_7')(x)
    x = SpatialDropout2D(0.5)(x)

    x = Convolution2D(3, 1, 1, activation='relu', name='fconv1')(x)
    x = Deconvolution2D(3, 3, 3, output_shape=(None,) + input_shape, activation='relu', name='deconv1')(x)
    x = Convolution2D(1, 1, 1, name='predictions')(x)
    x = Reshape(input_shape[1:])(x)

    x = Activation('softmax')(x)

    model = Model(img_input, x, name='FCN_32s')
    return model


