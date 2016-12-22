""" This module implements the variants of original AlexNet described in paper Alex Krizhevsky - ImageNet Classification
with Deep Convolutional Neural Networks.

The implementation is based on https://github.com/heuritech/convnets-keras and it is adjusted for needs
of project 2 - PCML 2016.
"""

# Keras
from keras.layers import Convolution2D, Activation, MaxPooling2D, Dense, Flatten, Input, ZeroPadding2D, merge, Dropout
from keras.layers.core import Lambda
from keras.models import Model
from keras import backend as K


def crosschannelnormalization(alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
    """
    This is the function used for cross channel normalization in the original
    Alexnet. Implementation taken from
    https://github.com/heuritech/convnets-keras/blob/master/convnetskeras/customlayers.py.
    """
    def f(X):
        b, ch, r, c = X.shape
        half = n // 2
        square = K.square(X)
        extra_channels = K.spatial_2d_padding(K.permute_dimensions(square, (0,2,3,1))
                                              , (0,half))
        extra_channels = K.permute_dimensions(extra_channels, (0,3,1,2))
        scale = k
        for i in range(n):
            scale += alpha * extra_channels[:,i:i+ch,:,:]
        scale = scale ** beta
        return X / scale

    return Lambda(f, output_shape=lambda input_shape:input_shape, **kwargs)


def splittensor(axis=1, ratio_split=1, id_split=0,**kwargs):
    """ Splits the tensor into two consecutive parts given `ratio_split`. Implementation
    taken from https://github.com/heuritech/convnets-keras/blob/master/convnetskeras/customlayers.py.
    """
    def f(X):
        div = X.shape[axis] // ratio_split

        if axis == 0:
            output =  X[id_split*div:(id_split+1)*div,:,:,:]
        elif axis == 1:
            output =  X[:, id_split*div:(id_split+1)*div, :, :]
        elif axis == 2:
            output = X[:,:,id_split*div:(id_split+1)*div,:]
        elif axis == 3:
            output = X[:,:,:,id_split*div:(id_split+1)*div]
        else:
            raise ValueError("This axis is not possible")

        return output

    def g(input_shape):
        output_shape=list(input_shape)
        output_shape[axis] = output_shape[axis] // ratio_split
        return tuple(output_shape)

    return Lambda(f,output_shape=lambda input_shape : g(input_shape), **kwargs)


def alexnet_v1(inputShape):
    """ This variant of AlexNet is meant for input images (patches) of size 16x16. The output
    layer consists of 1 neuron with logistic activation function (1 = signal, 0 = background).

    Parameters
    ----------
    inputShape : 3-tuple of int32
        Shape of input data disregarding the batch axis (e.g. (channels, height, width) for
        theano backend).

    Returns
    -------
    model : keras.Model
        Resulting AlexNet model.

    """

    # Check for correct input size.
    if len(inputShape) != 3:
        raise('Input shape must be 3-tuple.')

    inputs = Input(shape=inputShape)

    conv_1 = Convolution2D(96, 11, 11, activation='relu', name='conv_1', border_mode='same')(inputs)
    conv_2 = crosschannelnormalization(name="convpool_1")(conv_1)

    conv_2 = merge([Convolution2D(128, 5, 5, activation="relu", name='conv_2_' + str(i + 1), border_mode='same')(
                        splittensor(ratio_split=2, id_split=i)(conv_2)
                    ) for i in range(2)], mode='concat', concat_axis=1, name="conv_2")
    conv_3 = crosschannelnormalization()(conv_2)

    conv_3 = Convolution2D(384, 3, 3, activation='relu', name='conv_3', border_mode='same')(conv_3)

    conv_4 = merge([Convolution2D(192, 3, 3, activation="relu", name='conv_4_' + str(i + 1), border_mode='same')(
                        splittensor(ratio_split=2, id_split=i)(conv_3)
                    ) for i in range(2)], mode='concat', concat_axis=1, name="conv_4")

    conv_5 = merge([Convolution2D(128, 3, 3, activation="relu", name='conv_5_' + str(i + 1), border_mode='same')(
                        splittensor(ratio_split=2, id_split=i)(conv_4)
                    ) for i in range(2)], mode='concat', concat_axis=1, name="conv_5")

    dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name="convpool_5")(conv_5)
    dense_1 = Flatten(name="flatten")(dense_1)
    dense_1 = Dense(4096, activation='relu', name='dense_1')(dense_1)
    dense_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(4096, activation='relu', name='dense_2')(dense_2)
    dense_3 = Dropout(0.5)(dense_2)
    dense_3 = Dense(2, name='dense_3')(dense_3)

    # prediction = Dense(1, activation='sigmoid', name='dense_3')(dense_3)
    prediction = Activation('softmax', name='softmax')(dense_3)

    model = Model(input=inputs, output=prediction, name='AlexNet_v1')

    return model
