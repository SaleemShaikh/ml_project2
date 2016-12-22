"""
The implementation of adjusted LeNet5 architecture (3 convolutional layers, different filter sizes).
"""
from keras.engine import Input
from keras.models import Model
from keras.layers import Convolution2D, Deconvolution2D, Flatten
from keras.layers import MaxPooling2D, Dropout, Dense, Activation


def leNet(input_shape, nb_class=2, boarder_mode='same', activation='relu'):
    """
    Modified version of LeNet5, change the kernal size from 5,5 to 3,3,

    Parameters
    ----------

    inputShape : 3-tuple of int32
        Shape of input data disregarding the batch axis (e.g. (channels, height, width) for
        theano backend).

    boarder_mode : str
        Theano style border mode. Either 'same' or 'valid'.

    nb_class : int32
        Number of output classes.

    activation : str
        Activation function, theano style.

    :return:
    """
    img_input = Input(input_shape)

    x = Convolution2D(6, 3, 3, border_mode=boarder_mode)(img_input)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Activation(activation)(x)

    x = Convolution2D(16, 3, 3, border_mode=boarder_mode)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Activation(activation)(x)
    x = Dropout(0.5)(x)

    x = Convolution2D(120, 1, 1, border_mode=boarder_mode)(x)
    x = Flatten()(x)
    x = Dense(84)(x)
    x = Activation(activation)(x)
    x = Dense(nb_class)(x)
    x = Activation('softmax')(x)

    model = Model(img_input, x, name='LeNet5')

    return model
