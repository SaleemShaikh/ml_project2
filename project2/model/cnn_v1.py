"""
Translated the implementation mentioned in the paper

Ref:
    ETH Master thesis

    FCN paper PAMI 2016


"""
from keras.engine import Input
from keras.models import Model
from keras.layers import Convolution2D, Deconvolution2D, Flatten
from keras.layers import MaxPooling2D, Dropout, Dense, Activation



# Definition of blocks


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




