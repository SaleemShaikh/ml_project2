from __future__ import absolute_import

from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

from project2.cnn_v1 import leNet
from project2.utils.data_utils import RoadImageIterator
from project2.utils.io_utils import get_road_image_dir
from project2.exampe_engine import ExampleEngine


IMG_HEIGHT = 16
IMG_WIDTH = 16
IMG_CHANNEL = 3
STRIDE = (16,16)
ORIGINAL_TRAIN_IMAGE_SIZE = (400, 400)

NB_EPOCH = 100


def run_routine1():
    """
    Routine 1: Baseline training process to obtain a classifier for LeNet structure

    Returns
    -------
    history
    """
    generator = ImageDataGenerator(rescale=1./255)
    train_img_iterator = RoadImageIterator(get_road_image_dir(), generator,
                                           stride=STRIDE,
                                           original_img_size=ORIGINAL_TRAIN_IMAGE_SIZE,
                                           target_size=(IMG_WIDTH, IMG_HEIGHT),
                                           batch_size=128)
    input_shape = (IMG_CHANNEL, IMG_WIDTH, IMG_HEIGHT)
    model = leNet(input_shape)
    opt = optimizers.rmsprop()
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    engine = ExampleEngine(train_img_iterator, model, load_weight=False, lr_decay=True, title='LeNet_baseline',
                           nb_epoch=NB_EPOCH)

    history = engine.fit_generator()
    engine.plot_result()
    engine.save_history(history, tmp=True)


if __name__ == '__main__':
    run_routine1()