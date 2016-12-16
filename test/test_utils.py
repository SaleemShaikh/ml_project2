import pytest
import keras
from keras.preprocessing.image import ImageDataGenerator
from project2.utils.data_utils import RoadImageIterator, DirectoryImageLabelIterator
from project2.utils.io_utils import *


def test_getfolder():
    get_dataset_dir()
    get_project_dir()
    get_weight_path('tmp')
    get_plot_path('tmp')
    get_absolute_dir_project('a')


def test_RoadImageIterator():
    dir = get_road_image_dir()
    gen = ImageDataGenerator(rescale=1./255)
    path = get_plot_path('img_patch', 'dataset')

    IMG_HEIGHT = 16
    IMG_WIDTH = 16
    IMG_CHANNEL = 3
    STRIDE = (16, 16)
    ORIGINAL_TRAIN_IMAGE_SIZE = (400, 400)

    NB_EPOCH = 100
    generator = ImageDataGenerator(rescale=1. / 255)
    for mode in range(3):
        train_img_iterator = RoadImageIterator(get_road_image_dir(), generator,
                                               stride=STRIDE,
                                               original_img_size=ORIGINAL_TRAIN_IMAGE_SIZE,
                                               target_size=(IMG_WIDTH, IMG_HEIGHT),
                                               batch_size=32,
                                               preload=mode,
                                               save_to_dir=path,
                                               save_prefix='test')
        batch_x, batch_y = train_img_iterator.next()


def test_DirectoryImageLabelIterator():
    dir = get_road_image_dir()
    gen = None
    path = get_plot_path('img_patch', 'dataset')
    itr = DirectoryImageLabelIterator(dir, gen, batch_size=2,
                                      dim_ordering='tf',
                                      rescale=True,
                                      data_folder='massachuttes',
                                      image_folder='sat', label_folder='label',
                                      target_size=(224,224), stride=(128,128),
                                      save_to_dir=path, save_prefix='test')
    batch_x, batch_y = itr.next()
    assert batch_x.shape[:2] == batch_y.shape[:2]


if __name__ == '__main__':
    pytest.main([__file__])