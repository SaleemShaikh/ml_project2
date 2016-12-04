import pytest

from keras.preprocessing.image import ImageDataGenerator
from project2.utils.data_utils import RoadImageIterator
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
    path = get_plot_path('img_patch', 'dataset', mkdir=True)
    itr = RoadImageIterator(dir, gen, save_to_dir=path, save_prefix='test')
    batch_x, batch_y = itr.next()
    print(type(batch_x))


if __name__ == '__main__':
    pytest.main([__file__])