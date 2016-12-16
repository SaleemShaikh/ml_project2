import pytest
import keras
from keras.preprocessing.image import ImageDataGenerator
from project2.utils.data_utils import RoadImageIterator, DirectoryImageLabelIterator, \
    concatenate_images, make_img_overlay, array_to_img, img_to_array
from project2.utils.io_utils import *

import numpy as np


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
    itr = DirectoryImageLabelIterator(dir, gen, batch_size=4,
                                      dim_ordering='tf',
                                      image_only=True,
                                      rescale=True,
                                      data_folder='massachuttes',
                                      image_folder='sat', label_folder='label',
                                      target_size=(224,224), stride=(128,128),
                                      save_to_dir=path, save_prefix='test')
    # batch_x, batch_y = itr.next()
    # assert batch_x.shape[:2] == batch_y.shape[:2]
    batch_x = itr.next()
    assert batch_x.shape[1] == batch_x.shape[2]


def test_ratio_DirectoryImageLabelGenerator():
    dir = get_road_image_dir()
    gen = None
    path = get_plot_path('img_patch', 'dataset')
    img_w = 200
    original_size = 400
    index_lim = original_size / img_w
    itr = DirectoryImageLabelIterator(dir, gen, batch_size=8,
                                      dim_ordering='tf',
                                      image_only=False,
                                      rescale=False,
                                      data_folder='massachuttes',
                                      image_folder='sat', label_folder='label',
                                      ratio=0.5,
                                      shuffle=False,
                                      original_img_size=(original_size, original_size),
                                      target_size=(img_w, img_w), stride=(img_w, img_w),
                                      save_to_dir=path, save_prefix='test'
                                      )
    batch_x, batch_y = itr.next()
    result_image = np.zeros(shape=(400,400,3))
    result_label = np.zeros(shape=(400,400,1))
    for i in range(index_lim):
        for j in range(index_lim):
            result_image[img_w * i:img_w*(i+1), img_w*j:img_w*(j+1),:] = batch_x[index_lim*j + i]
            result_label[img_w*i:img_w*(i+1), img_w*j:img_w*(j+1),:] = batch_y[index_lim*j + i]
    result_image = img_to_array(result_image, 'tf')
    result_label = img_to_array(result_label, 'tf')
    result_label = np.squeeze(result_label)
    result = make_img_overlay(result_image, result_label)
    # result.save(path + '/result.png')

    res_img, res_lab = DirectoryImageLabelIterator.concatenate_batches(
        (batch_x, batch_y), (index_lim, index_lim), nb_image_per_batch=4)
    res_lab = np.squeeze(res_lab)
    res = make_img_overlay(res_img[0], res_lab[0])
    res.save(path+'/result2.png')
    result.save(path+'/result.png')

if __name__ == '__main__':
    # pytest.main([__file__])
    test_ratio_DirectoryImageLabelGenerator()