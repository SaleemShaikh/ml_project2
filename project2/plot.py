from keras.utils.visualize_util import plot
from project2.utils.io_utils import get_plot_path_with_subdir

from project2.cnn_v1 import leNet

import logging

def plot_model(model):
    path = get_plot_path_with_subdir(model.name + '.png', subdir='models', dir='project')
    logging.info("Plot to path {}".format(path))
    plot(model, to_file=path)

def plot_lenet():
    model = leNet((3, 16, 16))
    plot_model(model)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    plot_lenet()

