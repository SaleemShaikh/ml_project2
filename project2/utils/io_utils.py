from __future__ import absolute_import

import gzip
import os
import cPickle
import sys


def cpickle_load(filename):
    """
    Support loading from files directly.
    Proved to be same speed, but large overhead.
    Switch to load generator.
    :param filename:
    :return:
    """
    if os.path.exists(filename):
        if filename.endswith(".gz"):
            f = gzip.open(filename, 'rb')
        else:
            f = open(filename, 'rb')
        # f = open(path, 'rb')
        if sys.version_info < (3,):
            data = cPickle.load(f)
        else:
            data = cPickle.load(f, encoding="bytes")
        f.close()
        return data  # (data
    else:
        print("File not found under location {}".format(filename))
        return None


def cpickle_save(data, output_file, ftype='gz'):
    """
    Save the self.label, self.image before normalization, to Pickle file
    To the specific output directory.
    :param output_file:
    :param ftype
    :return:
    """

    if ftype is 'gz' or ftype is 'gzip':
        print("compress with gz")
        output_file = output_file + "." + ftype
        f = gzip.open(output_file, 'wb')
    elif ftype is '':
        f = open(output_file, 'wb')
    else:
        raise ValueError("Only support type as gz or '' ")
    cPickle.dump(data, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    return output_file


def get_project_dir():
    """
    Get the current project directory
    :return:
    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    current_dir = os.path.dirname(current_dir)
    current_dir = os.path.dirname(current_dir)
    current_dir = os.path.dirname(current_dir)
    current_dir = os.path.dirname(current_dir)
    return current_dir


def get_dataset_dir():
    """
    Get current dataset directory
    :return:
    """
    # current_dir = get_project_dir()
    # return os.path.join(current_dir, 'data')
    datadir_base = os.path.expanduser(os.path.join('~', '.keras'))
    return os.path.join(datadir_base, 'datasets')


def get_road_image_dir():
    return os.path.join(get_dataset_dir(), 'prml2')


def get_absolute_dir_project(filepath):
    """
    Get the absolute dir path under project folder.
    :param dir_name: given dir name
    :return: the absolute path.
    """
    path = os.path.join(get_project_dir(), filepath)
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        # os.mkdir(dir)
        return None
    return path


def get_weight_path(filename, dir='project'):
    if dir is 'project':
        path = get_absolute_dir_project('model_saved')
        return os.path.join(path, filename)
    elif dir is 'dataset':
        dir_base = os.path.expanduser(os.path.join('~', '.keras'))
        dir = os.path.join(dir_base, 'models')
        if not os.path.exists(dir):
            os.mkdir(dir)
        return os.path.join(dir, filename)


def get_plot_path(filename=None, dir='project', mkdir=False):
    if dir is 'project':
        path = get_absolute_dir_project('model_saved/plots')
        if not os.path.exists(path):
            os.mkdir(path)
    elif dir is 'dataset':
        dir_base = os.path.expanduser(os.path.join('~', '.keras'))
        path = os.path.join(dir_base, 'plots')
        if not os.path.exists(path):
            os.mkdir(path)
    else:
        raise ValueError("only support project, dataset as dir, input {}".format(dir))
    if filename is None:
        return path
    file_path = os.path.join(path, filename)
    if mkdir and not os.path.exists(file_path):
        os.mkdir(file_path)
    return file_path
