"""
Engine for example standard carring:

Supply with
    model
    input of following type:
        whole dataset as numpy array
        generator
    verbose
    log_file location
    model I/O

"""
from __future__ import absolute_import

import logging

from keras.callbacks import ModelCheckpoint, \
    CSVLogger, LearningRateScheduler, \
    EarlyStopping, ReduceLROnPlateau

from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator, Iterator

import os
import sys


from .utils.io_utils import get_absolute_dir_project, get_weight_path
from .utils.io_utils import cpickle_load, cpickle_save



def getlogfiledir():
    return get_absolute_dir_project('model_saved/log')


def gethistoryfiledir():
    return get_absolute_dir_project('model_saved/history')

def hist_average(historys):
    raise NotImplementedError


class ExampleEngine(object):
    """
    Goal:
        Unified test purpose for secondary image statistics.
        Separate the model creation and testing purpose
        Reduce code duplication.

    Function:
        Give a
            model
            output path for log
            data / data generator
        Implement:
            fit function for data array / generator
            evaluate function
            weights path and load/save
            cross validation for selected parameters


    """
    def __init__(self, data, model, validation=None, test=None,
                 load_weight=True, save_weight=True,
                 lr_decay=False, early_stop=False,
                 save_per_epoch=False,
                 batch_size=128, nb_epoch=100,
                 verbose=2, logfile=None, save_log=True,
                 title='default'):
        """

        Parameters
        ----------
        data    (X, Y) or Generator
        model   Keras.Model
        validation  same as data
        test        same as data
        load_weight True to support loading weights
        save_weight True to support saving weights
        save_per_epoch  True to save the weight after each epochs
        batch_size      # Should not be called.
        nb_epoch        # should not be called
        verbose         verbose same as keras, 0 quiet, 1 detail, 2 brief
        logfile         LogFile location
        save_log        True to save log
        title           Title of all saving files.
        """
        self.model = model
        self.title = title
        self.mode = 0 # 0 for fit ndarray, 1 for fit generator
        if isinstance(data, (list,tuple)):
            assert len(data) == 2
            self.train = data
        elif isinstance(data, (ImageDataGenerator, Iterator)):
            self.train = data
            self.mode = 1
        else:
            raise TypeError('data is not supported {}'.format(data.__class__))
            return

        # Load the validation and test data.
        if self.mode == 0:
            if validation is not None:
                assert len(validation) == 2
            if test is not None:
                assert len(test) == 2

        if self.mode == 1:
            if validation is not None:
                assert isinstance(validation, (Iterator,))
                self.nb_te_sample = validation.nb_sample
            if test is not None:
                assert isinstance(test, (ImageDataGenerator, Iterator))

        self.model_label = ['com', 'gen']

        self.validation = validation
        self.test = test

        self.verbose = verbose
        self.logfile = logfile
        self.log_flag = save_log
        self.default_stdout = sys.stdout
        if logfile is None:
            self.logfile = os.path.join(getlogfiledir(), "{}-{}_{}.log".format(
                self.title, model.name, self.mode))
        else:
            self.logfile = os.path.join(getlogfiledir(), "{}-{}_{}_{}.log".format(
                self.title, model.name, self.mode, logfile))

        if self.log_flag:
            # TODO Modified Logger to logging (Python)
            # self.stdout = Logger(self.logfile)
            self.verbose = 2

        self.nb_epoch = nb_epoch
        self.batch_size = batch_size

        # Set the weights
        self.weight_path = get_weight_path(
            "{}-{}_{}.weights".format(self.title, model.name, self.mode),
            'dataset')
        logging.debug("weights path {}".format(self.weight_path))
        self.save_weight = save_weight
        self.load_weight = load_weight
        self.save_per_epoch = save_per_epoch
        if not os.path.exists(self.weight_path):
            print("weight not found, create a new one or transfer from current weight")
            self.load_weight = False
        self.cbks = []
        if self.save_weight and self.save_per_epoch:
            self.cbks.append(ModelCheckpoint(self.weight_path + ".tmp", verbose=1))

        self.lr_decay = lr_decay
        if self.lr_decay:
            self.cbks.append(ReduceLROnPlateau(min_lr=0.0001, verbose=1))
        self.early_stop = early_stop
        if self.early_stop:
            self.cbks.append(EarlyStopping(patience=20, verbose=1))

    def fit(self, batch_size=32, nb_epoch=100, verbose=2, augmentation=False):
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        if self.load_weight:
            print("Load weights from {}".format(self.weight_path))
            self.model.load_weights(self.weight_path, by_name=True)
        try:
            # Handle ModelCheckPoint
            if self.mode == 0:
                history = self.fit_ndarray(augmentation)
            elif self.mode == 1:
                history = self.fit_generator()
            else:
                history = None
        except (KeyboardInterrupt, SystemExit):
            print("System catch do some thing (like save the model)")
            if self.save_weight:
                self.model.save_weights(self.weight_path + ".tmp", overwrite=True)
                try:
                    history = self.model.history
                    self.save_history(history, tmp=True)
                except Exception:
                    raise Exception
            raise
        self.history = history
        if self.save_weight and self.save_per_epoch:
            # Potentially remove the tmp file.
            os.remove(self.weight_path + '.tmp')
        if self.log_flag:
            sys.stdout.close()
        return history

    def fit_generator(self):
        print("{} fit with generator".format(self.model.name))
        history = self.model.fit_generator(
            self.train, samples_per_epoch=128*200, nb_epoch=self.nb_epoch,
            nb_worker=4,
            validation_data=self.validation, nb_val_samples=self.nb_te_sample,
            verbose=self.verbose,
            callbacks=self.cbks)
        if self.save_weight:
            self.model.save_weights(self.weight_path)
        self.history = history
        return self.history

    def fit_ndarray(self, augmentation=True):
        print('model fitting with ndarray data')
        X_train = self.train[0]
        Y_train = self.train[1]
        if self.test is not None:
            X_test = self.test[0]
            Y_test = self.test[1]

        if self.validation is not None:
            valid = self.validation
        if not augmentation:
            print('Not using data augmentation.')
            hist = self.model.fit(X_train, Y_train,
                                  batch_size=self.batch_size,
                                  nb_epoch=self.nb_epoch,
                                  validation_data=valid,
                                  shuffle=True,
                                  callbacks=self.cbks)
        else:
            print('Using real-time data augmentation.')
            # this will do preprocessing and realtime data augmentation
            datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False)  # randomly flip images

            # compute quantities required for featurewise normalization
            # (std, mean, and principal components if ZCA whitening is applied)
            datagen.fit(X_train)

            # fit the model on the batches generated by datagen.flow()
            hist = self.model.fit_generator(datagen.flow(X_train, Y_train,
                                                         batch_size=self.batch_size),
                                            samples_per_epoch=X_train.shape[0],
                                            nb_epoch=self.nb_epoch,
                                            verbose=self.verbose,
                                            validation_data=valid,
                                            callbacks=self.cbks)

            if self.save_weight:
                self.model.save_weights(self.weight_path)
        return hist

    def plot_result(self, metric='acc', linestyle='-', show=False, dictionary=None):
        if self.history is None:
            return
        history = self.history.history
        if dictionary is not None:
            history = dictionary
        if metric is 'acc':
            train = history['acc']
            valid = history['val_acc']
        elif metric is 'loss':
            train = history['loss']
            valid = history['val_loss']
        else:
            raise RuntimeError("plot only support metric as loss, acc")
        x_factor = range(len(train))
        filename = "{}-{}_{}.png".format(self.title, self.model.name, self.mode)
        from .utils.visualize_util import plot_train_test
        from _tkinter import TclError
        try:
            plot_train_test(train, valid, x_factor=x_factor, show=show,
                            xlabel='epoch', ylabel=metric,
                            linestyle=linestyle,
                            filename=filename, plot_type=0)
            self.save_history(history)
        except TclError:
            print("Catch the Tcl Error, save the history accordingly")
            self.save_history(history)
            return

    def save_history(self, history, tmp=False):
        from keras.callbacks import History
        if isinstance(history, History):
            history = history.history
        import numpy as np
        filename = "{}-{}_{}.history".format(self.title, self.model.name, np.random.randint(1e4))
        if tmp:
            filename = 'tmp_' + filename
        if history is None:
            return
        logging.debug("compress with gz")
        dir = gethistoryfiledir()
        if not os.path.exists(dir):
            os.mkdir(dir)
        filename = os.path.join(dir, filename)
        filename = cpickle_save(data=history, output_file=filename)
        return filename

    @staticmethod
    def load_history(filename):
        logging.debug("load history from {}".format(filename))
        hist = cpickle_load(filename)
        if isinstance(hist, dict):
            return hist
        else:
            raise ValueError("Should read a dict term")

    @staticmethod
    def load_history_from_log(filename):
        logging.debug('Load history from {}'.format(filename))
        with open(filename, 'r') as f:
            lines = [line.rstrip() for line in f]
            hist = dict()
            hist['loss'] = []
            hist['val_loss'] = []
            hist['acc'] = []
            hist['val_acc'] = []
            for line in lines:
                if not line.startswith('Epoch'):
                    for i, patch in enumerate(line.replace(' ', '').split('-')[1:]):
                        n, val = patch.split(":")
                        hist[n].append(float(val))
        return hist


class CVEngine(ExampleEngine):

    def __init__(self, models, data, validation=None, test=None,
                 load_weight=True, save_weight=True,
                 save_per_epoch=False,
                 verbose=2, logfile=None, save_log=True,
                 title='cvalid_default'):
        super(CVEngine, self).__init__(data, None, validation=validation, test=test,
                                       load_weight=load_weight, save_weight=save_weight,
                                       save_per_epoch=save_per_epoch, verbose=verbose, logfile=logfile,
                                       save_log=save_log, title=title)
        # Override the parameters
        self.__dict__.update(locals())

        self.mode = 0  # 0 for fit nd-array, 1 for fit generator
        if isinstance(data, (list, tuple)):
            assert len(data) == 2
            self.train = data
        elif isinstance(data, (ImageDataGenerator, DirectoryIterator)):
            self.train = data
            self.mode = 1

        # Load the validation and test data.
        if self.mode == 0:
            if validation is not None:
                assert len(validation) == 2
            if test is not None:
                assert len(test) == 2

        if self.mode == 1:
            if validation is not None:
                assert isinstance(validation, (DirectoryIterator))
                self.nb_te_sample = validation.nb_sample
            if test is not None:
                assert isinstance(test, (ImageDataGenerator, DirectoryIterator))

        # Override part
        if logfile is not None:
            self.logfile = os.path.join(getlogfiledir(), "{}-{}_{}.log".format(
                self.title, 'crossvalid', self.mode))
        else:
            self.logfile = os.path.join(getlogfiledir(), "{}-{}_{}_{}.log".format(
                self.title, "crossvalid", self.mode, logfile))

        # Set the weights
        self.weight_path = get_weight_path(
            "{}-{}_{}.cv.weights".format(self.title, "cross_valid", self.mode),
            'dataset')
        logging.debug("weights path {}".format(self.weight_path))

        if not os.path.exists(self.weight_path):
            print("weight not found, create a new one or transfer from current weight")
            self.load_weight = False

        self.models = models
        self.model_label = ['com', 'gen']

    def cross_validation(self, models=[], k_fold=1, nb_epoch=50, batch_size=16):
        """

        Parameters
        ----------
        models:  [keras.Model]
                    if models is [] or None, the cross_validation won't work
        k_fold:      k_fold validation, default = 1
        nb_epoch:
        batch_size:

        Returns
        -------
            result: dict    dictionary as format {'model1.name': history}
        """
        if models is None or []:
            models = self.models
        result = dict()
        for ind, model in enumerate(models):
            hist_list = []
            for k in range(k_fold):
                raise NotImplementedError




