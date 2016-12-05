"""
Data generator, inherit from keras.preprocessing.image.Iterator
follow the design of Minc generator

"""

import numpy as np
import re
import scipy.ndimage as ndi
import keras.backend as K
from keras.preprocessing.image import Iterator
import os





def crop(x, center_x, center_y, ratio=.23, channel_index=0):
    """

    :param x: as nparray (3,x,x)
    :param center_x:
    :param center_y:
    :param ratio:
    :return:
    """
    ratio = max(0,ratio)
    ratio = min(ratio,1)
    assert len(x.shape) == 3 and x.shape[channel_index] in {1, 3}
    r_x = [max(0, center_x - ratio/2), min(1, center_x + ratio/2)]
    r_y = [max(0, center_y - ratio/2), min(1, center_y + ratio/2)]
    if channel_index == 0:
        w = x.shape[1]
        h = x.shape[2]
        return x[:,int(r_x[0]*w):int(r_x[1]*w),int(r_y[0]*h):int(r_y[1]*h)]
    elif channel_index == 2:
        w = x.shape[0]
        h = x.shape[1]
        return x[int(r_x[0] * w):int(r_x[1] * w), int(r_y[0] * h):int(r_y[1] * h), :]
    else:
        raise ValueError("Only support channel as 0 or 2")


def random_rotation(x, rg, row_index=1, col_index=2, channel_index=0,
                    fill_mode='nearest', cval=0.):
    theta = np.pi / 180 * np.random.uniform(-rg, rg)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    return x


def random_shift(x, wrg, hrg, row_index=1, col_index=2, channel_index=0,
                 fill_mode='nearest', cval=0.):
    h, w = x.shape[row_index], x.shape[col_index]
    tx = np.random.uniform(-hrg, hrg) * h
    ty = np.random.uniform(-wrg, wrg) * w
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])

    transform_matrix = translation_matrix  # no need to do offset
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    return x


def random_shear(x, intensity, row_index=1, col_index=2, channel_index=0,
                 fill_mode='nearest', cval=0.):
    shear = np.random.uniform(-intensity, intensity)
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    return x


def random_zoom(x, zoom_range, row_index=1, col_index=2, channel_index=0,
                fill_mode='nearest', cval=0.):
    if len(zoom_range) != 2:
        raise Exception('zoom_range should be a tuple or list of two floats. '
                        'Received arg: ', zoom_range)

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    return x


def random_barrel_transform(x, intensity):
    # TODO
    pass


def random_channel_shift(x, intensity, channel_index=0):
    x = np.rollaxis(x, channel_index, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [np.clip(x_channel + np.random.uniform(-intensity, intensity), min_x, max_x)
                      for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)
    return x


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x, transform_matrix, channel_index=0, fill_mode='nearest', cval=0.):
    x = np.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                      final_offset, order=0, mode=fill_mode, cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)
    return x


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def array_to_img(x, dim_ordering='default', scale=True):
    from PIL import Image
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    if dim_ordering == 'th':
        x = x.transpose(1, 2, 0)
    if scale:
        x += max(-np.min(x), 0)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    if x.shape[2] == 3:
        # RGB
        return Image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        return Image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise Exception('Unsupported channel number: ', x.shape[2])


def img_to_array(img, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    if dim_ordering not in ['th', 'tf']:
        raise Exception('Unknown dim_ordering: ', dim_ordering)
    # image has dim_ordering (height, width, channel)
    x = np.asarray(img, dtype='float32')
    if len(x.shape) == 3:
        if dim_ordering == 'th':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if dim_ordering == 'th':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise Exception('Unsupported image shape: ', x.shape)
    return x


def load_img(path, grayscale=False, target_size=None):
    '''Load an image into PIL format.

    # Arguments
        path: path to image file
        grayscale: boolean
        target_size: None (default to original size)
            or (img_height, img_width)
    '''
    from PIL import Image
    img = Image.open(path)
    if grayscale:
        img = img.convert('L')
    else:  # Ensure 3 channel even when loaded image is grayscale
        img = img.convert('RGB')
    if target_size:
        img = img.resize((target_size[1], target_size[0]))
    return img


def get_binary_label_from_image_path(path, center_x, center_y, threshold=0):
    from PIL import Image
    img = Image.open(path)
    w, h = img.size
    d = img.getpixel((center_x*w, center_y*h))
    if d > threshold:
        return 1
    else:
        return 0

def crop_img(img, x, y, ratio=.23, target_size=None):
    ratio = max(0, ratio)
    ratio = min(ratio, 1)
    r_x = [max(0, x - ratio / 2), min(1, x + ratio / 2)]
    r_y = [max(0, y - ratio / 2), min(1, y + ratio / 2)]
    w, h = img.size
    img = img.crop((int(r_x[0] * w), int(r_y[0] * h), int(r_x[1] * w), int(r_y[1] * h)))

    if target_size:
        img = img.resize((target_size[1], target_size[0]))
    return img


def list_pictures(directory, ext='jpg|jpeg|bmp|png'):
    return [os.path.join(directory, f) for f in sorted(os.listdir(directory))
            if os.path.isfile(os.path.join(directory, f)) and re.match('([\w]+\.(?:' + ext + '))', f)]


class RoadImageIterator(Iterator):
    ## TODO modified the loading structure
    """
    overall logic should be:
        Read the image files
        loop with sliding-window to cover every 16x16 patch in the iamge
        Support scaling (use 16x16 patch as center, take the surrounding as
            input patch, used as a factor,
            ### TODO This is achieved in the crop image blocks

    Structure :
        root/
            training/
                groundtruth/
                    satImage_xxx.png
                images/
                    satImage_xxx.png
            test_set_images/
                test_x/
                    test_x.png

    Make sure the later generated data is following the same structure and naming convention


    """
    def __init__(self, directory, image_data_generator,
                 classes={'non-road':0,'road':1}, ratio=None,
                 img_folder='training', label_folder='groundtruth', org_folder='images',
                 original_img_size=(400,400),
                 stride=(32,32),
                 nb_per_class=10000,
                 target_size=(64, 64), color_mode='rgb',
                 dim_ordering='default',
                 batch_size=32, shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg'):
        """
        Initilize the road iamge loading iterators

        :param directory:               root absolute directory
        :param image_data_generator:    potential generator (for translation and etc)
        :param classes:                 number of class
        :param ratio:                   patch size from original image
        :param img_folder:              image folder under root absolute directory
        :param target_size:             target patch size
        :param color_mode:              color mode
        :param dim_ordering:            Keras dim ordering, default as 'th' for theano
        :param batch_size:
        :param shuffle:                 shuffle the training data
        :param seed:                    random seed
        :param save_to_dir:             directory of saving
        :param save_prefix:             saving prefix
        :param save_format:
        """
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.original_image_size = original_img_size
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.dim_ordering = dim_ordering
        if self.color_mode == 'rgb':
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.classes = classes

        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        # Make it absolute
        self.img_folder = os.path.join(directory, img_folder)
        self.label_folder = os.path.join(self.img_folder, label_folder)
        self.org_folder = os.path.join(self.img_folder, org_folder)

        if ratio is None:
            ratio = float(target_size[0]) / float(original_img_size[0])
        self.ratio = ratio
        self.stride = stride
        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}

        # first, count the number of samples and classes
        self.nb_sample = 0

        self.batch_array = []

        self.nb_class = len(classes)
        self.class_indices = {v: k for k, v in classes.iteritems()}

        # TODO modified loading logic
        dirs = ('groundtruth', 'images')
        self.label_files = []
        self.img_files = []

        for subdir in dirs:
            file_list = []
            subpath = os.path.join(self.img_folder, str(subdir))
            for fname in sorted(os.listdir(subpath)):
                if fname.lower().startswith('._'):
                    continue
                is_valid = False
                for extension in white_list_formats:
                    if fname.lower().endswith('.' + extension):
                        file_list.append(fname)
                        is_valid = True
                        break
                if is_valid:
                    self.nb_sample += 1
            if subdir is 'groundtruth':
                self.img_files = file_list
            else:
                self.label_files = file_list

        assert len(self.label_files) == len(self.img_files)
        self.nb_sample /= 2
        print('Found %d images belonging to %d classes.' % (self.nb_sample, self.nb_class))
        # Store the complete list

        # second, build an index of the images in the different class subfolders
        super(RoadImageIterator, self).__init__(self.nb_sample, batch_size, shuffle, seed)
        # TODO figure out how to generate balanced dataset
        self.nb_per_class = nb_per_class
        self.index_generator = self._flow_index(self.nb_per_class * self.nb_class, batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape)
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        # build batch of labels
        batch_y = np.zeros((len(batch_x), self.nb_class), dtype='float32')
        batch_label = []
        # print(index_array)
        # print(self.batch_array)
        for i, j in enumerate(index_array):
            # get the index
            # print(self.batch_array[j])
            fname = self.batch_array[j][0]
            center_x, center_y = self.batch_array[j][1]
            # print(fname)
            # print(self.img_folder)
            img = load_img(os.path.join(self.org_folder, fname), grayscale=grayscale)

            # ADD MORE LOGIC HERE, LOAD
            img = crop_img(img, center_x, center_y, ratio=self.ratio, target_size=self.target_size)
            x = img_to_array(img, dim_ordering=self.dim_ordering)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
            # update self.batch_classes
            img_label = get_binary_label_from_image_path(os.path.join(self.label_folder, fname), center_x, center_y)
            batch_y[i, img_label] = 1.
            batch_label.append(img_label)

        batch_label_name = []
        # print(self.class_indices)
        for i, label in enumerate(batch_label):
            batch_label_name.append(self.class_indices[int(label)])

        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix +\
                                                                  batch_label_name[i],
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)

                img.save(os.path.join(self.save_to_dir, fname))
        return batch_x, batch_y

    def generatelistfromtxt(self, fname):
        if self.classes is None:
            raise ValueError("classes should be initialized before calling")

        path = fname
        # label, photo_id, x, y
        with(open(path, 'r')) as f:
            file_list = [line.rstrip().split(',') for line in f]
            f.close()
        # load the category and generate the look up table
        classlist = [[] for i in range(self.nb_class)]
        for i, fname, x, y in file_list:
            fn = os.path.join(fname.split('.')[0][-1], fname + ".jpg")
            classlist[int(i)].append([fn, float(x), float(y)])
        return classlist

    def reset(self):
        """
        reset the batch_array and index
        :return:
        """
        self.batch_index = 0
        self.file_indices = []
        self.batch_array = []
        self.batch_classes = []
        # Generate file indices again
        self.file_indices = np.random.permutation((len(self.label_files)))
        # Generate batch_array, for batch_classes, leave it to runtime generation
        valid_patch_per_image = []
        w, h = self.target_size[0]/2, self.target_size[1]/2
        while w < self.original_image_size[0] - self.target_size[0]/2:
            h = self.target_size[1]/2
            while h < self.original_image_size[1] - self.target_size[1]/2:
                valid_patch_per_image.append((float(w) / self.original_image_size[0],
                                              float(h) / self.original_image_size[1]))
                h += self.stride[1]
            w += self.stride[0]
        nb_per_image = len(valid_patch_per_image)
        print('number of batch generated per image is {}'.format(nb_per_image))

        for file_index in self.file_indices:
            for center in valid_patch_per_image:
                self.batch_array.append((self.img_files[file_index], center))
        self.nb_batch_array = len(self.batch_array)

    def _flow_index(self, N, batch_size=32, shuffle=False, seed=None):
        """
        flow for the Road extraction dataset
        Create a random 20,000 patches data set

        :param N:
        :param batch_size:
        :param shuffle:
        :param seed:
        :return:
        """
        # Special flow_index for the balanced training sample generation
        # ensure self.batch_index is 0
        self.reset()
        while 1:
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)
            if self.batch_index == 0:
                index_array = np.arange(N)
                if shuffle:
                    index_array = np.random.permutation(N)
                if N > self.nb_batch_array:
                    index_array %= self.nb_batch_array

            current_index = (self.batch_index * batch_size) % N
            if N >= current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = N - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)