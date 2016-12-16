#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import matplotlib.image as mpimg
import re
import glob
from project2.utils.data_utils import img_to_array, load_img
from project2.tf_fcn.utils import save_image
from project2.utils.data_utils import DirectoryImageLabelIterator

foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch

# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0


def mask_to_submission_strings(image_filename):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, *image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn))


def collect_and_concat_into_images(path, nb_patch_per_image, target_size=None, dim_ordering='tf', prefix='pred'):
    """
    Collect the file and concatenate patches into image

    Parameters
    ----------
    path: str
    prefix

    Returns
    -------

    """
    flist = glob.glob(os.path.join(path, prefix + "*.png"))
    img_list = []
    for fname in flist:
        img = load_img(fname, target_size=target_size)
        img_list.append(img_to_array(img, 'tf'))

    res_list = DirectoryImageLabelIterator.concatenate_batches((img_list,), (3,3),
                                                               nb_image_per_batch=nb_patch_per_image)
    return res_list


def create_concat_test():
    """
    Create concatenated test files
    Returns
    -------

    """
    path = '/home/kyu/Dropbox/git/ml_project2/fcn4s_visual/plot_finetune_5000_test'
    save_path = os.path.join(path, 'complete_test_label')
    img_list = collect_and_concat_into_images(path, nb_patch_per_image=9, target_size=(200, 200))
    for ind, img in enumerate(img_list[0]):
        save_image(img, save_path, 'pred_complete_{}'.format(str(ind)))


if __name__ == '__main__':
    # submission_filename = 'dummy_submission.csv'
    # image_filenames = []
    # for i in range(1, 51):
    #     image_filename = 'training/groundtruth/satImage_' + '%.3d' % i + '.png'
    #     print image_filename
    #     image_filenames.append(image_filename)
    # masks_to_submission(submission_filename, *image_filenames)
    create_concat_test()