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
from project2.utils.data_utils import DirectoryImageLabelIterator, make_img_overlay, concatenate_batches
import tensorflow as tf

foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
patch_size = 16

# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0


def normalize_image_by_patch(image_filename, patch_size):
    """
    Normalize the image by patch_size, given the patch to label threshold = predefined in
    this python file.

    Parameters
    ----------
    image_filename: str         absolute path to image file
    patch_size  : int           patch size

    Returns
    -------
    numpy.array                 wish shape same as the read image file.
    """
    _, fn = os.path.split(image_filename)
    im = mpimg.imread(image_filename)
    new_im = normalize_image_array_by_patch(im, patch_size)
    return new_im


def normalize_image_array_by_patch(image_array, patch_size):
    """
    Normalize the image array by given patch size.

    Parameters
    ----------
    image_array : numpy.array   numpy.array
    patch_size : int            patch size given

    Returns
    -------
    numpy.array                 shape like image-array, and normalized
    """
    im = image_array
    new_im = np.zeros_like(im)
    for j in range(0, im.shape[0], patch_size):
        for i in range(0, im.shape[1], patch_size):
            patch = im[i: i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            new_im[i:i + patch_size, j: j + patch_size] = label
    return new_im


def mask_to_submission_strings(image_filename):
    """Reads a single image and outputs the strings that should go into the submission file
    Make python generator and send into writelines

    Update 2016.12.18
        Revise the image_filename logic to support absolute path
        Revise to support both path or image array directly
    """
    if isinstance(image_filename, str):
        _, fn = os.path.split(image_filename)
        img_number = int(re.search(r"\d+", fn).group(0))
        im = mpimg.imread(image_filename)
    else:
        im = image_filename
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, *image_filenames):
    """
    Converts images into a submission file
    """
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f_list = ('{}\n'.format(s) for s in mask_to_submission_strings(fn))
            f.writelines(f_list)


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


def create_concat_test_from_images(label_list, image_list, save_path, save_normalized=False, save_overlay=False):
    """
    Create concatenated test files from given image list
    Parameters
    ----------
    label_list
    image_list
    save_path
    save_normalized
    save_overlay

    Returns
    -------

    """

    for ind, (inp_img, img) in enumerate(zip(image_list, label_list)):
        if save_overlay:
            if save_normalized:
                img_norm = normalize_image_array_by_patch(img, patch_size)
                save_image(img_norm, save_path, 'normal_test_{}'.format(str(ind + 1)))
                overlay_img = make_img_overlay(inp_img, img_norm, pixel_depth=255)
            else:
                overlay_img = make_img_overlay(inp_img, img, pixel_depth=255)
            overlay_img.save(os.path.join(save_path, 'overlay_input_{}.png'.format(str(ind+1))))
        if not save_overlay:
            save_image(inp_img, save_path, 'input_{}'.format(str(ind + 1)))
        save_image(img, save_path, 'test_{}'.format(str(ind + 1)))


def create_concat_test_from_path(label_path, save_path, save_normalized=False, save_overlay=False):
    """
    Create concatenated test files
    Returns
    -------

    """
    img_list = collect_and_concat_into_images(label_path, nb_patch_per_image=9, target_size=(200, 200))
    input_list = collect_and_concat_into_images(label_path, nb_patch_per_image=9, target_size=(200, 200),
                                                prefix='inp')
    create_concat_test_from_images(img_list, input_list[0], save_path[0], save_normalized, save_overlay)
    #
    # for ind, img in enumerate(img_list[0]):
    #     if save_normalized:
    #         img_norm = normalize_image_array_by_patch(img, patch_size=patch_size)
    #         save_image(img_norm, save_path, 'test_normal_{}'.format(str(ind + 1)))
    #     save_image(img, save_path, 'test_{}'.format(str(ind+1)))


def pipeline_from_masks_to_submission(model, title, proj_path, regenerate=False,
                                      save_normalized=False, save_overlay=True):
    label_path = os.path.join(proj_path, model, title)
    save_path = os.path.join(label_path, 'complete_test_label_submission')
    if regenerate:
        # Clean the submission files before generation
        if tf.gfile.Exists(save_path):
            tf.gfile.DeleteRecursively(save_path)
        tf.gfile.MakeDirs(save_path)
        # Generate the concatenated submission files
        create_concat_test_from_path(label_path=label_path, save_path=save_path, save_normalized=save_normalized,
                                     save_overlay=save_overlay)
    # Generate the submission csv according to the list
    complete_label_files = glob.glob(save_path + '/test*')
    submission_filename = proj_path + model + '_' + title + '_patch' + str(patch_size) + '.csv'
    masks_to_submission(submission_filename, *complete_label_files)


def pipeline_runtime_from_mask_to_submission(model, title, proj_path, input_imgs, pred_imgs,
                                             nb_patch_per_image, index_lim,
                                             save_normalized=False, save_overlay=True):
    """
    Generate runtime submission files and related image inputs. Directly from evaluation_fcn.py

    Parameters
    ----------
    model
    title
    proj_path
    input_imgs
    pred_imgs
    nb_patch_per_image
    index_lim
    save_normalized
    save_overlay

    Returns
    -------

    """
    label_path = os.path.join(proj_path, model, title)
    save_path = os.path.join(label_path, 'complete_test_label_submission')

    # Clean the submission files before generation
    if tf.gfile.Exists(save_path):
        tf.gfile.DeleteRecursively(save_path)
    tf.gfile.MakeDirs(save_path)

    # Concatenate images
    image_list, pred_list = concatenate_batches((input_imgs, pred_imgs), index_lim, dim_ordering='tf',
                                                nb_patch_per_image=nb_patch_per_image)

    # Save the concatenated images with normalization, overlay option
    create_concat_test_from_images(pred_list, image_list, save_path, save_normalized, save_overlay)

    # Fetch the test image generated to use the masks_to_submission file pipeline.
    complete_label_files = glob.glob(save_path + '/test_*.png')
    submission_filename = os.path.join(proj_path, model + '_' + title + '_patch' + str(patch_size) + '.csv')
    masks_to_submission(submission_filename, *complete_label_files)


if __name__ == '__main__':
    # create_concat_test()
    model = 'fcn4s_visual'
    title = 'plot_finetune_5000_test'
    proj_path = '/home/kyu/Dropbox/git/ml_project2/'
    pipeline_from_masks_to_submission(model, title, proj_path, regenerate=True, save_normalized=True, save_overlay=True)