import os
from PIL import Image
import numpy as np
import logging
import logging.config

from scipy import misc as misc

from project2.model.utils import unprocess_image


def load_img(path, greyscale=False, target_size=None):
    '''Load an image into PIL format.

    # Arguments
        path: path to image file
        grayscale: boolean
        target_size: None (default to original size)
            or (img_height, img_width)
    '''
    img = Image.open(path)
    if greyscale:
        img = img.convert('L')
    else:  # Ensure 3 channel even when loaded image is grayscale
        img = img.convert('RGB')
    if target_size:
        img = img.resize((target_size[1], target_size[0]))
    return img

def extractPatches(img, patchSize, stride):
    """ Returns the list of image patches of size `patch size` which
    are extracted with regards to `stride`.

    Patches are extracted from left to roght from top to bottom,
    first patch is exacty in the top left corner so that is is nicely
    aligned with the border of the image. If the patch would transcend
    the border of the image, it is not extracted (i.e. akin to behaviour
    of 'order_mode=valid').

    Parameters
    ----------
    img : PIL.Image
        Input image

    patchSize : tuple, int32
        2-tuple, width and height of the image patches to be extracted.

    stride : tuple, int32
        2-tuple, width and height of the stride to make when extracting patches.

    Returns
    -------
    patches : list of PIL.Image
        List of extracted patches.
    """

    # Sanity check for correct size and values in patchSize and stride sizes.
    if len(patchSize) != 2 or len(stride) != 2 or \
        patchSize[0] < 1   or patchSize[1] < 1 or \
        stride[0]    < 1   or stride[1]    < 1:
        raise('Both parameters "patchSize" and "stride" must be 2-tuples and '
              'the values must be positive.')

    # Image size
    cols = img.width
    rows = img.height

    # Check that at least one patch can be extracted.
    if patchSize[0] > cols or patchSize[1] > rows:
        logging.warning('Patch size is bigger, then the image. '
                        'No patches will be extracted.')

    # Steps to place patch centers.
    numPatchesInRow = (cols + stride[0] - patchSize[0]) // stride[0]
    numPatchesInCol = (rows + stride[1] - patchSize[1]) // stride[1]

    # Coordinates of patches' top left corner.
    tlInRow = np.arange(numPatchesInRow) * stride[0]
    tlInCol = np.arange(numPatchesInCol) * stride[1]

    # Extract patches
    patches = []
    for top in tlInCol:
        for left in tlInRow:
            # top, left, right, botom
            roi = (left, top, left + patchSize[0], top + patchSize[1])
            patches.append(img.crop(roi))

    return patches


def getLabels(patches, threshold):
    """ Generates the binary labels {0, 1} for each patch according to whether the
    ratio of white pixels is higher (label=1) or lower (label=0) than `threshold`.
    Expects the image with integer data type, where pixel values are in [0, 255].

    Parameters
    ----------
    patches : list of PIL.Image
        Input list of patches represented as grayscale images (i.e. one channel).

    threshold : float32
        Ratio of white to black pixels, above which the patch is labeled as class 1,
        otherwise it is class 0. Must be in range [0.0, 1.0]

    Returns
    -------
    labels : np.array (int32)
        N-vector of labels belonging to classes {0, 1}.
    """

    # Check the value of threshold.
    if threshold < 0.0 or threshold > 1.0:
        raise Exception('Threshold must be in range [0.0, 1.0], now it is {t}. Aborting,'.
               format(t=threshold))

    # Process patches.
    labels = np.zeros(len(patches), dtype=np.int32)
    for idx, patch in enumerate(patches):
        img = np.asarray(patch, dtype=np.float32) / 255.0

        # Check that image is grayscale
        if len(img.shape) != 2 and img.shape[2] != 1:
            raise Exception('Input patches must be grayscale images. Found {ch} '
                            'channels. Aborting'.format(ch=img.shape[2]))

        # Find average pixel value in patch and compare to threshold.
        avgPxValue = np.mean(img)
        labels[idx] = (0, 1)[avgPxValue >= threshold]

    return labels


def extractLabeledPatches(imgData, imgLabel, patchSize, stride, threshold=0.25):

    """ Extracts the patches
    of size `patchSize` from `imgData` and saves them into two lists corresponding
    to two classes. The class for each patch corresponds to the prevalent color in
    the corresponding patch in `imgLabel` (which should be black-and-white). Both
    images are expected to be normalized, so that pixel values take values in range
    [0.0, 1.0].

    Parameters
    ----------
    imgData : PIL.Image
        Data image (RGB or grayscale).

    imgLabel : PIL.Image
        Label image (BW).

    patchSize : tuple, int32
        2-tuple, width and height of the image patches to be extracted.

    stride : tuple, int32
        2-tuple, width and height of the stride to make when extracting patches.

    threshold : float32
        Ratio of white to black pixels, above which the patch is labeled as class 1,
        otherwise it is class 0.

    Returns
    -------
    patchesBackground : list, PIL.Image
        List of images of size `patchSize` corresponding to class 0.

    patchesSignal : list, PIL.Image
        List of images of size `patchSize` corresponding to class 1.
    """

    # Extract patches.
    dataPatches = extractPatches(imgData, patchSize, stride)
    # labelPatches = extractPatches(normalizedImgLabel, patchSize, stride)
    labelPatches = extractPatches(imgLabel, patchSize, stride)

    # Get the labels.
    labels = getLabels(labelPatches, threshold)

    # Divide the patches into two lists.
    N = len(dataPatches)
    patchesBackground = [dataPatches[i] for i in range(N) if labels[i] == 0]
    patchesSignal     = [dataPatches[i] for i in range(N) if labels[i] == 1]

    return patchesBackground, patchesSignal


# Tests of functions defined in this module.
if __name__ == "__main__":
    # Set up logging.
    logging.config.fileConfig('../log/logging.conf')

    ####################################################################################
    # Test of extractPatches()
    logging.info('Testing extractPatches()')
    img = load_img('test/testimg.tiff')

    patchSize = (64, 64)
    stride = (64, 64)

    patchesImg = extractPatches(img, patchSize, stride)
    # for idx, p in enumerate(patchesImg):
    #     p.save('test/patchesimg/patch_{:04d}.tiff'.format(idx), 'TIFF')


    ####################################################################################
    # Test of getLabels()
    logging.info('Testing extractPatches()')
    imgLabel = load_img('test/testlabel.tiff', greyscale=True)

    patchesLabels = extractPatches(imgLabel, patchSize, stride)
    # for idx, p in enumerate(patchesLabels):
    #     p.save('test/patcheslabels/patch_{:04d}.tiff'.format(idx), 'TIFF')

    labels = getLabels(patchesLabels, 0.1)
    print('num samples class 1: {n}'.format(n=np.sum(labels == 1)))
    # for idx, p in enumerate(patchesLabels):
    #     if labels[idx] == 1:
    #         p.save('test/patcheslabels/patch_{:04d}.tiff'.format(idx), 'TIFF')
    #         patchesImg[idx].save('test/patchesimg/patch_{:04d}.tiff'.format(idx), 'TIFF')

    ####################################################################################
    # Test of extractLabeledPatches()
    img = load_img('test/testimg.tiff')
    imgLabel = load_img('test/testlabel.tiff', greyscale=True)

    background, signal = extractLabeledPatches(img, imgLabel, patchSize, stride, threshold=0.1)

    for idx, p in enumerate(background):
        p.save('test/patchesimg/bg_{:04d}.tiff'.format(idx), 'TIFF')

    for idx, p in enumerate(signal):
        p.save('test/patchesimg/sig_{:04d}.tiff'.format(idx), 'TIFF')


def save_image(image, save_dir, name, mean=None):
    """
    Save image by unprocessing if mean given else just save
    :param mean:
    :param image:
    :param save_dir:
    :param name:
    :return:
    """
    if mean:
        image = unprocess_image(image, mean)
    misc.imsave(os.path.join(save_dir, name + ".png"), image)