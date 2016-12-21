# keras
from keras.models import load_model

# python
import os
from PIL import Image
import numpy as np
import re
import matplotlib.image as mpimg

# project files
from utils.io_utils import get_models_dir, get_dataset_dir
from utils.data_utils import sort_human, make_img_overlay
from utils.image_utils import extractPatches
from model.alexnet import alexnet_v1

########################################################################################################################
# Input parameters
########################################################################################################################

imgSize = (608, 608) # width, height
patchSize = (16, 16) # width, height

########################################################################################################################
# Main script
########################################################################################################################

# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
    if df > 0.25:
        return 1
    else:
        return 0

def mask_to_submission_strings(image_filename):
    """Reads a single image and outputs the strings that should go into the submission file
    Make python generator and send into writelines

    Update 2016.12.18
        Revise the image_filename logic to support absolute path
        Revise to support both path or image array directly
    """
    patch_size = 16

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

def predictions2labels(predictions):
    labels = np.zeros(predictions.shape[0], dtype=np.int32)

    for i, p in enumerate(predictions):
        labels[i] = int(p[1] > p[0])

    return labels

# Check the size of image and patch.
assert(imgSize[0] % patchSize[0] == 0)
assert(imgSize[1] % patchSize[1] == 0)

patchesInRow = imgSize[0] // patchSize[0]
patchesInCol = imgSize[1] // patchSize[1]
patchesInImage = patchesInRow * patchesInCol

# Input paths and file names.
datasetDir = get_dataset_dir('prml2/test')
inputPathModels = get_models_dir()
outputPathBlendImages = get_dataset_dir('prml2/results_overlay')
outputPathMasks = get_dataset_dir('prml2/results_masks')
outputPathSubmission = get_dataset_dir('prml2')
outputFileSubmission = 'submission_resnet34.csv'
# modelFile = 'LeNet5_50tr50va_bs32_noBalance-LeNet5_1_model.h5'
# modelFile = 'alex_50tr50va_bs32_noBalance-AlexNet_v1_1_model.h5'
# modelFile = 'ResNet_50tr50va_bs32_noBalance-resnet_1_model.h5'
# modelFile = 'AlexNet_v1-AlexNet_v1_1.weights'
modelFile = 'ResNet34orig_50tr50va_bs32_noBalance-resnet_1_model.h5'

modelFlag = 'resnet'

# Load model.
if modelFlag == 'alex':
    model = alexnet_v1((3, 16, 16))
    model.load_weights(os.path.join(inputPathModels, modelFile))
else:
    model = load_model(os.path.join(inputPathModels, modelFile))

# Load test set images.
imgFiles = os.listdir(datasetDir)
imgFiles = sort_human(imgFiles)
numImages = len(imgFiles)
images = [Image.open(os.path.join(datasetDir, f)) for f in imgFiles]

X = np.zeros((numImages * patchesInImage, 3, patchSize[1], patchSize[0]))
for imIdx, img in enumerate(images):
    # img = Image.open(os.path.join(datasetDir, f))
    imgPatches = extractPatches(img, patchSize, patchSize)

    assert(len(imgPatches) == patchesInImage)

    for imgPatchesIdx, patchesIdx in enumerate(range(imIdx * patchesInImage, (imIdx + 1) * patchesInImage)):
        X[patchesIdx] = np.transpose(np.asarray(imgPatches[imgPatchesIdx], dtype=np.float32) / 255.0, (2, 0, 1))

# Predict
patchesLabels = predictions2labels(model.predict(X, verbose=1))

# Create images from labels
for i in range(numImages):
    patchesLabelsInOneImage = patchesLabels[i * patchesInImage: (i + 1) * patchesInImage].reshape((patchesInCol, patchesInRow))
    # labelImage = np.kron(patchesLabelsInOneImage, np.ones(patchSize) * 255).astype(np.uint8)
    labelImage = np.kron(patchesLabelsInOneImage, np.ones(patchSize)).astype(np.uint8)

    # Save mask
    mask = Image.fromarray(labelImage * 255)
    mask.save(os.path.join(outputPathMasks, imgFiles[i]))

    # Load original image
    image = np.asarray(images[i], dtype=np.uint8)

    # Make overlay
    blend = make_img_overlay(image, labelImage)
    blend.save(os.path.join(outputPathBlendImages, 'res_{:02d}.png'.format(i)))

# Generate submission
masksNames = [os.path.join(outputPathMasks, fn) for fn in imgFiles]
masks_to_submission(os.path.join(outputPathSubmission, outputFileSubmission), *masksNames)
