"""
Train.py as a pipe-line for the training process

Step 1.  Data read: [ RoadImageIterator]
    All data is read in a format:
        data = [image_id, center (x,y)]
        labels = [one_hot(label)]

Step 2. Feed the training into example_engine to obtain a classifier

Step 3. Save the model to the disk after tuned
    TODO Implement API in engine

Step 4. Predict the test data in patch-format
    TODO Implement prediction overlay generator
    Generate the data in the same format as data-read
    Pass it to the model.evaluate
    Generate the labels
    # Compare with the raw labels?
    Generate submit-files to the server

Step 5. Possible post processing like Conditional Random Field (CRF)


"""

from .utils.data_utils import RoadImageIterator
import tensorflow as tf
import keras.backend as K



