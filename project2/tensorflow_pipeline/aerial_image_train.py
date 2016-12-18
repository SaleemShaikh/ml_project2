
import gzip
import os
import sys
import urllib
import matplotlib.image as mpimg
from PIL import Image

import code

import tensorflow.python.platform

import numpy as np
import tensorflow as tf

from project2.utils.data_utils import img_crop, value_to_class, concatenate_images, label_to_img, make_img_overlay, error_rate
from project2.tensorflow_pipeline.arial_image_load import get_data
from project2.tensorflow_pipeline.aerial_image_model import lenet_example, get_prediction_with_groundtruth, \
    get_prediction_with_overlay

NUM_EPOCHS = 5

NUM_LABELS = 2
TRAINING_SIZE = 20
VALIDATION_SIZE = 5  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 16 # 64
RESTORE_MODEL = False # If True, restore existing model instead of training a new one
RECORDING_STEP = 1000


tf.app.flags.DEFINE_string('train_dir', '/tmp/prml2',
                           """Directory where to write event logs """
                           """and checkpoint.""")
FLAGS = tf.app.flags.FLAGS


def main(argv=None):  # pylint: disable=unused-argument

    # SET LABELS
    # TODO use io_utils to get data directory
    data_dir = '/home/kyu/.keras/datasets/prml2/training/'
    train_data_filename = data_dir + 'images/'
    train_labels_filename = data_dir + 'groundtruth/'
    num_epochs = NUM_EPOCHS


    # Get the image data and pack into data, label, size
    train_data, train_labels, train_size = get_data(TRAINING_SIZE)

    train_data_node, train_labels_node, logits, loss, model = lenet_example(train_data)

    # Optimizer: set up a variable that's incremented once per batch and
    # controls the learning rate decay.
    batch = tf.Variable(0)
    # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(
        0.01,                # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        train_size,          # Decay step.
        0.95,                # Decay rate.
        staircase=True)
    tf.scalar_summary('learning_rate', learning_rate)
    
    # Use simple momentum for the optimization.
    optimizer = tf.train.MomentumOptimizer(learning_rate,
                                           0.0).minimize(loss,
                                                         global_step=batch)

    # Predictions for the minibatch, validation set and test set.
    train_prediction = tf.nn.softmax(logits)
    # We'll compute them only once in a while by calling their {eval()} method.
    # train_all_prediction = tf.nn.softmax(model(train_all_data_node))

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Create a local session to run this computation.
    with tf.Session() as s:


        if RESTORE_MODEL:
            # Restore variables from disk.
            saver.restore(s, FLAGS.train_dir + "/model.ckpt")
            print("Model restored.")

        else:
            # Run all the initializers to prepare the trainable parameters.
            tf.global_variables_initializer().run()

            # Build the summary operation based on the TF collection of Summaries.
            summary_op = tf.merge_all_summaries()
            summary_writer = tf.train.SummaryWriter(FLAGS.train_dir,
                                                    graph_def=s.graph_def)
            print ('Initialized!')
            # Loop through training steps.
            print ('Total number of iterations = ' + str(int(num_epochs * train_size / BATCH_SIZE)))

            training_indices = range(train_size)

            for iepoch in range(num_epochs):

                # Permute training indices
                perm_indices = np.random.permutation(training_indices)

                for step in range (int(train_size / BATCH_SIZE)):

                    offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
                    batch_indices = perm_indices[offset:(offset + BATCH_SIZE)]

                    # Compute the offset of the current minibatch in the data.
                    # Note that we could use better randomization across epochs.
                    batch_data = train_data[batch_indices, :, :, :]
                    batch_labels = train_labels[batch_indices]
                    # This dictionary maps the batch data (as a numpy array) to the
                    # node in the graph is should be fed to.
                    feed_dict = {train_data_node: batch_data,
                                 train_labels_node: batch_labels}

                    if step % RECORDING_STEP == 0:

                        summary_str, _, l, lr, predictions = s.run(
                            [summary_op, optimizer, loss, learning_rate, train_prediction],
                            feed_dict=feed_dict)
                        #summary_str = s.run(summary_op, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, step)
                        summary_writer.flush()

                        # print_predictions(predictions, batch_labels)

                        print ('Epoch %.2f' % (float(step) * BATCH_SIZE / train_size))
                        print ('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                        print ('Minibatch error: %.1f%%' % error_rate(predictions,
                                                                     batch_labels))

                        sys.stdout.flush()
                    else:
                        # Run the graph and fetch some of the nodes.
                        _, l, lr, predictions = s.run(
                            [optimizer, loss, learning_rate, train_prediction],
                            feed_dict=feed_dict)

                # Save the variables to disk.
                save_path = saver.save(s, FLAGS.train_dir + "/model.ckpt")
                print("Model saved in file: %s" % save_path)

        print ("Running prediction on training set")
        prediction_training_dir = "predictions_training/"
        if not os.path.isdir(prediction_training_dir):
            os.mkdir(prediction_training_dir)
        for i in range(1, TRAINING_SIZE+1):
            pimg = get_prediction_with_groundtruth(train_data_filename, i, model)
            Image.fromarray(pimg).save(prediction_training_dir + "prediction_" + str(i) + ".png")
            oimg = get_prediction_with_overlay(train_data_filename, i, model)
            oimg.save(prediction_training_dir + "overlay_" + str(i) + ".png")
if __name__ == '__main__':
    tf.app.run()
