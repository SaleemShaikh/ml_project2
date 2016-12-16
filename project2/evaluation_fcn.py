"""

FCN evaluation pipeline to generate submission for Kaggle

Evaluate FCN model based on the FCN model trained, it should process the test image
and generate patches aligning with the same format.

2016.12.15
"""

import datetime
import os

from tf_fcn.fcn32_vgg import FCN32VGG

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
import tensorflow as tf
from project2.tf_fcn.fcn8_vgg import FCN8VGG
from project2.tf_fcn.loss import loss as dloss
from project2.tf_fcn.fcn_vgg_v2 import fcn4s, fcn32s
from project2.tf_fcn.utils import add_to_regularization_and_summary, save_image
from project2.utils.data_utils import DirectoryImageLabelIterator, concatenate_images, make_img_overlay, \
    greyscale_to_rgb
from project2.utils.io_utils import get_dataset_dir

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "10", "batch size for visualization")
tf.flags.DEFINE_string("logs_dir", "/home/kyu/.keras/tensorboard/fcn4s_visual/", "path to logs directory")
tf.flags.DEFINE_string("plot_dir", "/home/kyu/Dropbox/git/ml_project2/fcn4s_visual/plot_finetune_4000", "path to plots")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "/home/kyu/.keras/models/tensorflow", "Path to vgg model mat")
# tf.flags.DEFINE_string("fcn_dir", "/home/kyu/.keras/tensorboard/fcn4s_finetune", "Path to FCN model")
tf.flags.DEFINE_string("fcn_dir", "/home/kyu/.keras/tensorboard/fcn4s_finetune_5000_newdata", "Path to FCN model")
tf.flags.DEFINE_string("data_dir", get_dataset_dir('prml2'), 'path to data directory')
tf.flags.DEFINE_bool('debug', "True", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "predict", "Mode predict/ test/ visualize")
# tf.flags.DEFINE_string('mode', "visualize", "Mode predict/ test/ visualize")

MAX_ITERATION = int(10e5 + 1)
NUM_OF_CLASSESS = 2
IMAGE_SIZE = 400
INPUT_SIZE = 224


def main(argv=None):
    """
    Adapt and inspired by train_fcn.py

    Update 2016.12.16
        Implement the visualize pipeline to generate concatenated images
            Concat = [ground truth, image, predicted image]

    """

    # Make dir of plot dir
    if tf.gfile.Exists(FLAGS.plot_dir):
        tf.gfile.DeleteRecursively(FLAGS.plot_dir)
    tf.gfile.MakeDirs(FLAGS.plot_dir)

    image = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE, INPUT_SIZE, 3], name='input_img_tensor')
    annotation = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE, INPUT_SIZE, 1], name='groundtruth')
    keep_probability = tf.placeholder(tf.float32, name='keep_probability')

    pred_annotation, logits = fcn4s(image, keep_prob=keep_probability, FLAGS=FLAGS)

    loss = tf.reduce_mean(
        (tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits, tf.squeeze(pred_annotation, squeeze_dims=[3], name='xentropy')
        ))
    )

    tf.image_summary('input_image', image, max_images=2)
    tf.image_summary('ground_truth', tf.cast(tf.mul(annotation, 128), tf.uint8), max_images=2)
    tf.image_summary('pred_annotation', tf.cast(tf.abs(tf.mul(pred_annotation, 128)), tf.uint8), max_images=2)

    tf.scalar_summary('entropy', loss)

    trainable_var = tf.trainable_variables()
    if FLAGS.debug:
        for var in trainable_var:
            # Monitor the regularization and summary
            add_to_regularization_and_summary(var)

    print("setting up summary op ...")
    # summary_op = tf.merge_all_summaries()
    if FLAGS.mode == 'visualize':
        valid_itr = DirectoryImageLabelIterator(FLAGS.data_dir, None, stride=(128, 128),
                                                dim_ordering='tf',
                                                data_folder='training',
                                                image_folder='images', label_folder='groundtruth',
                                                batch_size=FLAGS.batch_size,
                                                target_size=(INPUT_SIZE, INPUT_SIZE),
                                                shuffle=False,
                                                rescale=False
                                                )

    # Config settings
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    print("Setting up saver")
    saver = tf.train.Saver()
    summary_writer = tf.train.SummaryWriter(FLAGS.logs_dir, sess.graph)

    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(FLAGS.fcn_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Model restored')

    if FLAGS.mode == 'visualize':
        index = 0
        while valid_itr.has_next_before_reset():
            valid_image, valid_annotation = valid_itr.next()
            pred = sess.run(pred_annotation, feed_dict={image: valid_image,
                                                        annotation: valid_annotation,
                                                        keep_probability: 1.0})

            valid_annotation = np.squeeze(valid_annotation, axis=3) * 255
            pred = np.squeeze(pred, axis=3) * 255

            for itr in range(FLAGS.batch_size):
                img = valid_image[itr].astype(np.uint8)
                pred_img = pred[itr].astype(np.uint8)
                gt = valid_annotation[itr].astype(np.uint8)
                pred_img_rgb = greyscale_to_rgb(pred_img, pixel_depth=1)
                gt_rgb = greyscale_to_rgb(gt, pixel_depth=1)
                res = np.concatenate((gt_rgb, img, pred_img_rgb), 1)
                # res = make_img_overlay(img, pred_img, 1)
                # res = concatenate_images(res, gt)
                # res.save(FLAGS.plot_dir + "/overlay_" + str(index) + '.png')
                save_image(res, FLAGS.plot_dir, name='concat_' + str(index))
                save_image(valid_image[itr].astype(np.uint8), FLAGS.plot_dir, name="inp_" + str(index))
                save_image(valid_annotation[itr].astype(np.uint8), FLAGS.plot_dir, name="gt_" + str(index))
                save_image(pred[itr].astype(np.uint8), FLAGS.plot_dir, name="pred_" + str(index))
                print("Saved image: %d" % index)
                index += 1
    if FLAGS.mode == 'predict':
        # Create prediction pipeline
        index = 0




if __name__ == '__main__':
    tf.app.run()