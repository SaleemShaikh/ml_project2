"""
FCN Training flow.

To train the model successfully, it is important to make sure the corresponding path to,
data set, hyper-parameters, are well defined
"""
import os

##################################################################
#          Runtime path TO BE SET before run                     #
##################################################################

PROJECT_DIR = '/cvlabdata1/home/kyu/ml_project2'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['KERAS_IMAGE_DIM_ORDERING'] = 'tf'

import datetime
import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator
from project2.tf_fcn.fcn_vgg_v2 import fcn8s, fcn32s
from project2.tf_fcn.utils import add_to_regularization_and_summary, add_gradient_summary, save_image

from project2.utils.data_utils import DirectoryImageLabelIterator
from project2.utils.io_utils import get_dataset_dir

# Specify which model to be trained.
train_function = fcn8s
MODEL_NAME = 'fcn8s_clean'
PLOT_DIR = 'plot_finetune'
MAX_ITERATION = int(50000 + 1)
FINETUNE_NAME = ''

FLAGS = tf.flags.FLAGS

# Specify train to train, finetune to finetune.
tf.flags.DEFINE_string('mode', "train", "Mode train/ finetune")

if FLAGS.mode == 'finetune':
    FINETUNE_NAME = '_{}_{}'.format('finetune', str(MAX_ITERATION-1))

# Directory related settings
tf.flags.DEFINE_string("data_dir", os.path.join(PROJECT_DIR, 'data'), 'path to data directory')
tf.flags.DEFINE_string("logs_dir", os.path.join(PROJECT_DIR, 'tensorboard', MODEL_NAME), "path to logs directory")
tf.flags.DEFINE_string("model_dir", os.path.join(PROJECT_DIR, 'model'), "Path to vgg model mat")
tf.flags.DEFINE_string('logs_dir_finetune',
                       os.path.join(PROJECT_DIR, 'tensorboard', MODEL_NAME + FINETUNE_NAME),
                       'Finetune log path')
tf.flags.DEFINE_string("plot_dir", os.path.join(PROJECT_DIR, 'output', MODEL_NAME + FINETUNE_NAME),
                       "path to plots")


# Hyper parameters and mode setting
tf.flags.DEFINE_integer("batch_size", "4", "batch size for training")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_bool('augmentation', 'False', 'Data runtime augmentation mode : True/ False')
tf.flags.DEFINE_bool('debug', "True", "Debug mode: True/ False")

NUM_OF_CLASSESS = 2
IMAGE_SIZE = 400
INPUT_SIZE = 224


def train(loss, var):
    """
    Define training operation (denoted as train_op) in the train loop
    Parameters
    ----------
    loss : tensor   return from loss function
    var : tensor    trainable weights

    Returns
    -------
    optimizer.apply_gradients(grad)
    """

    opt = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = opt.compute_gradients(loss, var_list=var)
    if FLAGS.debug:
        for grad, var in grads:
            add_gradient_summary(grad, var)
    return opt.apply_gradients(grads)


def main(argv=None):

    """
    Adapt from
        References: https://github.com/shekkizh/FCN.tensorflow/blob/master/FCN.py

    And this follows the standard training procedure of tensorflow

    Update 2016.12.16
        Finetune:
            converged model produce worse result due to two reason:
                0. Bugs in DirectoryIterator, only part of the training data is used
                1. Overfitting on the specific sub-sample of training data set
            Based on the Histogram of Gradient, the model is fully converged around Iterations 5000,
            for partial data.
            Comparing the result with not fully converged model, it indicates that the earlier model
             performs better actually.
    Update 2016.12.18
        Change the finetune to
            1. Stride to (100,100)
            2. Ratio = 1/2
        Implement Random horizontal and vertical flip

    Update 2016.12.20
        # DONE 1. Exhaust the revised model to converge on the complete training data set
        # DONE 2. Use the converged model, further exhaust on the data with run-time augmentation
        # DONE 3. With saving of Fully Exhausted model's training process (1000 iter per save),
        # DONE    generate the predictions on training set and testing set. Make the submission to
        # DONE    see the differences in F1 scores
        # CANC 4. Implement native F1 score in current FCN training process
        # DONE 5. Implement rotation
        # DONE 6. Finetune on another setting of dataset
    """

    # print flags:
    if FLAGS.debug:
        print(FLAGS.__dict__)

    keep_probability = tf.placeholder(tf.float32, name='keep_probability')
    image = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE, INPUT_SIZE, 3], name='input_img_tensor')
    annotation = tf.placeholder(tf.int32, shape=[None, INPUT_SIZE, INPUT_SIZE, 1], name='segmentation')

    pred_annotation, logits = train_function(image, keep_probability, FLAGS)
    loss = tf.reduce_mean(
        (tf.nn.sparse_softmax_cross_entropy_with_logits(
            # Use sqeeze because expand_dims first
            logits, tf.squeeze(annotation, squeeze_dims=[3]), name='xentropy')
        )
    )
    val_loss = tf.reduce_mean(
        (tf.nn.sparse_softmax_cross_entropy_with_logits(
            # Use sqeeze because expand_dims first
            logits, tf.squeeze(annotation, squeeze_dims=[3]), name='val_xentropy')
        )
    )

    # Write the result to tensor-board
    with tf.name_scope('train') as scope:
        tf.image_summary('input_image', image, max_images=2)
        tf.image_summary('ground_truth', tf.cast(tf.mul(annotation, 128), tf.uint8), max_images=2)
        tf.image_summary('pred_annotation', tf.cast(tf.abs(tf.mul(pred_annotation, 128)), tf.uint8), max_images=2)
    tf.scalar_summary('loss', loss)

    trainable_var = tf.trainable_variables()
    if FLAGS.debug:
        for var in trainable_var:
            # Monitor the regularization and summary
            add_to_regularization_and_summary(var)

    train_op = train(loss, trainable_var)

    print("setting up summary op ...")
    summary_op = tf.merge_all_summaries()

    with tf.name_scope('validation') as scope:
        val_input = tf.image_summary('val_input_image', image, max_images=2)
        val_gt = tf.image_summary('val_ground_truth', tf.cast(tf.mul(annotation, 128), tf.uint8), max_images=2)
        val_pd = tf.image_summary('val_pred_annotation', tf.cast(tf.abs(tf.mul(pred_annotation, 128)), tf.uint8), max_images=2)
    val_loss_summary = tf.scalar_summary('val_loss', val_loss)
    print("setting up validation summary op ...")
    val_summary_op = tf.merge_all_summaries()

    print('Setting up image reader ')
    if FLAGS.mode == 'train':
        train_itr = DirectoryImageLabelIterator(FLAGS.data_dir, None, stride=(128, 128),
                                                dim_ordering='tf',
                                                data_folder='massachuttes',
                                                image_folder='sat', label_folder='label',
                                                batch_size=FLAGS.batch_size,
                                                target_size=(INPUT_SIZE, INPUT_SIZE),
                                                )

    elif FLAGS.mode == 'finetune':
        gen = None
        if FLAGS.augmentation:
            train_itr = DirectoryImageLabelIterator(FLAGS.data_dir, gen, stride=(100, 100),
                                                    dim_ordering='tf',
                                                    data_folder='training',
                                                    image_folder='images', label_folder='groundtruth',
                                                    batch_size=FLAGS.batch_size,
                                                    target_size=(INPUT_SIZE, INPUT_SIZE),
                                                    ratio=1. / 2,
                                                    rotation='naive',
                                                    original_img_size=(400, 400),
                                                    )
        else:
            train_itr = DirectoryImageLabelIterator(FLAGS.data_dir, gen, stride=(100, 100),
                                                    dim_ordering='tf',
                                                    data_folder='training',
                                                    image_folder='images', label_folder='groundtruth',
                                                    batch_size=FLAGS.batch_size,
                                                    target_size=(INPUT_SIZE, INPUT_SIZE),
                                                    ratio=1./2,
                                                    original_img_size=(400,400),
                                                    )

    elif FLAGS.mode == 'test':
        if tf.gfile.Exists(FLAGS.plot_dir):
            tf.gfile.DeleteRecursively(FLAGS.plot_dir)
        tf.gfile.MakeDirs(FLAGS.plot_dir)
        gen = ImageDataGenerator(
            rotation_range=0.5,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            dim_ordering='tf'
        )
        train_itr = DirectoryImageLabelIterator(FLAGS.data_dir, gen, stride=(128, 128),
                                                dim_ordering='tf',
                                                data_folder='training',
                                                image_folder='images', label_folder='groundtruth',
                                                batch_size=FLAGS.batch_size,
                                                target_size=(INPUT_SIZE, INPUT_SIZE),
                                                shuffle=False,
                                                save_prefix='test',
                                                save_format='png',
                                                save_to_dir=FLAGS.plot_dir
                                                )

    valid_itr = DirectoryImageLabelIterator(FLAGS.data_dir, None, stride=(200, 200),
                                            dim_ordering='tf',
                                            data_folder='training',
                                            image_folder='images', label_folder='groundtruth',
                                            batch_size=FLAGS.batch_size,
                                            ratio=0.5,
                                            target_size=(INPUT_SIZE, INPUT_SIZE),
                                            )


    # Config settings
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    print("Setting up saver")
    saver = tf.train.Saver()
    if FLAGS.mode == 'train':
        summary_writer = tf.train.SummaryWriter(FLAGS.logs_dir, sess.graph)
    elif FLAGS.mode == 'finetune':
        summary_writer = tf.train.SummaryWriter(FLAGS.logs_dir_finetune, sess.graph)

    # Initialize model and possible restore
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path and FLAGS.mode != 'test':
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored from check point log {}".format(FLAGS.logs_dir))

    if FLAGS.mode == 'train':
        try:
            for itr in xrange(MAX_ITERATION):
                train_images, train_annotations = train_itr.next()
                feed_dict = {image: train_images,
                             annotation: train_annotations,
                             # onehot_annotation: onehot_train,
                             keep_probability: 0.85}
                sess.run(train_op, feed_dict=feed_dict)

                if itr % 10 == 0:
                    train_loss, summary_str = sess.run([loss, summary_op], feed_dict=feed_dict)
                    print("Step: %d, Train_loss:%g" % (itr, train_loss))
                    summary_writer.add_summary(summary_str, itr)

                if itr % 50 == 0:
                    valid_images, valid_annotations = valid_itr.next()
                    valid_loss, val_summary_str = sess.run([val_loss, val_summary_op], feed_dict={image: valid_images, annotation: valid_annotations,
                                                           keep_probability: 1.0})
                    print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))
                    summary_writer.add_summary(val_summary_str, itr)
                if itr % 10000 == 0:
                    saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)
        except KeyboardInterrupt:
            print("Stop training and save the model")
            saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)

    elif FLAGS.mode == 'finetune':
        try:
            for itr in xrange(MAX_ITERATION):
                train_images, train_annotations = train_itr.next()
                feed_dict = {image: train_images,
                             annotation: train_annotations,
                             keep_probability: 0.85}
                sess.run(train_op, feed_dict=feed_dict)

                if itr % 10 == 0:
                    train_loss, summary_str = sess.run([loss, summary_op], feed_dict=feed_dict)
                    print("Step: %d, Train_loss:%g" % (itr, train_loss))
                    summary_writer.add_summary(summary_str, itr)

                if itr % 50 == 0:
                    valid_images, valid_annotations = valid_itr.next()
                    valid_loss, val_summary_str = sess.run([loss, val_summary_op],
                                                           feed_dict={image: valid_images,
                                                                      annotation: valid_annotations,
                                                                      keep_probability: 1.0})
                    print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))
                    summary_writer.add_summary(val_summary_str, itr)

                if itr % 1000 == 0:
                    saver.save(sess, FLAGS.logs_dir_finetune + "model.ckpt", itr)

        except KeyboardInterrupt:
            print("Stop finetune and save the model")
            saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)

if __name__ == '__main__':
    tf.app.run()
