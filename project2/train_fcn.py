"""
FCN Training flow
"""
import datetime
import os

from tf_fcn.fcn32_vgg import FCN32VGG

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['KERAS_IMAGE_DIM_ORDERING'] = 'tf'

import numpy as np
import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator
from project2.tf_fcn.fcn8_vgg import FCN8VGG
from project2.tf_fcn.loss import loss as dloss
from project2.tf_fcn.fcn_vgg_v2 import fcn4s, fcn32s
from project2.tf_fcn.utils import add_to_regularization_and_summary, add_gradient_summary, save_image

from project2.utils.data_utils import DirectoryImageLabelIterator, concatenate_images
from project2.utils.io_utils import get_dataset_dir

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "4", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "/home/kyu/.keras/tensorboard/fcn4s/", "path to logs directory")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "/home/kyu/.keras/models/tensorflow", "Path to vgg model mat")
tf.flags.DEFINE_string('logs_dir_finetune', '/home/kyu/.keras/tensorboard/fcn4s_finetune_5000/', 'Finetune log path')
tf.flags.DEFINE_string("data_dir", get_dataset_dir('prml2'), 'path to data directory')
tf.flags.DEFINE_bool('debug', "True", "Debug mode: True/ False")
# tf.flags.DEFINE_string('mode', "visualize", "Mode train/ test/ visualize")
tf.flags.DEFINE_string('mode', "finetune", "Mode train/ test/ finetune/ visualize")
tf.flags.DEFINE_string("plot_dir", "/home/kyu/Dropbox/git/ml_project2/fcn4s_visual/plot_data_dir_test",
                       "path to plots")
MAX_ITERATION = int(5000 + 1)
NUM_OF_CLASSESS = 2
IMAGE_SIZE = 400
INPUT_SIZE = 224

# def train(loss_val, val_list):
#     optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)


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

    Parameters
    ----------
    argv

    Returns
    -------

    """
    # print flags:
    print(FLAGS.__dict__)


    keep_probability = tf.placeholder(tf.float32, name='keep_probability')
    image = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE, INPUT_SIZE, 3], name='input_img_tensor')
    annotation = tf.placeholder(tf.int32, shape=[None, INPUT_SIZE, INPUT_SIZE, 1], name='segmentation')
    # onehot_annotation = tf.placeholder(tf.int32, shape=[None, INPUT_SIZE, INPUT_SIZE, 2], name='onehot')

    pred_annotation, logits = fcn4s(image, keep_probability, FLAGS)



    """
    # Build VGG-FCN32
    ## It seems wrongly since FCN should not implement fully-connected layers ##
    Model design is good, however, it is not compatible with this train-procedure.
    In order to preceed, it is
    """
    # model = FCN32VGG(vgg16_npy_path=os.path.join(FLAGS.model_dir, 'vgg16.npy'))
    # model.build(image, train=True, num_classes=NUM_OF_CLASSESS, random_init_fc8=False, debug=True)
    # pred_annotation = tf.expand_dims(model.pred_up, dim=3)
    # logits = model.upscore

    # from project2.tf_fcn.loss import loss
    # loss = dloss(logits, onehot_annotation, num_classes=NUM_OF_CLASSESS)

    loss = tf.reduce_mean(
        (tf.nn.sparse_softmax_cross_entropy_with_logits(
            # Use sqeeze because expand_dims first
            logits, tf.squeeze(annotation, squeeze_dims=[3]), name='xentropy')
            # logits, annotation, name='xentropy')
        )
    )

    val_loss = tf.reduce_mean(
        (tf.nn.sparse_softmax_cross_entropy_with_logits(
            # Use sqeeze because expand_dims first
            logits, tf.squeeze(annotation, squeeze_dims=[3]), name='val_xentropy')
            # logits, annotation, name='xentropy')
        )
    )

    # f1score =

    # Write the result to tensor-board
    with tf.name_scope('train') as scope:
        tf.image_summary('input_image', image, max_images=2)
        tf.image_summary('ground_truth', tf.cast(tf.mul(annotation, 128), tf.uint8), max_images=2)
        tf.image_summary('pred_annotation', tf.cast(tf.abs(tf.mul(pred_annotation, 128)), tf.uint8), max_images=2)
        # tf.image_summary('ground_truth', tf.cast(annotation, tf.uint8), max_images=2)
        # tf.image_summary('pred_annotation', tfcast(pred_annotation, tf.uint8), max_images=2)

    tf.scalar_summary('loss', loss)

    trainable_var = tf.trainable_variables()
    if FLAGS.debug:
        for var in trainable_var:
            # Monitor the regularization and summary
            add_to_regularization_and_summary(var)

    train_op = train(loss, trainable_var)

    print("setting up summary op ...")
    summary_op = tf.merge_all_summaries()

    print("setting up validation summary op ...")

    # with tf.Graph().as_default() as graph:
    with tf.name_scope('validation') as scope:
        val_input = tf.image_summary('val_input_image', image, max_images=2)
        val_gt = tf.image_summary('val_ground_truth', tf.cast(tf.mul(annotation, 128), tf.uint8), max_images=2)
        val_pd = tf.image_summary('val_pred_annotation', tf.cast(tf.abs(tf.mul(pred_annotation, 128)), tf.uint8), max_images=2)
    val_loss_summary = tf.scalar_summary('val_loss', val_loss)
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
        gen = ImageDataGenerator(
            # rotation_range=0.2,
            # width_shift_range=0.1,
            # height_shift_range=0.1,
            # horizontal_flip=True,
            # vertical_flip=True,
            # dim_ordering='tf'
            )
        train_itr = DirectoryImageLabelIterator(FLAGS.data_dir, None, stride=(128, 128),
                                                dim_ordering='tf',
                                                data_folder='training',
                                                image_folder='images', label_folder='groundtruth',
                                                batch_size=FLAGS.batch_size,
                                                target_size=(INPUT_SIZE, INPUT_SIZE),
                                                )

    elif FLAGS.mode == 'test':
        if tf.gfile.Exists(FLAGS.plot_dir):
            tf.gfile.DeleteRecursively(FLAGS.plot_dir)
        tf.gfile.MakeDirs(FLAGS.plot_dir)
        gen = ImageDataGenerator(
            # rotation_range=0.2,
            # width_shift_range=0.1,
            # height_shift_range=0.1,
            # horizontal_flip=True,
            # vertical_flip=True,
            # dim_ordering='tf'
        )
        train_itr = DirectoryImageLabelIterator(FLAGS.data_dir, None, stride=(128, 128),
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

    # valid_itr = DirectoryImageLabelIterator(FLAGS.data_dir, None, stride=(128, 128),
    #                                         dim_ordering='tf',
    #                                         data_folder='massachuttes',
    #                                         image_folder='sat', label_folder='label',
    #                                         batch_size=FLAGS.batch_size,
    #                                         target_size=(INPUT_SIZE, INPUT_SIZE),
    #                                         )
    valid_itr = DirectoryImageLabelIterator(FLAGS.data_dir, None, stride=(128, 128),
                                            dim_ordering='tf',
                                            data_folder='training',
                                            image_folder='images', label_folder='groundtruth',
                                            batch_size=FLAGS.batch_size,
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
        print("Model restored")

    if FLAGS.mode == 'train':
        try:
            for itr in xrange(MAX_ITERATION):
                train_images, train_annotations = train_itr.next()
                # onehot_train = tf.one_hot(train_annotations, NUM_OF_CLASSESS)
                feed_dict = {image: train_images,
                             annotation: train_annotations,
                             # onehot_annotation: onehot_train,
                             keep_probability: 0.85}
                # feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.85}
                sess.run(train_op, feed_dict=feed_dict)

                if itr % 10 == 0:
                    train_loss, summary_str = sess.run([loss, summary_op], feed_dict=feed_dict)
                    print("Step: %d, Train_loss:%g" % (itr, train_loss))
                    summary_writer.add_summary(summary_str, itr)

                if itr % 500 == 0:
                    valid_images, valid_annotations = valid_itr.next()
                    valid_loss = sess.run(loss, feed_dict={image: valid_images, annotation: valid_annotations,
                                                           keep_probability: 1.0})
                    print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))
                    saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)
        except KeyboardInterrupt:
            print("Stop training and save the model")
            saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)

    elif FLAGS.mode == 'finetune':
        try:
            for itr in xrange(MAX_ITERATION):
                train_images, train_annotations = train_itr.next()
                # onehot_train = tf.one_hot(train_annotations, NUM_OF_CLASSESS)
                feed_dict = {image: train_images,
                             annotation: train_annotations,
                             # onehot_annotation: onehot_train,
                             keep_probability: 0.85}
                # feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.85}
                sess.run(train_op, feed_dict=feed_dict)

                if itr % 10 == 0:
                    train_loss, summary_str = sess.run([loss, summary_op], feed_dict=feed_dict)
                    print("Step: %d, Train_loss:%g" % (itr, train_loss))
                    summary_writer.add_summary(summary_str, itr)

                if itr % 500 == 0:
                    valid_images, valid_annotations = valid_itr.next()
                    valid_loss, val_summary_str = sess.run([loss, val_summary_op],
                                                           feed_dict={image: valid_images,
                                                                      annotation: valid_annotations,
                                                                      keep_probability: 1.0})
                    print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))
                    summary_writer.add_summary(val_summary_str, itr)
                    saver.save(sess, FLAGS.logs_dir_finetune + "model.ckpt", itr)
        except KeyboardInterrupt:
            print("Stop finetune and save the model")
            saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)


    elif FLAGS.mode == "visualize":
        valid_images, valid_annotations = valid_itr.next()
        pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations,
                                                    keep_probability: 1.0})
        # valid_annotations = np.squeeze(valid_annotations, axis=3)
        # pred = np.squeeze(pred, axis=3)

        for itr in range(FLAGS.batch_size):
            save_image(valid_images[itr].astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(5 + itr))
            save_image(valid_annotations[itr].astype(np.uint8), FLAGS.logs_dir, name="gt_" + str(5 + itr))
            save_image(pred[itr].astype(np.uint8), FLAGS.logs_dir, name="pred_" + str(5 + itr))
            print("Saved image: %d" % itr)

    elif FLAGS.mode == 'test':
        MAX_ITERATION = 100
        for itr in xrange(MAX_ITERATION):
            train_images, train_annotations = train_itr.next()

if __name__ == '__main__':
    tf.app.run()
