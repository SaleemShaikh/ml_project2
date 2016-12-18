"""

FCN evaluation pipeline to generate submission for Kaggle

Evaluate FCN model based on the FCN model trained, it should process the test image
and generate patches aligning with the same format.

2016.12.15
"""

import datetime
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
import tensorflow as tf
from scipy.misc import imresize

from tf_fcn.fcn32_vgg import FCN32VGG
from project2.tf_fcn.fcn8_vgg import FCN8VGG
from project2.tf_fcn.loss import loss as dloss
from project2.tf_fcn.fcn_vgg_v2 import fcn4s, fcn32s
from project2.tf_fcn.utils import add_to_regularization_and_summary, save_image
from project2.utils.data_utils import DirectoryImageLabelIterator, concatenate_images, make_img_overlay, \
    greyscale_to_rgb, concatenate_patches, array_to_img
from project2.utils.io_utils import get_dataset_dir
from project2.utils.mask_to_submission import create_concat_test_from_images, \
    pipeline_runtime_from_mask_to_submission, concatenate_images_and_save





TITLE = 'fcn4s_visual'
PLOT_DIR = 'plot_finetune'
ITER = '4000'
index_lim = 3

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "9", "batch size for visualization")
tf.flags.DEFINE_string("logs_dir", "/home/kyu/.keras/tensorboard/" + TITLE, "path to logs directory")
tf.flags.DEFINE_string('project_dir', "/home/kyu/Dropbox/git/ml_project2", 'path to dropbox for synchronization')
tf.flags.DEFINE_string("plot_dir", os.path.join(FLAGS.project_dir, TITLE, PLOT_DIR + '_' + ITER), "path to plots")
# tf.flags.DEFINE_string("fcn_dir", "/home/kyu/.keras/tensorboard/fcn4s_finetune", "Path to FCN model")
tf.flags.DEFINE_string("fcn_dir", "/home/kyu/.keras/tensorboard/fcn4s_finetune_5000_newdata", "Path to FCN model")
tf.flags.DEFINE_string("data_dir", get_dataset_dir('prml2'), 'path to data directory')
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "predict", "Mode predict/ test/ visualize")
# tf.flags.DEFINE_string('mode', "visualize", "Mode predict/ test/ visualize")

# Initialize the model
tf.flags.DEFINE_string("model_dir", "/home/kyu/.keras/models/tensorflow", "Path to vgg model mat")

NUM_OF_CLASSESS = 2
IMAGE_SIZE = 400
INPUT_SIZE = 224


def main(argv=None):
    """
    Adapt and inspired by train_fcn.py

    Update 2016.12.16
        Implement the visualize pipeline to generate concatenated images
            Concat = [ground truth, image, predicted image]

    Update 2016.12.18
        Implement the prediction pipeline which could produce the result by :
            1. Load the testing set on one stride-patch settings
            2. Generate the prediction and concatenate back
            3. (optional) Overlay the result
            4. (optional) Concatenate the result next to it
            5. (optional) Normalize the above result and save to normalized version
            6. Generate the corresponding submission file
            7. Support probability plot and output for post-processing

    """

    # Make dir of plot dir
    if tf.gfile.Exists(FLAGS.plot_dir):
        tf.gfile.DeleteRecursively(FLAGS.plot_dir)
    tf.gfile.MakeDirs(FLAGS.plot_dir)

    # Define the tensorflow graph
    image = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE, INPUT_SIZE, 3], name='input_img_tensor')
    annotation = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE, INPUT_SIZE, 1], name='groundtruth')
    keep_probability = tf.placeholder(tf.float32, name='keep_probability')

    pred_annotation, logits = fcn4s(image, keep_prob=keep_probability, FLAGS=FLAGS)
    pred_softmax = tf.nn.softmax(logits, name="Softmax")

    loss = tf.reduce_mean(
        (tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits, tf.squeeze(pred_annotation, squeeze_dims=[3], name='xentropy')
        ))
    )

    # Summary node
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
    elif FLAGS.mode == 'predict':

        valid_itr = DirectoryImageLabelIterator(FLAGS.data_dir, None, stride=(600/index_lim, 600/index_lim),
                                                dim_ordering='tf',
                                                image_only=True,
                                                data_folder='test_set_images',
                                                image_folder='test_sat',
                                                batch_size=FLAGS.batch_size,
                                                target_size=(INPUT_SIZE, INPUT_SIZE),
                                                ratio=1./index_lim,
                                                original_img_size=(600,600),
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
        # ckpt_dir, _ = os.path.split(ckpt.model_ckeckpoint_path)
        # prefix = 'model.ckpt-'
        # ckpt_target = os.path.join(ckpt_dir, prefix + ITER)
        # if os.path.exists(ckpt_target):
        #     saver.restore(sess, ckpt_target)
        # else:
        # ckpt_list = saver.recover_last_checkpoints(FLAGS.fcn_dir)
        prefix = 'model.ckpt-'
        ckpt_path = os.path.join(FLAGS.fcn_dir, prefix + ITER)
        if os.path.exists(ckpt_path +'.meta'):
            print("Restoring model from {}".format(ckpt_path))
            saver.restore(sess, ckpt_path)
        else:
            print("Restore from default checkpoint")
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
        # List holds the image
        input_list = []
        result_list = []
        prob_img_list = []
        # List holds the raw prob
        prob_array_list = []

        while valid_itr.has_next_before_reset():
            valid_image = valid_itr.next()
            valid_annotation = np.zeros(shape=valid_image.shape[:3] + (1,))
            pred, smax = sess.run([pred_annotation, pred_softmax], feed_dict={image: valid_image,
                                                        annotation: valid_annotation,
                                                        keep_probability: 1.0})
            # Save the Imput image and result prediction annotation
            pred = np.squeeze(pred, axis=3) * 255
            prob_road = smax[:,:,:,1]

            for itr in range(FLAGS.batch_size):
                # Here generate related images
                img = valid_image[itr].astype(np.uint8)
                pred_img = pred[itr].astype(np.uint8)
                gt = valid_annotation[itr].astype(np.uint8)

                # Get probability map and append
                prob_road_raw = prob_road[itr]
                # Resize and normalize the probability
                prob_road_resize = imresize(prob_road_raw, (600/index_lim, 600/index_lim))
                prob_road_resize = prob_road_resize.astype(np.float32) / 255
                prob_array_list.append(prob_road_resize)

                # Convert raw probability into RGB
                prob_road_rgb = greyscale_to_rgb(prob_road_raw, pixel_depth=255)
                # Convert raw prediction file into RGB
                pred_img_rgb = greyscale_to_rgb(pred_img, pixel_depth=1)
                # Append all the image with resize
                input_list.append(imresize(img, (600/index_lim, 600/index_lim)))
                result_list.append(imresize(pred_img_rgb, (600/index_lim, 600/index_lim)))
                prob_img_list.append(imresize(prob_road_rgb, (600/index_lim, 600/index_lim)))

                if FLAGS.debug:
                    # Output preliminary results
                    gt_rgb = greyscale_to_rgb(gt, pixel_depth=1)
                    # res = np.concatenate((gt_rgb, img, pred_img_rgb), 1)
                    res = make_img_overlay(img, pred_img, 1)
                    res.save(FLAGS.plot_dir + "/overlay_" + str(index) + '.png')
                    save_image(res, FLAGS.plot_dir, name='concat_pred_' + str(index))
                    save_image(valid_image[itr].astype(np.uint8), FLAGS.plot_dir, name="inp_" + str(index))
                    # save_image(valid_annotation[itr].astype(np.uint8), FLAGS.plot_dir, name="gt_" + str(index))
                    save_image(pred[itr].astype(np.uint8), FLAGS.plot_dir, name="pred_" + str(index))
                    save_image(prob_road_rgb, FLAGS.plot_dir, name='prob_' + str(index))
                    print("Saved image: %d" % index)
                index += 1

        # After the loop:
        # Revoke the method in masks_to_submission
        pipeline_runtime_from_mask_to_submission(TITLE, PLOT_DIR + '_' + ITER, FLAGS.project_dir,
                                                 input_list, result_list,
                                                 index_lim * index_lim,
                                                 (index_lim, index_lim),
                                                 save_normalized=True, save_overlay=True)

        # Concatenate the probability raw into shape as the complete original image
        prob_concat_array = concatenate_patches((prob_array_list,), (index_lim, index_lim),
                                                nb_patch_per_image=index_lim*index_lim)

        prob_concat = np.asanyarray(prob_concat_array[0], dtype=np.float32)
        np.save(os.path.join(FLAGS.project_dir, TITLE, 'probability_{}_{}.gz'.format(TITLE,ITER)), prob_concat,
                allow_pickle=False)
        # Save the probability maps
        for ind, prob_map in enumerate(prob_concat_array[0]):
            # TODO Create heatmaps maybe
            save_image(array_to_img(prob_map), FLAGS.plot_dir, 'prob_' + str(ind))

if __name__ == '__main__':
    tf.app.run()
