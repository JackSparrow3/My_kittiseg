"""
Detects Cars in an image using KittiSeg.

Input: Image
Output: Image (with Cars plotted in Green)

Utilizes: Trained KittiSeg weights. If no logdir is given,
pretrained weights will be downloaded and used.

Usage:
python demo.py --input_image data/demo.png [--output_image output_image]
                [--logdir /path/to/weights] [--gpus 0]

--------------------------------------------------------------------------------

The MIT License (MIT)

Copyright (c) 2017 Marvin Teichmann

Details: https://github.com/MarvinTeichmann/KittiSeg/blob/master/LICENSE
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import sys
import time
import collections

# configure logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

# https://github.com/tensorflow/tensorflow/issues/2034#issuecomment-220820070
import numpy as np
import scipy as scp
import scipy.misc
import tensorflow as tf
from evals import kitti_eval
from evals.kitti_eval import evaluate
import incl.tensorvision.utils as utils

import post_crf
flags = tf.app.flags
FLAGS = flags.FLAGS
print (sys.path)
sys.path.insert(1, 'incl')

from incl.seg_utils import seg_utils as seg
curdir=os.getcwd()
# print (sys.path)
try:
    # Check whether setup was done correctly

    import incl.tensorvision.utils as tv_utils
    import incl.tensorvision.core as core
except ImportError:
    # You forgot to initialize submodules
    logging.error("Could not import the submodules.")
    logging.error("Please execute:"
                  "'git submodule update --init --recursive'")
    exit(1)

flags.DEFINE_string('logdir', 'InceptionV3_2018_02_05_10.35',
                    'Path to logdir.')
flags.DEFINE_string('input_image', '/home/yu/projects/KittiSeg/DATA/data_road/testing/image_2/umm_000000.png',
                    'Image to apply KittiSeg.')
flags.DEFINE_string('output_image', None,
                    'Image to apply KittiSeg.')

default_run = 'KittiSeg_pretrained'
weights_url = ("ftp://mi.eng.cam.ac.uk/"
               "pub/mttt2/models/KittiSeg_pretrained.zip")


def maybe_download_and_extract(runs_dir):
    logdir = os.path.join(runs_dir, default_run)

    if os.path.exists(logdir):
        # weights are downloaded. Nothing to do
        return

    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)
    download_name = tv_utils.download(weights_url, runs_dir)
    logging.info("Extracting KittiSeg_pretrained.zip")

    import zipfile
    zipfile.ZipFile(download_name, 'r').extractall(runs_dir)

    return


def resize_label_image(image, gt_image, image_height, image_width):
    image = scp.misc.imresize(image, size=(image_height, image_width),
                              interp='cubic')
    shape = gt_image.shape
    gt_image = scp.misc.imresize(gt_image, size=(image_height, image_width),
                                 interp='nearest')

    return image, gt_image


def main(_):
    tv_utils.set_gpus_to_use()

    if FLAGS.input_image is None:
        logging.error("No input_image was given.")
        logging.info(
            "Usage: python demo.py --input_image data/test.png "
            "[--output_image output_image] [--logdir /path/to/weights] "
            "[--gpus GPUs_to_use] ")
        exit(1)

    if FLAGS.logdir is None:
        # Download and use weights from the MultiNet Paper
        if 'TV_DIR_RUNS' in os.environ:
            runs_dir = os.path.join(os.environ['TV_DIR_RUNS'],
                                    'KittiSeg')
        else:
            runs_dir = 'RUNS'
        maybe_download_and_extract(runs_dir)
        logdir = os.path.join(runs_dir, default_run)
    else:
        logging.info("Using weights found in {}".format(FLAGS.logdir))
        logdir = FLAGS.logdir

    # Loading hyperparameters from logdir
    hypes = tv_utils.load_hypes_from_logdir(logdir, base_path='hypes')

    logging.info("Hypes loaded successfully.")

    # Loading tv modules (encoder.py, decoder.py, eval.py) from logdir
    modules = tv_utils.load_modules_from_logdir(logdir)
    logging.info("Modules loaded successfully. Starting to build tf graph.")

    # Create tf graph and build module.
    with tf.Graph().as_default():

        image_placehold= tf.placeholder(tf.float32,shape=[1,None,None,3])

        # build Tensorflow graph using the model from logdir
        prediction = core.build_inference_graph(hypes, modules,
                                                image=image_placehold)

        logging.info("Graph build successfully.")

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        saver = tf.train.Saver()

        # Load weights from logdir
        core.load_weights(logdir, sess, saver)

        # eval_dict,_=kitti_eval.evaluate_without_crf(hypes,sess,image_placehold,prediction)
        eval_dict , _ =kitti_eval.evaluate_test(hypes,sess,image_placehold,prediction)
        for name, value in eval_dict:
            logging.info('    %s %s : % 0.04f ' % (name, '(raw)', value))
        # utils.print_eval_dict(eval_dict, prefix='(raw) ')












if __name__ == '__main__':
    tf.app.run()
