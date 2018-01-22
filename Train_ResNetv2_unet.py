"""
Trains

-------------------------------------------------

The MIT License (MIT)

Copyright (c) 2017 Marvin Teichmann


"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import commentjson
import logging
import os
import sys
slim = tf.contrib.slim

import collections
def dict_merge(dct, merge_dct):
    """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.
    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :return: None
    """
    for k, v in merge_dct.iteritems():
        if (k in dct and isinstance(dct[k], dict) and
                isinstance(merge_dct[k], collections.Mapping)):
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]


# configure logging
if 'TV_IS_DEV' in os.environ and os.environ['TV_IS_DEV']:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)
else:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)

# https://github.com/tensorflow/tensorflow/issues/2034#issuecomment-220820070
import numpy as np


flags = tf.app.flags
FLAGS = flags.FLAGS

sys.path.insert(1, 'incl')

import incl.tensorvision.train as train
import incl.tensorvision.utils as utils

flags.DEFINE_string('name', None,
                    'Append a name Tag to run.')

flags.DEFINE_string('hypes', 'hypes/InceptionV3.json',
                    'File storing model parameters.')

flags.DEFINE_string('project', None,
                    'Append a name Tag to run.')

# flags.DEFINE_string('hypes', 'hypes/InceptionV3.json',
#                     'File storing model parameters.')

flags.DEFINE_string('mod', None,
                    'Modifier for model parameters.')
#tf.flags.DEFINE_string('train_dir', 'logs/all','Directory where checkpoints and event logs are written to.')

tf.flags.DEFINE_string('checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')
# this should be 'InceptionV3/Logits,InceptionV3/AuxLogits', used to be None

tf.flags.DEFINE_string('trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')
# this should be 'InceptionV3/Logits,InceptionV3/Upsampling', used to be None

tf.flags.DEFINE_boolean(
    'ignore_missing_vars', True,'When restoring a checkpoint would ignore missing variables.')

tf.flags.DEFINE_string(
    'checkpoint_path', 'ResNet/resnet_v2_101.ckpt','The path to a checkpoint from which to fine-tune.')
# this should be 'logs', used to be None

tf.flags.DEFINE_string(
    'mystring', 'True','The path to a checkpoint from which to fine-tune.')


if 'TV_SAVE' in os.environ and os.environ['TV_SAVE']:
    tf.app.flags.DEFINE_boolean(
        'save', True, ('Whether to save the run. In case --nosave (default) '
                       'output will be saved to the folder TV_DIR_RUNS/debug, '
                       'hence it will get overwritten by further runs.'))
else:
    tf.app.flags.DEFINE_boolean(
        'save', True, ('Whether to save the run. In case --nosave (default) '
                       'output will be saved to the folder TV_DIR_RUNS/debug '
                       'hence it will get overwritten by further runs.'))




def train_loop(myhypes=None):
    utils.set_gpus_to_use()


    try:
        import tensorvision.train
        import tensorflow_fcn.utils
    except ImportError:
        logging.error("Could not import the submodules.")
        logging.error("Please execute:"
                      "'git submodule update --init --recursive'")
        exit(1)



    with open(myhypes, 'r') as f:
        logging.info("f: %s", f)
        hypes = commentjson.load(f)
    utils.load_plugins()

    if tf.app.flags.FLAGS.mod is not None:
        import ast
        mod_dict = ast.literal_eval(tf.app.flags.FLAGS.mod)
        dict_merge(hypes, mod_dict)

    if 'TV_DIR_RUNS' in os.environ:
        os.environ['TV_DIR_RUNS'] = os.path.join(os.environ['TV_DIR_RUNS'],
                                                 'KittiSeg')
    utils.set_dirs(hypes, tf.app.flags.FLAGS.hypes)

    utils._add_paths_to_sys(hypes)

    train.maybe_download_and_extract(hypes)
    logging.info("Initialize training folder")

    # TODO initialize the train folder and copy some arg files to it--------------------------------yu
    train.initialize_training_folder(hypes)
    logging.info("Start training")

    train.do_training(hypes, trainable_scopes=FLAGS.trainable_scopes, exclude_scopes=FLAGS.checkpoint_exclude_scopes,
                      checkpoint_path=FLAGS.checkpoint_path)


def main(_):
    # train_loop('hypes/InceptionV3.json')
    # train_loop('hypes/InceptionV3_concat.json')
    train_loop('hypes/Resnet-v2_ppmgcn_101.json')



if __name__ == '__main__':
    tf.app.run()
