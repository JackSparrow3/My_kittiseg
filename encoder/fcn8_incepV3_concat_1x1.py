"""
Utilize inceptionV3 as encoder.
------------------------

The MIT License (MIT)

Copyright (c) 2017 Marvin Teichmann


"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow.contrib.slim as slim
import tensorflow as tf
from inception import inception_v3_concat_1x1
import os


def inference(hypes, images, train=True):
    """.

    Args:
      images: Images placeholder, from inputs().
      train: whether the network is used for train of inference

    Returns:
      softmax_linear: Output tensor with the computed logits.
    """

    print ('dropout set to ',hypes['solver']['dropout'])
    with slim.arg_scope(inception_v3_concat_1x1.inception_v3_arg_scope()):
        _, logit, _ = inception_v3_concat_1x1.inception_v3_fcn(images,is_training=train,dropout_keep_prob=hypes['solver']['dropout'])
    logits = {}

    logits['images'] = images




#TODO this is what we want
    logits['fcn_logits'] = logit


    return logits
