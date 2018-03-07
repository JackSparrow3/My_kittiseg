"""
Utilize inceptionV3 as encoder.
------------------------

The MIT License (MIT)

Copyright (c) 2017 Marvin Teichmann


"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import tensorflow as tf
from inception import v3_refinev3 as inception
import os
import tensorflow.contrib.slim as slim

def inference(hypes, images, train=True):
    """.

    Args:
      images: Images placeholder, from inputs().
      train: whether the network is used for train of inference

    Returns:
      softmax_linear: Output tensor with the computed logits.
    """


    with slim.arg_scope(inception.inception_v3_arg_scope(weight_decay=5e-5)):
         logit, _ = inception.inception_v3_fcn(images,is_training=train,dropout_keep_prob=hypes['solver']['dropout'])
    logits = {}

    logits['images'] = images




    #TODO this is what we want
    logits['fcn_logits'] = logit


    return logits
