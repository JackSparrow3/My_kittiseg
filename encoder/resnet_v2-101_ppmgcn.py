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
from ResNet import resnet_v2 as resnet
import os


def inference(hypes, images, train=True):
    """.

    Args:
      images: Images placeholder, from inputs().
      train: whether the network is used for train of inference

    Returns:
      softmax_linear: Output tensor with the computed logits.
    """


    with slim.arg_scope(resnet.resnet_arg_scope()):
        logit, _ = resnet.resnet_v2_50(images,2,is_training=train,global_pool=False)
    logits = {}

    logits['images'] = images




    #TODO this is what we want
    logits['fcn_logits'] = logit


    return logits
