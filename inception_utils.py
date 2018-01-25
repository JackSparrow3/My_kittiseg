# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains common code shared by all inception models.

Usage of arg scope:
  with slim.arg_scope(inception_arg_scope()):
    logits, end_points = inception.inception_v3(images, num_classes,
                                                is_training=is_training)

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import tensorflow.contrib.slim as slim
import numpy as np
from math import ceil

def inception_arg_scope(weight_decay=5e-4,
                        use_batch_norm=True,
                        batch_norm_decay=0.9997,
                        batch_norm_epsilon=0.001):
  """Defines the default arg scope for inception models.

  Args:
    weight_decay: The weight decay to use for regularizing the model.
    use_batch_norm: "If `True`, batch_norm is applied after each convolution.
    batch_norm_decay: Decay for batch norm moving average.
    batch_norm_epsilon: Small float added to variance to avoid dividing by zero
      in batch norm.

  Returns:
    An `arg_scope` to use for the inception models.
  """

  batch_norm_params = {
      # Decay for the moving averages.
      'decay': batch_norm_decay,
      # epsilon to prevent 0s in variance.
      'epsilon': batch_norm_epsilon,
      # collection containing update_ops.
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
  }
  if use_batch_norm:
    normalizer_fn = slim.batch_norm
    normalizer_params = batch_norm_params
  else:
    normalizer_fn = None
    normalizer_params = {}
  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                      weights_regularizer=slim.l2_regularizer(weight_decay)):
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=slim.variance_scaling_initializer(),
        activation_fn=tf.nn.relu,
        normalizer_fn=normalizer_fn,
        normalizer_params=normalizer_params) as sc:
      return sc


def ppm(input,end_points,name=None):
  with tf.variable_scope('Pyramid_Pooling'):
    end_point=name+'branch_0'
    net=slim.avg_pool2d(input,[10,37],stride=1,padding='VALID',scope=end_point)
    end_points[end_point]=net
    end_point=end_point+'conv_0'
    net=slim.conv2d(net,1,[1,1],stride=1,padding='SAME',scope=end_point)
    end_points[end_point]=net
    end_point=end_point+'up'
    net=slim.conv2d_transpose(net,1,[10,37],stride=2,padding='VALID',scope=end_point)
    end_points[end_point]=net

    end_point=name+'branch_1'
    net=slim.avg_pool2d(input,[5,18],stride=[5,18],padding='VALID',scope=end_point)
    end_points[end_point]=net
    end_point=name+'conv_1'
    net=slim.conv2d(net,1,[1,1],stride=1,padding='SAME',scope=end_point)
    end_points[end_point]=net

    end_point=name+'branch_2'
    net=slim.avg_pool2d(input,[3,12],stride=[3,12],padding='VALID',scope=end_point)
    end_points[end_point]=net
    end_point=name+'conv_2'
    net=slim.conv2d(net,1,[1,1],stride=1,padding='SAME',scope=end_point)
    end_points[end_point]=net

    end_point=name+'branch_3'
    net=slim.avg_pool2d(input,[2,7],stride=[2,7],padding='VALID',scope=end_point)
    end_points[end_point]=net
    end_point=name+'conv_3'
    net=slim.conv2d(net,1,[1,1],stride=1,padding='SAME',scope=end_point)
    end_points[end_point]=net

def _upscore_layer(input, end_points=None, out_shape=None,depth=None, wkersize=None,hkersize=None,stride=2, padding='VALID', name=None):
  end_point=name
  strides = [1, stride, stride, 1]
  with tf.variable_scope(name):
    new_shape=[1,out_shape[1],out_shape[2],depth]
    # out_shape=tf.convert_to_tensor(new_shape)
    # new_shape = tf.stack(new_shape)
    in_channel = input.get_shape()[3].value
    out_channel=depth
    f_shape=[wkersize,hkersize,out_channel,in_channel]
    weights=_get_deconv_filter(f_shape=f_shape)

    net = tf.nn.conv2d_transpose(input, weights, output_shape=new_shape,
	                        strides=strides, padding=padding, name=end_point)
    end_points[end_point]=net
    return net , end_points

def _get_deconv_filter(f_shape=None):
  width = f_shape[0]
  heigh = f_shape[1]
  f = ceil(width / 2.0)
  c = (2 * f - 1 - f % 2) / (2.0 * f)
  bilinear = np.zeros([f_shape[0], f_shape[1]])
  for x in range(width):
    for y in range(heigh):
      value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
      bilinear[x, y] = value
  weights = np.zeros(f_shape)
  for i in range(f_shape[2]):
    weights[:, :, i, i] = bilinear

    init = tf.constant_initializer(value=weights,
	                               dtype=tf.float32)
    var = tf.get_variable(name="up_filter", initializer=init,
	                      shape=weights.shape)
    return var

def conv(input,end_points,shape=None,wd=0.00004,name=None,):
  end_point=name
  var_init=slim.xavier_initializer()
  var=tf.get_variable(name=name+'filter',shape=shape,dtype=tf.float32,initializer=var_init)
  # add to l2 loss
  weigh_deacy=tf.multiply(tf.nn.l2_loss(var),wd,name=name+'filter_wd')
  tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,weigh_deacy)

  bias_init=tf.constant(0.1,shape=[shape[3]],name=end_point+'bias')
  biases=tf.get_variable(name=name+'biases',initializer=bias_init)

  conv = tf.nn.conv2d(input, var, [1, 1, 1, 1], 'SAME', name=end_point)
  net=tf.nn.relu(conv+biases)

  end_points[end_point]=net
  return net, end_points

