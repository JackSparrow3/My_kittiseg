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
"""Contains definitions for the preactivation form of Residual Networks.
Residual networks (ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
The full preactivation 'v2' ResNet variant implemented in this module was
introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027
The key difference of the full preactivation 'v2' variant compared to the
'v1' variant in [1] is the use of batch normalization before every weight layer.
Typical use:
   from tensorflow.contrib.slim.nets import resnet_v2
ResNet-101 for image classification into 1000 classes:
   # inputs has shape [batch, 224, 224, 3]
   with slim.arg_scope(resnet_v2.resnet_arg_scope()):
      net, end_points = resnet_v2.resnet_v2_101(inputs, 1000, is_training=False)
ResNet-101 for semantic segmentation into 21 classes:
   # inputs has shape [batch, 513, 513, 3]
   with slim.arg_scope(resnet_v2.resnet_arg_scope()):
      net, end_points = resnet_v2.resnet_v2_101(inputs,
                                                21,
                                                is_training=False,
                                                global_pool=False,
                                                output_stride=16)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from ResNet import resnet_utils
from ResNet.resnet_utils import gcn_small as gcn
from ResNet.resnet_utils import br as br
from ResNet.resnet_utils import ppm as ppm
from ResNet.resnet_utils import _upscore_layer as _upscore_layer
from ResNet.resnet_utils import CRP as crp
from ResNet.resnet_utils import conv as conv
slim = tf.contrib.slim
resnet_arg_scope = resnet_utils.resnet_arg_scope


@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride, rate=1,
               outputs_collections=None, scope=None):
  """Bottleneck residual unit variant with BN before convolutions.
  This is the full preactivation residual unit variant proposed in [2]. See
  Fig. 1(b) of [2] for its definition. Note that we use here the bottleneck
  variant which has an extra bottleneck layer.
  When putting together two consecutive ResNet blocks that use this unit, one
  should use stride = 2 in the last unit of the first block.
  Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth: The depth of the ResNet unit output.
    depth_bottleneck: The depth of the bottleneck layers.
    stride: The ResNet unit's stride. Determines the amount of downsampling of
      the units output compared to its input.
    rate: An integer, rate for atrous convolution.
    outputs_collections: Collection to add the ResNet unit output.
    scope: Optional variable_scope.
  Returns:
    The ResNet unit's output.
  """
  with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
    depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
    if depth == depth_in:
      shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
    else:
      shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,
                             normalizer_fn=None, activation_fn=None,
                             scope='shortcut')

    residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1,
                           scope='conv1')
    residual = resnet_utils.conv2d_same(residual, depth_bottleneck, 3, stride,
                                        rate=rate, scope='conv2')
    residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                           normalizer_fn=None, activation_fn=None,
                           scope='conv3')

    output = shortcut + residual

    return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.name,
                                            output)


def resnet_v2(inputs,
              blocks,
              num_classes=None,
              is_training=True,
              global_pool=True,
              output_stride=None,
              include_root_block=True,
              spatial_squeeze=True,
              reuse=None,
              scope=None):
  """Generator for v2 (preactivation) ResNet models.
  This function generates a family of ResNet v2 models. See the resnet_v2_*()
  methods for specific model instantiations, obtained by selecting different
  block instantiations that produce ResNets of various depths.
  Training for image classification on Imagenet is usually done with [224, 224]
  inputs, resulting in [7, 7] feature maps at the output of the last ResNet
  block for the ResNets defined in [1] that have nominal stride equal to 32.
  However, for dense prediction tasks we advise that one uses inputs with
  spatial dimensions that are multiples of 32 plus 1, e.g., [321, 321]. In
  this case the feature maps at the ResNet output will have spatial shape
  [(height - 1) / output_stride + 1, (width - 1) / output_stride + 1]
  and corners exactly aligned with the input image corners, which greatly
  facilitates alignment of the features to the image. Using as input [225, 225]
  images results in [8, 8] feature maps at the output of the last ResNet block.
  For dense prediction tasks, the ResNet needs to run in fully-convolutional
  (FCN) mode and global_pool needs to be set to False. The ResNets in [1, 2] all
  have nominal stride equal to 32 and a good choice in FCN mode is to use
  output_stride=16 in order to increase the density of the computed features at
  small computational and memory overhead, cf. http://arxiv.org/abs/1606.00915.
  Args:
    inputs: A tensor of size [batch, height_in, width_in, channels].
    blocks: A list of length equal to the number of ResNet blocks. Each element
      is a resnet_utils.Block object describing the units in the block.
    num_classes: Number of predicted classes for classification tasks.
      If 0 or None, we return the features before the logit layer.
    is_training: whether batch_norm layers are in training mode.
    global_pool: If True, we perform global average pooling before computing the
      logits. Set to True for image classification, False for dense prediction.
    output_stride: If None, then the output will be computed at the nominal
      network stride. If output_stride is not None, it specifies the requested
      ratio of input to output spatial resolution.
    include_root_block: If True, include the initial convolution followed by
      max-pooling, if False excludes it. If excluded, `inputs` should be the
      results of an activation-less convolution.
    spatial_squeeze: if True, logits is of shape [B, C], if false logits is
        of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
        To use this parameter, the input images must be smaller than 300x300
        pixels, in which case the output logit layer does not contain spatial
        information and can be removed.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.
  Returns:
    net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
      If global_pool is False, then height_out and width_out are reduced by a
      factor of output_stride compared to the respective height_in and width_in,
      else both height_out and width_out equal one. If num_classes is 0 or None,
      then net is the output of the last ResNet block, potentially after global
      average pooling. If num_classes is a non-zero integer, net contains the
      pre-softmax activations.
    end_points: A dictionary from components of the network to the corresponding
      activation.
  Raises:
    ValueError: If the target output_stride is not valid.
  """
  with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    with slim.arg_scope([slim.conv2d, bottleneck,
                         resnet_utils.stack_blocks_dense],
                        outputs_collections=end_points_collection):
      with slim.arg_scope([slim.batch_norm], is_training=is_training):
        net = inputs
        if include_root_block:
          if output_stride is not None:
            if output_stride % 4 != 0:
              raise ValueError('The output_stride needs to be a multiple of 4.')
            output_stride /= 4
          # We do not include batch normalization or activation functions in
          # conv1 because the first ResNet unit will perform these. Cf.
          # Appendix of [2].
          with slim.arg_scope([slim.conv2d],
                              activation_fn=None, normalizer_fn=None):
            net = resnet_utils.conv2d_same(net,32, 3 ,2,scope='conv0_1')
            net = resnet_utils.conv2d_same(net, 64,3, 1,scope='conv0_2')
            net = resnet_utils.conv2d_same(net, 64, 3, 1, scope='conv0_3')
          # net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
        net = resnet_utils.stack_blocks_dense(net, blocks, output_stride)
        # This is needed because the pre-activation variant does not have batch
        # normalization or activation functions in the residual unit output. See
        # Appendix of [2].

        # Convert end_points_collection into a dictionary of end_points.
        end_points = slim.utils.convert_collection_to_dict(
            end_points_collection)
        net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
        end_points['postnorm']=net
        with slim.arg_scope([slim.conv2d_transpose], stride=2, padding='VALID', activation_fn=None,
                            normalizer_fn=None):
            # net = end_points['resnet_v2_50/block4']
            # 47x156x2048

            # 12x39x2048
            end_point = 'Mixed_fuse_0'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = tf.concat(axis=3, values=[
			            slim.conv2d(branch_1, 192, [1, 3], scope='Conv2d_0b_1x3'),
			            slim.conv2d(branch_1, 192, [3, 1], scope='Conv2d_0c_3x1')])
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(
			            branch_2, 384, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = tf.concat(axis=3, values=[
			            slim.conv2d(branch_2, 192, [1, 3], scope='Conv2d_0c_1x3'),
			            slim.conv2d(branch_2, 192, [3, 1], scope='Conv2d_0d_3x1')])
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], 1, 'SAME', scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(
			            branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net
            # 12x39x1280
            end_point = 'CRP_0'
            net, end_points = crp(net, end_points, 1280, end_point)
            # 12x39x1280
            end_point = 'trans_0'
            net, end_points = _upscore_layer(net, end_points, tf.shape(end_points['resnet_v2_50/block3']),1024,
                                             wkersize=2, hkersize=2, name=end_point)
            # 24x78x1024
            end_point = 'fuse_0'
            net = tf.concat([net, end_points['resnet_v2_50/block3']], axis=3, name=end_point)
            end_points[end_point] = net
            # 24x78x2048
            end_point = 'Mixed_fuse_1'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0, end_points = conv(net, end_points, [1, 1, 2048, 192],
		                                        name='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1, end_points = conv(net, end_points, [1, 1, 2048, 160],
		                                        name='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 160, [1, 7],
		                                   scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1],
		                                   scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2, end_points = conv(net, end_points, [1, 1, 2048, 160],
		                                        name='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1],
		                                   scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 160, [1, 7],
		                                   scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1],
		                                   scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7],
		                                   scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], 1, 'SAME', scope='AvgPool_0a_3x3')
                    branch_3, end_points = conv(branch_3, end_points, [1, 1, 2048, 192],
		                                        name='Conv2d_0b_1x1')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net

            # 21x75x768
            end_point = 'CRP_1'
            net, end_points = crp(net, end_points, 768, end_point)
            # 21x75x768
            end_point = 'trans_1'
            net, end_points = _upscore_layer(net, end_points, tf.shape(end_points['resnet_v2_50/block2']),512,
                                             wkersize=2, hkersize=2, name=end_point)
            # 47x156x512
            end_point = 'fuse_1'
            net = tf.concat([net, end_points['resnet_v2_50/block2']], axis=3, name=end_point)
            end_points[end_point] = net
            # 47x156x1024
            end_point = 'Mixed_fuse_2'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0, end_points = conv(net, end_points, [1, 1, 1024, 96],
		                                        name='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1, end_points = conv(net, end_points, [1, 1, 1024, 128],
		                                        name='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 128, [5, 5],
		                                   scope='Conv_1_0b_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2, end_points = conv(net, end_points, [1, 1, 1024, 128],
		                                        name='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 128, [3, 3],
		                                   scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, 192, [3, 3],
		                                   scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], 1, 'SAME', scope='AvgPool_0a_3x3')
                    branch_3, end_points = conv(branch_3, end_points, [1, 1, 1024, 96],
		                                        name='Conv2d_0b_1x1')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net

            # 45x153x512
            end_point = 'CRP_2'
            net, end_points = crp(net, end_points, 512, end_point)
            # 45x153x512
            end_point = 'trans_2'
            net, end_points = _upscore_layer(net, end_points, tf.shape(end_points['resnet_v2_50/block1']),256,
                                             wkersize=2, hkersize=2, name=end_point)
            # 94x311x256
            end_point = 'fuse_2'
            net = tf.concat([net, end_points['resnet_v2_50/block1']], axis=3, name=end_point)
            end_points[end_point] = net
            # 94x311x512
            end_point = 'Mixed_fuse_3'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0, end_points = conv(net, end_points, [1, 1, 512, 48],
		                                        name='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1, end_points = conv(net, end_points, [1, 1, 512, 32],
		                                        name='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 48, [5, 5],
		                                   scope='Conv_1_0b_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2, end_points = conv(net, end_points, [1, 1, 512, 32],
		                                        name='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 48, [3, 3],
		                                   scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, 48, [3, 3],
		                                   scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], 1, 'SAME', scope='AvgPool_0a_3x3')
                    branch_3, end_points = conv(branch_3, end_points, [1, 1, 512, 48],
		                                        name='Conv2d_0b_1x1')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net
            # 94x311x192
            end_point = 'CRP_3'
            net, end_points = crp(net, end_points, 192, end_point)
            # 94x311x192
            end_point = 'trans_3'
            if is_training:
                net, end_points = _upscore_layer(net, end_points, tf.shape(end_points['resnet_v2_50/conv0_3']),64,
                                             wkersize=2, hkersize=2, name=end_point)
                # 188x621x64
                end_point = 'fuse_3'
                net = tf.concat([net, end_points['resnet_v2_50/conv0_3']], axis=3, name=end_point)
                end_points[end_point] = net
            else:
                net, end_points = _upscore_layer(net, end_points, tf.shape(end_points['Validation/Validation/resnet_v2_50/conv0_3']), 64,
                                                 wkersize=2, hkersize=2, name=end_point)
                # 188x621x64
                end_point = 'fuse_3'
                net = tf.concat([net, end_points['Validation/Validation/resnet_v2_50/conv0_3']], axis=3, name=end_point)
                end_points[end_point] = net
            # 188x621x128
            end_point = 'Mixed_fuse_4'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0, end_points = conv(net, end_points, [1, 1, 128, 4],
		                                        name='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1, end_points = conv(net, end_points, [1, 1, 128, 4],
		                                        name='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 4, [5, 5],
		                                   scope='Conv_1_0b_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2, end_points = conv(net, end_points, [1, 1, 128, 2],
		                                        name='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 2, [3, 3],
		                                   scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, 4, [3, 3],
		                                   scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], 1, 'SAME', scope='AvgPool_0a_3x3')
                    branch_3, end_points = conv(branch_3, end_points, [1, 1, 128, 4],
		                                        name='Conv2d_0b_1x1')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net

            # 189x621x16
            end_point = 'CRP_4'
            net, end_points = crp(net, end_points, 16, end_point)
            # 189x621x16
            end_point = 'trans_4'
            net, end_points = _upscore_layer(net, end_points, tf.shape(inputs),num_classes,
                                             wkersize=2, hkersize=2, name=end_point)
            # 375x1242x2
            net, end_points = conv(net, end_points, [1, 1, 2, 2], name='Trans_5_conv_1')
            end_point = 'CRP_5'
            net, end_points = crp(net, end_points, 2, end_point)



        return net, end_points



resnet_v2.default_image_size = 224


def resnet_v2_block(scope, base_depth, num_units, stride,rate):
  """Helper function for creating a resnet_v2 bottleneck block.
  Args:
    scope: The scope of the block.
    base_depth: The depth of the bottleneck layer for each unit.
    num_units: The number of units in the block.
    stride: The stride of the block, implemented as a stride in the last unit.
      All other units have stride=1.
  Returns:
    A resnet_v2 bottleneck block.
  """
  return resnet_utils.Block(scope, bottleneck, [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': 1,
      'rate': rate
  }] * (num_units - 1) + [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': stride,
      'rate':rate
  }])
resnet_v2.default_image_size = 224


def resnet_v2_50(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=False,
                 output_stride=None,
                 spatial_squeeze=False,
                 reuse=None,
                 scope='resnet_v2_50'):
  """ResNet-50 model of [1]. See resnet_v2() for arg and return description."""
  blocks = [
      resnet_v2_block('block1', base_depth=64, num_units=3, stride=2,rate=1),
      resnet_v2_block('block2', base_depth=128, num_units=4, stride=2,rate=1),
      resnet_v2_block('block3', base_depth=256, num_units=6, stride=2,rate=1),
      resnet_v2_block('block4', base_depth=512, num_units=3, stride=2,rate=1),
  ]
  return resnet_v2(inputs, blocks, num_classes, is_training=is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, spatial_squeeze=spatial_squeeze,
                   reuse=reuse, scope=scope)
resnet_v2_50.default_image_size = resnet_v2.default_image_size


def resnet_v2_101(inputs,
                  num_classes=None,
                  is_training=True,
                  global_pool=False,
                  output_stride=None,
                  spatial_squeeze=False,
                  reuse=None,
                  scope='resnet_v2_101'):
  """ResNet-101 model of [1]. See resnet_v2() for arg and return description."""
  blocks = [
      resnet_v2_block('block1', base_depth=64, num_units=3, stride=2,rate=1),
      resnet_v2_block('block2', base_depth=128, num_units=4, stride=2,rate=1),
      resnet_v2_block('block3', base_depth=256, num_units=23, stride=1,rate=2),
      resnet_v2_block('block4', base_depth=512, num_units=3, stride=1,rate=4),
  ]
  return resnet_v2(inputs, blocks, num_classes, is_training=is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, spatial_squeeze=spatial_squeeze,
                   reuse=reuse, scope=scope)
resnet_v2_101.default_image_size = resnet_v2.default_image_size


def resnet_v2_152(inputs,
                  num_classes=None,
                  is_training=True,
                  global_pool=True,
                  output_stride=None,
                  spatial_squeeze=True,
                  reuse=None,
                  scope='resnet_v2_152'):
  """ResNet-152 model of [1]. See resnet_v2() for arg and return description."""
  blocks = [
      resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
      resnet_v2_block('block2', base_depth=128, num_units=8, stride=2),
      resnet_v2_block('block3', base_depth=256, num_units=36, stride=2),
      resnet_v2_block('block4', base_depth=512, num_units=3, stride=1),
  ]
  return resnet_v2(inputs, blocks, num_classes, is_training=is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, spatial_squeeze=spatial_squeeze,
                   reuse=reuse, scope=scope)
resnet_v2_152.default_image_size = resnet_v2.default_image_size


def resnet_v2_200(inputs,
                  num_classes=None,
                  is_training=True,
                  global_pool=True,
                  output_stride=None,
                  spatial_squeeze=True,
                  reuse=None,
                  scope='resnet_v2_200'):
  """ResNet-200 model of [2]. See resnet_v2() for arg and return description."""
  blocks = [
      resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
      resnet_v2_block('block2', base_depth=128, num_units=24, stride=2),
      resnet_v2_block('block3', base_depth=256, num_units=36, stride=2),
      resnet_v2_block('block4', base_depth=512, num_units=3, stride=1),
  ]
  return resnet_v2(inputs, blocks, num_classes, is_training=is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, spatial_squeeze=spatial_squeeze,
                   reuse=reuse, scope=scope)
resnet_v2_200.default_image_size = resnet_v2.default_image_size
