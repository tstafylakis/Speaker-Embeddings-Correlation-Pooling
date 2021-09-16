# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for Residual Networks.

Residual networks ('v1' ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

The full preactivation 'v2' ResNet variant was introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The key difference of the full preactivation 'v2' variant compared to the
'v1' variant in [1] is the use of batch normalization before every weight layer
rather than after.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
DEFAULT_VERSION = 2
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES


################################################################################
# Convenience functions for building the ResNet model.
################################################################################

def SE_standard(input_x, out_dim, reg, ratio):
  DS = tf.keras.layers.Dense(units=int(out_dim / ratio), kernel_regularizer=reg, bias_regularizer=reg)
  DE = tf.keras.layers.Dense(units=out_dim, kernel_regularizer=reg, bias_regularizer=reg)
  squeeze = tf.math.reduce_mean(input_x, axis=[2,3])
  excitation = DS(squeeze)
  excitation = tf.nn.relu(excitation)
  excitation = DE(excitation)
  excitation = tf.nn.sigmoid(excitation)
  excitation = tf.reshape(excitation, [-1,out_dim,1,1])
  scale = input_x * excitation
  return scale,	[excitation]

def Squeeze_excitation_layer(input_x, out_dim, reg, ratio, SE_type):
  SE_type_lst = ['standard']
  assert(SE_type in SE_type_lst)
  if SE_type == 'standard':
    return SE_standard(input_x, out_dim, reg, ratio)
  
def batch_norm(inputs, training, data_format, renorm=False):
  """Performs a batch normalization using a standard set of parameters."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  #return tf.compat.v1.layers.batch_normalization(
  #    inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
  #    momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
  #    scale=True, training=training, fused=True)

  bn = tf.compat.v1.keras.layers.BatchNormalization(
    axis=1 if data_format == 'channels_first' else 3,
    momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True,
    beta_initializer=tf.zeros_initializer(),
    gamma_initializer=tf.ones_initializer(),
    moving_mean_initializer=tf.zeros_initializer(),
    moving_variance_initializer=tf.ones_initializer(), beta_regularizer=None,
    gamma_regularizer=None, beta_constraint=None, gamma_constraint=None,
    trainable=True, name=None, renorm=renorm,
    renorm_clipping=None, renorm_momentum=0.99, fused=None, virtual_batch_size=None,
    adjustment=None
  )
  output = bn (inputs, training)
  for u in bn.updates:
    tf.compat.v1.add_to_collection(tf.GraphKeys.UPDATE_OPS, u )
  return output 

def fixed_padding(inputs, kernel_size, data_format, dilation=1):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """

  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  if data_format == 'channels_first':
    padded_inputs = tf.pad(tensor=inputs,
                           paddings=[[0, 0], [0, 0], [pad_beg, pad_end],
                                     [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(tensor=inputs,
                           paddings=[[0, 0], [pad_beg, pad_end],
                                     [pad_beg, pad_end], [0, 0]])
  """

  if isinstance(kernel_size, int):
    kernel_size = (kernel_size, kernel_size)

  if isinstance(dilation, int):
    dilation = (dilation, dilation)

  pad_total_H = (kernel_size[0] - 1)*dilation[0]
  pad_beg_H   = pad_total_H // 2
  pad_end_H   = pad_total_H - pad_beg_H
  pad_total_W = (kernel_size[1] - 1)*dilation[1]
  pad_beg_W   = pad_total_W // 2
  pad_end_W   = pad_total_W - pad_beg_W
    
  if data_format == 'channels_first':
    padded_inputs = tf.pad(tensor=inputs,
                           paddings=[[0, 0], [0, 0], [pad_beg_H, pad_end_H],
                                     [pad_beg_W, pad_end_W]])
  else:
    padded_inputs = tf.pad(tensor=inputs,
                           paddings=[[0, 0], [pad_beg_H, pad_end_H],
                                     [pad_beg_W, pad_end_W], [0, 0]])
  return padded_inputs
  
  
def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format, dilation_rate=1, reg=None, use_bias=False):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  in_s = inputs.get_shape()
  
  if isinstance(strides, int):
    strides = (strides, strides)

  if strides > (1, 1):
    inputs = fixed_padding(inputs, kernel_size, data_format)

  conv = tf.keras.layers.Conv2D(
      filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == (1, 1) else 'VALID'), use_bias=use_bias,
      kernel_initializer=tf.compat.v1.keras.initializers.he_uniform(),
      bias_initializer=tf.compat.v1.keras.initializers.he_uniform(),
      data_format=data_format, dilation_rate=dilation_rate, kernel_regularizer=reg, bias_regularizer=reg)
  outputs = conv(inputs)

  for l in conv.losses:
    tf.compat.v1.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, l )
  
  # I think this is only needed when dilation rate > (1,1)
  outputs.set_shape([None, filters, None, in_s[3] // strides[1]])
  #print(outputs.get_shape)
  return outputs


################################################################################
# ResNet block definitions.
################################################################################

def _building_block_SE_v2(inputs, filters, training, projection_shortcut, strides,
                          data_format, block_kernel_sizes=3, block_dilation_rate=1,
                          reg=None, use_bias=False, SE_ratio=2, SE_type="standard", renorm=False, activation=tf.nn.relu):
  shortcut = inputs
  inputs = batch_norm(inputs, training, data_format, renorm=renorm)
  inputs = activation(inputs)

  excitation = None
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=block_kernel_sizes, strides=strides,
      data_format=data_format, dilation_rate=block_dilation_rate, reg=reg, use_bias=use_bias)

  inputs = batch_norm(inputs, training, data_format, renorm=renorm)
  inputs = activation(inputs)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=block_kernel_sizes, strides=1,
      data_format=data_format, dilation_rate=block_dilation_rate, reg=reg, use_bias=use_bias)
  
  if SE_ratio>0:
      inputs, excitation = Squeeze_excitation_layer(inputs, filters, reg, SE_ratio, SE_type)

  return inputs + shortcut, excitation

def block_layer(inputs, filters, bottleneck, block_fn, blocks, strides, training, name,
                data_format, block_kernel_sizes=3, block_dilation_rate=1, reg=None, use_bias=False, SE_type="standard", SE_ratio=None, renorm=False, activation=tf.nn.relu):

  filters_out = filters * 4 if bottleneck else filters

  def projection_shortcut(inputs):
    return conv2d_fixed_padding(
        inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
        data_format=data_format, dilation_rate=block_dilation_rate, reg=reg, use_bias=use_bias)
  excitation_block = []
  # Only the first block per block_layer uses projection_shortcut and strides
  inputs, excitation = block_fn(inputs, filters, training, projection_shortcut, strides, data_format,
                                block_kernel_sizes, block_dilation_rate, reg=reg, use_bias=use_bias, SE_ratio=SE_ratio, SE_type=SE_type,
                                activation=activation, renorm=renorm)
  excitation_block.append(excitation)
  
  for _ in range(1, blocks):
    inputs, excitation = block_fn(inputs, filters, training, None, 1, data_format,
                                  block_kernel_sizes, block_dilation_rate, reg=reg, use_bias=use_bias, SE_ratio=SE_ratio, SE_type=SE_type,
                                  activation=activation, renorm=renorm)
    excitation_block.append(excitation)

  #return (tf.identity(inputs, name), tf.identity(excitation_block, name + "_excitation_block")) if SE_ratio>0 else (tf.identity(inputs, name), [tf.zeros([1], tf.int32)])
  return (tf.identity(inputs, name), excitation_block) if SE_ratio>0 else (tf.identity(inputs, name), [tf.zeros([1], tf.int32)])

