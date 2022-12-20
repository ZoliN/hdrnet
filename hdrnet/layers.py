# Copyright 2016 Google Inc.
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

"""Shortcuts for some graph operators."""

import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import numpy as np

from hdrnet import hdrnet_ops



w_initializer = tf.initializers.VarianceScaling()
b_initializer = tf.initializers.zeros()

def conv(inputs, num_outputs, kernel_size, stride=1, rate=1,
    use_bias=True,
    batch_norm=False, is_training=False,
    activation_fn=tf.nn.relu, 
    scope=None, reuse=False):
  if batch_norm:
    normalizer_fn = tfv1.nn.fused_batch_norm
    b_init = None
  else:
    normalizer_fn = None
    if use_bias:
      b_init = b_initializer
    else:
      b_init = None

  output = tf.keras.layers.Conv2D(
      
      filters=num_outputs, kernel_size=kernel_size, 
      strides=stride, padding='SAME',
      dilation_rate=rate,
      kernel_initializer=w_initializer,
      bias_initializer=b_init,
      activation=activation_fn,
      name = scope
      )(inputs)


      # normalizer_fn=normalizer_fn,
      # normalizer_params={
      #   'center':True, 'is_training':is_training,
      #   'variables_collections':{
      #     'beta':[tfv1.GraphKeys.BIASES],
      #     'moving_mean':[tfv1.GraphKeys.MOVING_AVERAGE_VARIABLES],
      #     'moving_variance':[tfv1.GraphKeys.MOVING_AVERAGE_VARIABLES]},
      #   }, 


  return output


def fc(inputs, num_outputs,
    use_bias=True,
    batch_norm=False, is_training=False,
    activation_fn=tf.nn.relu, 
    scope=None):
  if batch_norm:
    normalizer_fn = tfv1.nn.fused_batch_norm
    b_init = None
  else:
    normalizer_fn = None
    if use_bias:
      b_init = b_initializer
    else:
      b_init = None

  output = tf.keras.layers.Dense(
      units=num_outputs,
      kernel_initializer=w_initializer,
      bias_initializer=b_init,
      activation=activation_fn,
      name = scope
      )(inputs)
  return output
      # normalizer_fn=normalizer_fn,
      # normalizer_params={
      #   'center':True, 'is_training':is_training,
      #   'variables_collections':{
      #     'beta':[tfv1.GraphKeys.BIASES],
      #     'moving_mean':[tfv1.GraphKeys.MOVING_AVERAGE_VARIABLES],
      #     'moving_variance':[tfv1.GraphKeys.MOVING_AVERAGE_VARIABLES]},
      #   }, 

# -----------------------------------------------------------------------------

# pylint: disable=redefined-builtin
def bilateral_slice(grid, guide, name=None):
  """Slices into a bilateral grid using the guide map.

  Args:
    grid: (Tensor) [batch_size, grid_h, grid_w, depth, n_outputs]
      grid to slice from.
    guide: (Tensor) [batch_size, h, w ] guide map to slice along.
    name: (string) name for the operation.
  Returns:
    sliced: (Tensor) [batch_size, h, w, n_outputs] sliced output.
  """

  with tfv1.name_scope(name):
    gridshape = grid.get_shape().as_list()
    if len(gridshape) == 6:
      _, _, _, _, n_out, n_in = gridshape
      grid = tf.concat(tf.unstack(grid, None, axis=5), 4)

    sliced = hdrnet_ops.bilateral_slice(grid, guide)

    if len(gridshape) == 6:
      sliced = tf.stack(tf.split(sliced, n_in, axis=3), axis=4)
    return sliced
# pylint: enable=redefined-builtin


def bilateral_slice_apply(grid, guide, input_image, has_offset=True, name=None):
  """Slices into a bilateral grid using the guide map.

  Args:
    grid: (Tensor) [batch_size, grid_h, grid_w, depth, n_outputs]
      grid to slice from.
    guide: (Tensor) [batch_size, h, w ] guide map to slice along.
    input_image: (Tensor) [batch_size, h, w, n_input] input data onto which to
      apply the affine transform.
    name: (string) name for the operation.
  Returns:
    sliced: (Tensor) [batch_size, h, w, n_outputs] sliced output.
  """

  with tfv1.name_scope(name):
    gridshape = grid.get_shape().as_list()
    if len(gridshape) == 6:
      gs = tf.shape(input=grid)
      _, _, _, _, n_out, n_in = gridshape
      grid = tf.reshape(grid, tf.stack([gs[0], gs[1], gs[2], gs[3], gs[4]*gs[5]]))
      # grid = tf.concat(tf.unstack(grid, None, axis=5), 4)

    sliced = hdrnet_ops.bilateral_slice_apply(grid, guide, input_image, has_offset=has_offset)
    return sliced
# pylint: enable=redefined-builtin

'''
# pylint: disable=redefined-builtin
def apply(sliced, input_image, has_affine_term=True, name=None):
  """Applies a sliced affined model to the input image.

  Args:
    sliced: (Tensor) [batch_size, h, w, n_output, n_input+1] affine coefficients
    input_image: (Tensor) [batch_size, h, w, n_input] input data onto which to
      apply the affine transform.
    name: (string) name for the operation.
  Returns:
    ret: (Tensor) [batch_size, h, w, n_output] the transformed data.
  Raises:
    ValueError: if the input is not properly dimensioned.
    ValueError: if the affine model parameter dimensions do not match the input.
  """

  with tf.name_scope(name):
    if len(input_image.get_shape().as_list()) != 4:
      raise ValueError('input image should have dims [b,h,w,n_in].')
    in_shape = input_image.get_shape().as_list()
    sliced_shape = sliced.get_shape().as_list()
    if (in_shape[:-1] != sliced_shape[:-2]):
      raise ValueError('input image and affine coefficients'
                       ' dimensions do not match: {} and {}'.format(
                       in_shape, sliced_shape))
    _, _, _, n_out, n_in = sliced.get_shape().as_list()
    if has_affine_term:
      n_in -= 1

    scale = sliced[:, :, :, :, :n_in]

    if has_affine_term:
      offset = sliced[:, :, :, :, n_in]

    out_channels = []
    for chan in range(n_out):
      ret = scale[:, :, :, chan, 0]*input_image[:, :, :, 0]
      for chan_i in range(1, n_in):
        ret += scale[:, :, :, chan, chan_i]*input_image[:, :, :, chan_i]
      if has_affine_term:
        ret += offset[:, :, :, chan]
      ret = tf.expand_dims(ret, 3)
      out_channels.append(ret)

    ret = tf.concat(out_channels, 3)

  return ret
# pylint: enable=redefined-builtin
'''

import time
def local_bilateral_slice(guide, coefs):
    """
        For each pixel of guide image we get affince coefs from a bilateral grid
    
    """
    bs = guide.get_shape()[0]
    spatial_bins = coefs.get_shape()[1]
    luma_bins = coefs.get_shape()[3]
    yw = guide.get_shape()[1]
    xw = guide.get_shape()[2]
    sp_y = tf.math.floordiv(guide.get_shape()[1], spatial_bins)
    sp_x = tf.math.floordiv(guide.get_shape()[2], spatial_bins)
    val  = tf.math.floordiv(256, luma_bins)

    guide = guide * 255
    guide_flat = tf.expand_dims(tf.reshape(guide, (bs, -1)),-1) # flat
    idx = tf.expand_dims(tf.tile(tf.expand_dims(tf.range(guide_flat.get_shape()[1]),0), [bs, 1]),-1) # index
    guide_indexed = tf.concat([tf.cast(idx, tf.float32), guide_flat],2)
    batch = []
    for i in xrange(bs):
        res = tf.map_fn(lambda x: 
        coefs[
            i,
            tf.math.floordiv(tf.math.floordiv(tf.cast(x[0],tf.int32), yw), sp_y),
            tf.math.floordiv(tf.math.floormod(tf.cast(x[0],tf.int32), xw), sp_x),
            tf.math.floordiv(tf.cast(x[1],tf.int32), val),
            :,:
        ]
        , guide_indexed[i], back_prop=True, parallel_iterations=4)
        res = tf.reshape(res, (1,yw,xw,3,4))
        batch.append(res)
    out = tf.cast(tf.concat(batch, axis=0), tf.float32)
    # example (2, 256, 256, 3, 4) 3 channels, 4 affince coefs(1 for channel and bias)
    return out

# pylint: disable=redefined-builtin
def apply(sliced, input_image, has_affine_term=True, name=None):
    """Applies a sliced affined model to the input image.

    Args:
      sliced: (Tensor) [batch_size, h, w, n_output, n_input+1] affine coefficients
      input_image: (Tensor) [batch_size, h, w, n_input] input data onto which to
        apply the affine transform.
      name: (string) name for the operation.
    Returns:
      ret: (Tensor) [batch_size, h, w, n_output] the transformed data.
    Raises:
      ValueError: if the input is not properly dimensioned.
      ValueError: if the affine model parameter dimensions do not match the input.
    """
    with tfv1.name_scope(name):
        if len(input_image.get_shape().as_list()) != 4:
            raise ValueError('input image should have dims [b,h,w,n_in].')
        in_shape = input_image.get_shape().as_list()
        sliced_shape = sliced.get_shape().as_list()
        if (in_shape[:-1] != sliced_shape[:-2]):
            raise ValueError('input image and affine coefficients'
                             ' dimensions do not match: {} and {}'.format(
                                 in_shape, sliced_shape))
        _, _, _, n_out, n_in = sliced.get_shape().as_list()
        if has_affine_term:
            n_in -= 1

        scale = sliced[:, :, :, :, :n_in]

        if has_affine_term:
            offset = sliced[:, :, :, :, n_in]

        # foreach chanel:
        #     a*x[0] + b*x[1] + c*x[2] + d = (h,w)
        #   res [ch1]   (x [ch1]   [aff11]           )   (x [ch1]   [aff21]           )
        #       [ch2] = (  [ch2] * [aff12] + [aff14] ) + (  [ch2] * [aff22] + [aff24] ) + same for aff3[1-4]
        #       [ch3]   (  [ch3]   [aff13]           )   (  [ch3]   [aff23]           )
        #

        out_channels = []
        for chan in range(n_out):
            ret = scale[:, :, :, chan, 0] * input_image[:, :, :, 0]
            for chan_i in range(1, n_in):
                ret += scale[:, :, :, chan, chan_i] * input_image[:, :, :, chan_i]
            if has_affine_term:
                ret += offset[:, :, :, chan]
            ret = tf.expand_dims(ret, 3)
            out_channels.append(ret)

        ret = tf.concat(out_channels, 3)
    return ret
# pylint: enable=redefined-builtin

def local_bilateral_slice_apply(grid, guide, input_image):
    sliced = local_bilateral_slice(guide, grid)
    out = apply(sliced, input_image, name='local_bilaterla_slice_apply')
    return out
