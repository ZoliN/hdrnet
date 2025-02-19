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

"""Defines computation graphs."""

import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import numpy as np
import os

from hdrnet.layers import (conv, fc, bilateral_slice_apply)
from hdrnet.hdrnet_ops import (useGPU)

numImgChs = 1

__all__ = [
  'HDRNetCurves',
  'HDRNetPointwiseNNGuide',
  'HDRNetGaussianPyrNN',
]


class HDRNetCurves(object):
  """Main model, as submitted in January 2017.
  """
  
  @classmethod
  def n_out(cls):
    return numImgChs

  @classmethod
  def n_in(cls):
    return numImgChs+1

  @classmethod
  def inference(cls, lowres_input, fullres_input, params,
                is_training=False):

    with tfv1.variable_scope('coefficients'):
      bilateral_coeffs = cls._coefficients(lowres_input, params, is_training)
      tfv1.add_to_collection('bilateral_coefficients', bilateral_coeffs)

    with tfv1.variable_scope('guide'):
      guide = cls._guide(fullres_input, params, is_training)
      tfv1.add_to_collection('guide', guide)

    with tfv1.variable_scope('output'):
      output = cls._output(
          fullres_input, guide, bilateral_coeffs)
      tfv1.add_to_collection('output', output)

    return output

  @classmethod
  def _coefficients(cls, input_tensor, params, is_training):
    bs = input_tensor.get_shape().as_list()[0] or -1
    gd = int(params['luma_bins'])
    cm = int(params['channel_multiplier'])
    spatial_bin = int(params['spatial_bin'])

    # -----------------------------------------------------------------------
    with tfv1.variable_scope('splat'):
      n_ds_layers = int(np.log2(params['net_input_size']/spatial_bin))

      current_layer = input_tensor
      for i in range(n_ds_layers):
        if i > 0:  # don't normalize first layer
          use_bn = params['batch_norm']
        else:
          use_bn = False
        current_layer = conv(current_layer, cm*(2**i)*gd, 3, stride=2,
                             batch_norm=use_bn, is_training=is_training,
                             scope='conv{}'.format(i+1))

      splat_features = current_layer
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    with tfv1.variable_scope('global'):
      n_global_layers = int(np.log2(spatial_bin/4))  # 4x4 at the coarsest lvl

      current_layer = splat_features
      for i in range(2):
        current_layer = conv(current_layer, 8*cm*gd, 3, stride=2,
            batch_norm=params['batch_norm'], is_training=is_training,
            scope="conv{}".format(i+1))
      _, lh, lw, lc = current_layer.get_shape().as_list()
      current_layer = tf.reshape(current_layer, [bs, lh*lw*lc])

      current_layer = fc(current_layer, 32*cm*gd, 
                         batch_norm=params['batch_norm'], is_training=is_training,
                         scope="fc1")
      current_layer = fc(current_layer, 16*cm*gd, 
                         batch_norm=params['batch_norm'], is_training=is_training,
                         scope="fc2")
      # don't normalize before fusion
      current_layer = fc(current_layer, 8*cm*gd, activation_fn=None, scope="fc3")
      global_features = current_layer
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    with tfv1.variable_scope('local'):
      current_layer = splat_features
      current_layer = conv(current_layer, 8*cm*gd, 3, 
                           batch_norm=params['batch_norm'], 
                           is_training=is_training,
                           scope='conv1')
      # don't normalize before fusion
      current_layer = conv(current_layer, 8*cm*gd, 3, activation_fn=None,
                           use_bias=False, scope='conv2')
      grid_features = current_layer
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    with tfv1.name_scope('fusion'):
      fusion_grid = grid_features
      fusion_global = tf.reshape(global_features, [bs, 1, 1, 8*cm*gd])
      fusion = tf.nn.relu(fusion_grid+fusion_global)
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    with tfv1.variable_scope('prediction'):
      current_layer = fusion
      current_layer = conv(current_layer, gd*cls.n_out()*cls.n_in(), 1,
                                  activation_fn=None, scope='conv1')

      with tfv1.name_scope('unroll_grid'):
        current_layer = tf.stack(
            tf.split(current_layer, cls.n_out()*cls.n_in(), axis=3), axis=4)
        current_layer = tf.stack(
            tf.split(current_layer, cls.n_in(), axis=4), axis=5)
      tfv1.add_to_collection('packed_coefficients', current_layer)
    # -----------------------------------------------------------------------

    return current_layer

  @classmethod
  def _guide(cls, input_tensor, params, is_training):
    npts = 16  # number of control points for the curve
    nchans = input_tensor.get_shape().as_list()[-1]

    guidemap = input_tensor

    # Color space change
    idtity = np.identity(nchans, dtype=np.float32) + np.random.randn(1).astype(np.float32)*1e-4
    ccm = tfv1.get_variable('ccm', dtype=tf.float32, initializer=idtity)
    with tfv1.name_scope('ccm'):
      ccm_bias = tfv1.get_variable('ccm_bias', shape=[nchans,], dtype=tf.float32, initializer=tfv1.constant_initializer(0.0))

      guidemap = tf.matmul(tf.reshape(input_tensor, [-1, nchans]), ccm)
      guidemap = tf.nn.bias_add(guidemap, ccm_bias, name='ccm_bias_add')

      guidemap = tf.reshape(guidemap, tf.shape(input=input_tensor))

    # Per-channel curve
    with tfv1.name_scope('curve'):
      shifts_ = np.linspace(0, 1, npts, endpoint=False, dtype=np.float32)
      shifts_ = shifts_[np.newaxis, np.newaxis, np.newaxis, :]
      shifts_ = np.tile(shifts_, (1, 1, nchans, 1))

      guidemap = tf.expand_dims(guidemap, 4)
      shifts = tfv1.get_variable('shifts', dtype=tf.float32, initializer=shifts_)

      slopes_ = np.zeros([1, 1, 1, nchans, npts], dtype=np.float32)
      slopes_[:, :, :, :, 0] = 1.0
      slopes = tfv1.get_variable('slopes', dtype=tf.float32, initializer=slopes_)

      guidemap = tf.reduce_sum(input_tensor=slopes*tf.nn.relu(guidemap-shifts), axis=[4])

    guidemap = tf.keras.layers.Conv2D(
        filters=1, 
        kernel_size=1, 
        kernel_initializer=tfv1.constant_initializer(1.0/nchans),
        bias_initializer=tfv1.constant_initializer(0),
        activation=None,
        name='channel_mixing'
        )(guidemap)

    guidemap = tf.clip_by_value(guidemap, 0, 1)
    guidemap = tf.squeeze(guidemap, axis=[3,])

    return guidemap

  @classmethod
  def _output(cls, im, guide, coeffs):
    with tf.device('/gpu:0' if useGPU else '/cpu:0'):
      out = bilateral_slice_apply(coeffs, guide, im, has_offset=True, name='slice')
    return out


class HDRNetPointwiseNNGuide(HDRNetCurves):
  """Replaces the pointwise curves in the guide by a pointwise neural net.
  """
  @classmethod
  def _guide(cls, input_tensor, params, is_training):
    n_guide_feats = params['guide_complexity']
    guidemap = conv(input_tensor, n_guide_feats, 1, 
                    batch_norm=True, is_training=is_training,
                    scope='conv1')
    guidemap = conv(guidemap, 1, 1, activation_fn=tf.nn.sigmoid, scope='conv2')
    guidemap = tf.squeeze(guidemap, axis=[3,])
    return guidemap


class HDRNetGaussianPyrNN(HDRNetPointwiseNNGuide):
  """Replace input to the affine model by a pyramid
  """
  @classmethod
  def n_scales(cls):
    return 5

  @classmethod
  def n_out(cls):
    return numImgChs*cls.n_scales()

  @classmethod
  def n_in(cls):
    return numImgChs+1

  @classmethod
  def inference(cls, lowres_input, fullres_input, params,
                is_training=False):

    with tfv1.variable_scope('coefficients'):
      bilateral_coeffs = cls._coefficients(lowres_input, params, is_training)
      tfv1.add_to_collection('bilateral_coefficients', bilateral_coeffs)

    with tfv1.variable_scope('multiscale'):
      multiscale = cls._multiscale_input(fullres_input)
      for m in multiscale:
        tfv1.add_to_collection('multiscale', m)

    with tfv1.variable_scope('guide'):
      guide = cls._guide(multiscale, params, is_training)
      for g in guide:
        tfv1.add_to_collection('guide', g)

    with tfv1.variable_scope('output'):
      output = cls._output(multiscale, guide, bilateral_coeffs)
      tfv1.add_to_collection('output', output)

    return output

  @classmethod
  def _multiscale_input(cls, fullres_input):
    full_sz = tf.shape(input=fullres_input)[1:3]
    sz = full_sz

    current_level = fullres_input
    lvls = [current_level]
    for lvl in range(cls.n_scales()-1):
      sz = tf.cast( sz / 2 , dtype=tf.int32)
      current_level = tf.image.resize(
          current_level, sz, tf.image.ResizeMethod.BILINEAR)
      lvls.append(current_level)
    return lvls

  @classmethod
  def _guide(cls, multiscale, params, is_training):
    guide_lvls = []
    for il, lvl in enumerate(multiscale):
      with tfv1.variable_scope('level_{}'.format(il)):
        guide_lvl = HDRNetPointwiseNNGuide._guide(lvl, params, is_training)
      guide_lvls.append(guide_lvl)
    return guide_lvls

  @classmethod
  def _output(cls, lvls, guide_lvls, coeffs):
    for il, (lvl, guide_lvl) in enumerate(reversed(list(zip(lvls, guide_lvls)))):
      c = coeffs[:, :, :, :, il*numImgChs:(il+1)*numImgChs, :]
      out_lvl = HDRNetPointwiseNNGuide._output(lvl, guide_lvl, c)

      if il == 0:
        current = out_lvl
      else:
        sz = tf.shape(input=out_lvl)[1:3]
        current = tf.image.resize(current, sz, tf.image.ResizeMethod.BILINEAR)
        current = tf.add(current, out_lvl)

    return current


