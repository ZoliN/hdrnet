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

"""Data pipeline for HDRnet."""

import abc
import json
import logging
import magic
import numpy as np
import os
import random

import tensorflow as tf

log = logging.getLogger("root")
log.setLevel(logging.INFO)

__all__ = [
  'ImageFilesDataPipeline',
  'StyleTransferDataPipeline',
  'HDRpDataPipeline',
  ]

def check_dir(dirname):
  fnames = os.listdir(dirname)
  if not os.path.isdir(dirname):
    log.error("Training dir {} does not exist".format(dirname))
    return False
  if not "filelist.txt" in fnames:
    log.error("Training dir {} does not containt 'filelist.txt'".format(dirname))
    return False
  if not "input" in fnames:
    log.error("Training dir {} does not containt 'input' folder".format(dirname))
  if not "output" in fnames:
    log.error("Training dir {} does not containt 'output' folder".format(dirname))

  return True


class DataPipeline(object):
  """Abstract operator to stream data to the network.

  Attributes:
    path:
    batch_size: number of sampes per batch.
    capacity: max number of samples in the data queue.
    pass
    min_after_dequeue: minimum number of image used for shuffling the queue.
    shuffle: random shuffle the samples in the queue.
    nthreads: number of threads use to fill up the queue.
    samples: a dict of tensors containing a batch of input samples with keys:
      'image_input' (the source image),
      'image_output' (the filtered image to match),
      'guide_image' (the image used to slice in the grid).
  """

  __metaclass__ = abc.ABCMeta

  def __init__(self, path, batch_size=32,
               capacity=16, 
               min_after_dequeue=4,
               output_resolution=[1080, 1920],
               shuffle=False,
               fliplr=False, 
               flipud=False,
               rotate=False,
               random_crop=False,
               params=None,
               nthreads=1, 
               num_epochs=None):
    self.path = path
    self.batch_size = batch_size
    self.capacity = capacity
    self.min_after_dequeue = min_after_dequeue
    self.shuffle = shuffle
    self.nthreads = nthreads
    self.num_epochs = num_epochs
    self.params = params

    self.output_resolution = output_resolution

    # Data augmentation
    self.fliplr = fliplr
    self.flipud = flipud
    self.rotate = rotate
    self.random_crop = random_crop

    sample = self._produce_one_sample()
    self.samples = self._batch_samples(sample)

  @abc.abstractmethod
  def _produce_one_sample(self):
    pass

  def _batch_samples(self, ds):
    """Batch several samples together."""

    # Batch and shuffle
    if self.shuffle:
      ds = ds.shuffle(self.min_after_dequeue)
    ds = ds.batch(self.batch_size).repeat()
    '''
    if self.shuffle:
      samples = tf.train.shuffle_batch(
          sample,
          batch_size=self.batch_size,
          num_threads=self.nthreads,
          capacity=self.capacity,
          min_after_dequeue=self.min_after_dequeue)
    else:
      samples = tf.train.batch(
          sample,
          batch_size=self.batch_size,
          num_threads=self.nthreads,
          capacity=self.capacity)
    '''
    return ds

  def _augment_data(self, inout, nchan=6):
    """Flip, crop and rotate samples randomly."""

    with tf.compat.v1.name_scope('data_augmentation'):
      if self.fliplr:
        inout = tf.image.random_flip_left_right(inout, seed=1234)
      if self.flipud:
        inout = tf.image.random_flip_up_down(inout, seed=3456)
      if self.rotate:
        angle = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32, seed=4567)
        inout = tf.case([(tf.equal(angle, 1), lambda: tf.image.rot90(inout, k=1)),
                         (tf.equal(angle, 2), lambda: tf.image.rot90(inout, k=2)),
                         (tf.equal(angle, 3), lambda: tf.image.rot90(inout, k=3))],
                        lambda: inout)

      inout.set_shape([None, None, nchan])

      with tf.compat.v1.name_scope('crop'):
        shape = tf.shape(input=inout)
        new_height = tf.cast(self.output_resolution[0], dtype=tf.int32)
        new_width = tf.cast(self.output_resolution[1], dtype=tf.int32)
        height_ok = tf.compat.v1.assert_less_equal(new_height, shape[0])
        width_ok = tf.compat.v1.assert_less_equal(new_width, shape[1])
        with tf.control_dependencies([height_ok, width_ok]):
          if self.random_crop:
            inout = tf.image.random_crop(
                inout, tf.stack([new_height, new_width, nchan]))
          else:
            height_offset = tf.cast((shape[0]-new_height)/2, dtype=tf.int32)
            width_offset = tf.cast((shape[1]-new_width)/2, dtype=tf.int32)
            inout = tf.image.crop_to_bounding_box(
                inout, height_offset, width_offset,
                new_height, new_width)

      inout.set_shape([None, None, nchan])
      inout = tf.image.resize(
          inout, [self.output_resolution[0], self.output_resolution[1]])
      fullres = inout

      with tf.compat.v1.name_scope('resize'):
        new_size = 256
        inout = tf.image.resize(
            inout, [new_size, new_size],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

      return fullres, inout


class ImageFilesDataPipeline(DataPipeline):
  """Pipeline to process pairs of images from a list of image files.

  Assumes path contains:
    - a file named 'filelist.txt' listing the image names.
    - a subfolder 'input'
    - a subfolder 'output'
  """

  def _produce_one_sample(self):
    dirname = os.path.dirname(self.path)
    if not check_dir(dirname):
      raise ValueError("Invalid data path.")
    with open(self.path, 'r') as fid:
      flist = [l.strip() for l in fid]

    if self.shuffle:
      random.shuffle(flist)

    print(flist)
    input_files = [os.path.join(dirname, 'input', f) for f in flist]
    output_files = [os.path.join(dirname, 'output', f) for f in flist]

    self.nsamples = len(input_files)
    print(self.nsamples)
    print(self.num_epochs)
    #input_queue, output_queue = tf.train.slice_input_producer(
    #    [input_files, output_files], shuffle=self.shuffle,
    #    seed=42, num_epochs=self.num_epochs)

    if '16-bit' in magic.from_file(input_files[0]):
      input_dtype = tf.uint16
      input_wl = 65535.0
    else:
      input_wl = 255.0
      input_dtype = tf.uint8
    if '16-bit' in magic.from_file(output_files[0]):
      output_dtype = tf.uint16
      output_wl = 65535.0
    else:
      output_wl = 255.0
      output_dtype = tf.uint8

    
    all_image_paths = list(zip(input_files,output_files))
    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)

    def preproc(x):
      input_file = tf.io.read_file(x[0])
      output_file = tf.io.read_file(x[1])

      if os.path.splitext(input_files[0])[-1] == '.jpg': 
        im_input = tf.image.decode_jpeg(input_file, channels=3)
      else:
        im_input = tf.image.decode_png(input_file, dtype=input_dtype, channels=3)

      if os.path.splitext(output_files[0])[-1] == '.jpg': 
        im_output = tf.image.decode_jpeg(output_file, channels=3)
      else:
        im_output = tf.image.decode_png(output_file, dtype=output_dtype, channels=3)

      # normalize input/output
      sample = {}
      with tf.compat.v1.name_scope('normalize_images'):
        im_input = tf.cast(im_input, dtype=tf.float32)/input_wl
        im_output = tf.cast(im_output, dtype=tf.float32)/output_wl

      inout = tf.concat([im_input, im_output], 2)
      fullres, inout = self._augment_data(inout, 6)

      sample = {}

      sample['lowres_input'] = inout[:, :, :3]
      sample['lowres_output'] = inout[:, :, 3:]
      sample['image_input'] = fullres[:, :, :3]
      sample['image_output'] = fullres[:, :, 3:]

      return sample

    ds = path_ds.map(preproc, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return ds

class HDRpDataPipeline(DataPipeline):
  """Pipeline to process HDR+ dumps

  Assumes :
    - path points to a txt file listing the image path, relative to
      the root of 'path'.
  """

  def _produce_one_sample(self):
    root = os.path.dirname(os.path.abspath(self.path))
    # TODO: check dir structure
    with open(self.path, 'r') as fid:
      flist = [l.strip() for l in fid]
    input_files = [os.path.join(root, f) for f in flist if '.tfrecords' in f]

    if self.shuffle:
      random.shuffle(input_files)

    self._reader = RecordReader(input_files, shuffle=self.shuffle, num_epochs=self.num_epochs)
    sample = self._reader.read()

    self.nsamples = len(input_files)

    # white-level
    input_wl = 32767.0
    output_wl = 255.0

    # normalize input/output
    with tf.compat.v1.name_scope('normalize_images'):
      im_input = sample['image_input']
      im_input = tf.cast(im_input, dtype=tf.float32)/input_wl

      im_output = sample['image_output']
      im_output = tf.cast(im_output, dtype=tf.float32)/output_wl

    inout = tf.concat([im_input, im_output], 2)
    fullres, inout = self._augment_data(inout, 6)

    sample['lowres_input'] = inout[:, :, :3]
    sample['lowres_output'] = inout[:, :, 3:]
    sample['image_input'] = fullres[:, :, :3]
    sample['image_output'] = fullres[:, :, 3:]

    return sample


class StyleTransferDataPipeline(DataPipeline):
  def _produce_one_sample(self):
    # TODO: check dir structure
    with open(os.path.join(self.path, 'filelist.txt'), 'r') as fid:
      flist = [l.strip() for l in fid]

    with open(os.path.join(self.path, 'targets.txt'), 'r') as fid:
      tlist = [l.strip() for l in fid]

    input_files = []
    model_files = []
    output_files = []
    for f in flist:
      for t in tlist:
        input_files.append(os.path.join(self.path, 'input', f))
        model_files.append(os.path.join(self.path, 'input', t+'.png'))
        output_files.append(os.path.join(self.path, 'output', t, f))

    self.nsamples = len(input_files)

    input_queue, model_queue, output_queue = tf.compat.v1.train.slice_input_producer(
        [input_files, model_files, output_files], capacity=1000, shuffle=self.shuffle,
        num_epochs=self.num_epochs, seed=1234)

    input_wl = 255.0
    input_dtype = tf.uint8

    input_file = tf.io.read_file(input_queue)
    model_file = tf.io.read_file(model_queue)
    output_file = tf.io.read_file(output_queue)

    im_input = tf.image.decode_png(input_file, dtype=input_dtype, channels=3)
    im_model = tf.image.decode_png(model_file, dtype=input_dtype, channels=3)
    im_output = tf.image.decode_png(output_file, dtype=input_dtype, channels=3)

    # normalize input/output
    with tf.compat.v1.name_scope('normalize_images'):
      im_input = tf.cast(im_input, dtype=tf.float32)/input_wl
      im_model = tf.cast(im_model, dtype=tf.float32)/input_wl
      im_output = tf.cast(im_output, dtype=tf.float32)/input_wl

    inout = tf.concat([im_input, im_output], 2)
    fullres, inout = self._augment_data(inout, 6)

    mdl = tf.image.resize(im_model, tf.shape(input=inout)[:2])

    sample = {}
    sample['lowres_input'] = tf.concat([inout[:, :, :3], mdl], 2) 
    sample['lowres_output'] = inout[:, :, 3:]
    fullres_mdl = tf.image.resize(im_model, tf.shape(input=fullres)[:2])
    sample['image_input'] = tf.concat([fullres[:, :, :3], fullres_mdl], 2) 
    sample['image_output'] = fullres[:, :, 3:]
    return sample


# __all__ = ['RecordWriter', 'RecordReader',]


# Index and range for type serialization
TYPEMAP = {
    np.dtype(np.uint8): 0,
    np.dtype(np.int16): 1,
    np.dtype(np.float32): 2,
    np.dtype(np.int32): 3,
}

REVERSE_TYPEMAP = {
    0: tf.uint8,
    1: tf.int16,
    2: tf.float32,
    3: tf.int32,
}

class RecordWriter(object):
  """Writes input/output pairs to .tfrecords.

  Attributes:
    output_dir: directory where the .tfrecords are saved.
  """

  def __init__(self, output_dir, records_per_file=500, prefix=''):
    self.output_dir = output_dir
    self.records_per_file = records_per_file
    self.written = 0
    self.nfiles = 0
    self.prefix = prefix

    # internal state
    self._writer = None
    self._fname = None

  def _get_new_filename(self):
    self.nfiles += 1
    return os.path.join(self.output_dir, '{}{:06d}.tfrecords'.format(self.prefix, self.nfiles))

  def write(self, data):
    """Write the arrays in data to the currently opened tfrecords.

    Args:
      data: a dict of numpy arrays.

    Returns:
      The filename just written to.
    """

    if self.written % self.records_per_file == 0:
      self.close()
      self._fname = self._get_new_filename()
      self._writer = tf.io.TFRecordWriter(self._fname)

    feature = {}
    for k in data.keys():
      feature[k] = self._bytes_feature(data[k].tobytes())
      feature[k+'_sz'] = self._int64_list_feature(data[k].shape)
      feature[k+'_dtype'] = self._int64_feature(TYPEMAP[data[k].dtype])

    example = tf.train.Example(
        features=tf.train.Features(feature=feature))

    self._writer.write(example.SerializeToString())
    self.written += 1
    return self._fname

  def close(self):
    if self._writer is not None:
      self._writer.close()

  def _bytes_feature(self, value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

  def _int64_feature(self, value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

  def _int64_list_feature(self, value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


class RecordReader(object):
  """Produces a queue of input/output data from .tfrecords.

  Attributes:
    shapes: dict of list of integers. The shape of each feature in the record.
  """

  FEATURES = ['image_input', 'image_output']
  NDIMS = [3, 3]

  def __init__(self, fnames, shuffle=True, num_epochs=None):
    """Init from a list of filenames to enqueue.

    Args:
      fnames: list of .tfrecords filenames to enqueue.
      shuffle: if true, shuffle the list at each epoch
    """
    self._fnames = fnames
    self._fname_queue = tf.compat.v1.train.string_input_producer(
        self._fnames,
        capacity=1000,
        shuffle=shuffle,
        num_epochs=num_epochs,
        shared_name='input_files')
    self._reader = tf.compat.v1.TFRecordReader()

    # Read first record to initialize the shape parameters
    with tf.Graph().as_default():
      fname_queue = tf.compat.v1.train.string_input_producer(self._fnames)
      reader = tf.compat.v1.TFRecordReader()
      _, serialized = reader.read(fname_queue)
      shapes = self._parse_shape(serialized)
      dtypes = self._parse_dtype(serialized)

      config = tf.compat.v1.ConfigProto()
      config.gpu_options.allow_growth = True

      with tf.compat.v1.Session(config=config) as sess:
        coord = tf.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)

        self.shapes = sess.run(shapes)
        self.shapes = {k: self.shapes[k+'_sz'].tolist() for k in self.FEATURES}

        self.dtypes = sess.run(dtypes)
        self.dtypes = {k: REVERSE_TYPEMAP[self.dtypes[k+'_dtype'][0]] for k in self.FEATURES}

        coord.request_stop()
        coord.join(threads)

  def read(self):
    """Read and parse a single sample from the input queue.

    Returns:
      sample: a dict with keys in self.FEATURES. Each entry holds the
      corresponding Tensor.
    """
    _, serialized = self._reader.read(self._fname_queue)
    sample = self._parse_example(serialized)
    return sample

  def _get_dtype_features(self):
    ret = {}
    for i, f in enumerate(self.FEATURES):
      ret[f+'_dtype'] = tf.io.FixedLenFeature([1,], tf.int64)
    return ret

  def _get_sz_features(self):
    ret = {}
    for i, f in enumerate(self.FEATURES):
      ret[f+'_sz'] = tf.io.FixedLenFeature([self.NDIMS[i],], tf.int64)
    return ret

  def _get_data_features(self):
    ret = {}
    for f in self.FEATURES:
      ret[f] = tf.io.FixedLenFeature([], tf.string)
    return ret

  def _parse_shape(self, serialized):
    sample = tf.io.parse_single_example(serialized=serialized,
                                     features=self._get_sz_features())
    return sample

  def _parse_dtype(self, serialized):
    sample = tf.io.parse_single_example(serialized=serialized,
                                     features=self._get_dtype_features())
    return sample

  def _parse_example(self, serialized):
    """Unpack a serialized example to Tensor."""
    feats = self._get_data_features()
    sz_feats = self._get_sz_features()
    for s in sz_feats:
      feats[s] = sz_feats[s]
    sample = tf.io.parse_single_example(serialized=serialized, features=feats)

    data = {}
    for i, f in enumerate(self.FEATURES):
      s = tf.cast(sample[f+'_sz'], dtype=tf.int32)

      data[f] = tf.io.decode_raw(sample[f], self.dtypes[f], name='decode_{}'.format(f))
      data[f] = tf.reshape(data[f], s)
      
    return data
