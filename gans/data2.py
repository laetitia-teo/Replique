# convert a directory of images to a list of TFRecords

import os.path as op
import os
from glob import glob

import numpy as np
import cv2
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

filenames = glob(op.join('..', 'datasets', '128_part', '*.jpg'))

image_count = len(filenames)

# The following functions can be used to convert a value to a type compatible
# with tf.Example.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

n_obs = int(1e4)

f0 = np.random.choice([False, True], n_obs)

f1 = np.random.randint(0, 5, n_obs)

strings = np.array([b'cat', b'dog', b'chicken', b'horse', b'goat'])
f2 = strings[f1]

f3 = np.random.randn(n_obs)

def serialize_example(f0, f1, f2, f3):
    feature = {
        'f0': _int64_feature(f0),
        'f1': _int64_feature(f1),
        'f2': _bytes_feature(f2),
        'f3': _float_feature(f3),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

example_obs = []

serialized_example = serialize_example(False, 4, b'goat', 0.9876)
example_proto = tf.train.Example.FromString(serialized_example)

features_ds = tf.data.Dataset.from_tensor_slices((f0, f1, f2, f3))
# Use `take(1)` to only pull one example from the dataset.

for f0,f1,f2,f3 in features_ds.take(1):
    print(f0)
    print(f1)
    print(f2)
    print(f3)

def tf_serialize_example(f0, f1, f2, f3):
    tf_string = tf.py_function(
        serialize_example,
        (f0, f1, f2, f3),
        tf.string)
    return tf.reshape(tf_string, ())

#print(tf_serialize_example(f0, f1, f2, f3))

def load_image(filename):
    img_raw = tf.io.read_file(filename)
    return img_raw

def create_tfrecords(directory, batch_size, write_dir='data128'):
    """
    Scans the data directory and creates a set of TFRecords files corresponding
    to the data. Each TFRecord file must contain less than 200MB worth of jpeg
    data, and the number of files contained in a TFRecord must be a multiple of 
    batch_size.
    """
    filenames = glob(op.join(directory, '*.jpg'))
    size_one = op.getsize(filenames[0])
    if size_one * batch_size > 2 * 1e6:
        raise Exception('resulting files will be too big, choose a smaller'
            + 'batch size.')
    n_batches = int(2*1e6 / (size_one*batch_size)) # number of batches in a record
    n_images = len(filenames)
    chunk_size = n_batches * batch_size
    n_chunks = int(n_images / chunk_size)
    i = 0
    while (i+1) < n_chunks:
        print('chunk %s of %s' % (i, n_chunks))
        current_chunk = filenames[int(i*chunk_size):int((i+1)*chunk_size)]
        # datasets
        path_ds = tf.data.Dataset.from_tensor_slices(current_chunk)
        raw_image_ds = path_ds.map(load_image, 
                                   num_parallel_calls=AUTOTUNE)
        # write TFRecord
        write_path = op.join(write_dir, str(i) + '.tfrecord')
        writer = tf.data.experimental.TFRecordWriter(write_path)
        writer.write(raw_image_ds)
        i += 1














































