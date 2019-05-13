"""
Image dataset defined by using the tensorflow.data api. 
"""

import os.path as op
import numpy a np
import tensorflow as tf

image_paths = op.join('..', 'datasets', '128_part')
batch_size = 64

def write_tf_record_file(image_paths):
    image_ds = tf.data.Dataset.from_tensor_slices(image_paths).map(tf.io.read_file)
    tfrec = tf.data.experimental.TFRecordWriter('tf_128/images.tfrec')
    tfrec.write(image_ds)

def load_and_batch(tfrecord_paths):
    ds = tf.data.TFRecordDataset('tf_128/images.tfrec')
    ds = ds.apply(
        tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
        ds=ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

