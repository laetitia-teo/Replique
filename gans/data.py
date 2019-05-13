import os.path as op
import os
from glob import glob

import numpy as np
import cv2
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

filenames = glob(op.join('..', 'datasets', '128', '*.jpg'))

image_count = len(filenames)

def load_image(filename):
    img_raw = tf.io.read_file(filename)
    img = tf.image.decode_image(img_raw)
    img = tf.dtypes.cast(img, dtype=tf.float32)
    print(img.dtype)
    img = img / 255.0
    return img

path_ds = tf.data.Dataset.from_tensor_slices(filenames)
image_ds = path_ds.map(load_image, num_parallel_calls=AUTOTUNE)

BATCH_SIZE = 32

ds = image_ds.shuffle(buffer_size=image_count)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(buffer_size=AUTOTUNE)
ds
