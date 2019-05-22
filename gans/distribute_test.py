import os.path as op
from glob import glob

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers

def dense():
    """
    Dummy operation, to test our distribute strategy.
    Constituted of one trainable dense layer.
    """
    model = tf.keras.Sequential()
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model

def pixel_sum(image):
    """
    Performs the sum of all pixel vaues in the image. This is the function we 
    would like to approximate with our dense layer.
    """
    return tf.reduce_sum(image)

paths = glob(op.join('data128', '*.tfrecord'))

strategy = tf.distribute.MirroredStrategy()

# not very elegant, but the API doesn't provide anything for scanning the 
# length of a TFRecordDataset
BUFFER_SIZE = 0
print('computing buffer size')
for _ in tf.data.TFRecordDataset(paths):
    BUFFER_SIZE += 1
print('done')

BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
EPOCHS = 10

def parse_image(data):
    #imgs_raw = tf.io.read_file(data)
    imgs = tf.image.decode_image(data)
    imgs = tf.dtypes.cast(imgs, 'float32')
    imgs = (imgs - 127.5) / 127.5
    return imgs

def tf_parse_image(data):
    imgs = tf.py_function(
        parse_image,
        (data,),
        tf.float32)
    return tf.reshape(imgs, (128, 128, 3))

with strategy.scope():
    
    # create dataset from tfrecords
    print('creating dataset')
    ds = tf.data.TFRecordDataset(paths).map(tf_parse_image)
    print('batching dataset')
    ds = ds.batch(GLOBAL_BATCH_SIZE)
    print('making iterator')
    it = strategy.make_dataset_iterator(ds)
    
    # create loss object and function
    loss_object = tf.keras.losses.MeanSquaredError(
        reduction=tf.keras.losses.Reduction.NONE)
    
    def compute_loss(true, pred):
        per_example_loss = loss_object(true, pred)
        return tf.reduce_sum(per_example_loss) * (1. / GLOBAL_BATCH_SIZE)
    
    # model, optimizers
    model = dense()
    optimizer = tf.keras.optimizers.Adam(0.1)
    
    # train step
    def train_step(images):
        
        with tf.GradientTape() as tape:
            pred = model(images)
            loss = compute_loss(pred, pixel_sum(images))
        
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss
            
    def test(images):
        
        pred = model(images)
        loss = compute_loss(pred, pixel_sum(images))
    
    @tf.function
    def distributed_train():
        return strategy.experimental_run(train_step, it)
    
    def run():
        for epoch in range(EPOCHS):
            it.initialize()
            
            for _ in range(int(BUFFER_SIZE / GLOBAL_BATCH_SIZE)):
                loss = distributed_train()
            
            print("Epoch : {}, Loss : {}".format(epoch+1, loss))
    
    


































