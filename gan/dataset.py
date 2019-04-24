import numpy as np
import tensorflow as tf
import cv2
from glob import glob
import os.path as op
from tqdm import tqdm

class MnistDataset():
    """
    Handler class for the MNIST  datset.
    Supports batch iteration as iterator.
    
    If n_iter is not specified, the iterator only does one pass on the data.
    """
    def __init__(self, batch_size, n_iter=None):
        mnist = tf.keras.datasets.mnist
        (Xtrain, ytrain), (Xtest, ytest) = mnist.load_data()
        X = np.concatenate((Xtrain, Xtest), axis=0)
        self.X = np.expand_dims(X, axis=-1).astype('float32') # add channel
        self.X = (self.X - 127.5)/127.5
        self.batch_size = batch_size
        self.size = len(X)
        if n_iter:
            self.n_iter = n_iter # max number of iterations
        else:
            self.n_iter = int(len(X)/self.batch_size)

    def __iter__(self):
        self.batch_idx = 0
        return self
    
    def __next__(self):
        if self.batch_idx >= self.n_iter:
            raise StopIteration
        inf = self.batch_idx * self.batch_size % self.size
        sup = (self.batch_idx + 1) * self.batch_size % self.size
        qinf = self.batch_idx * self.batch_size // self.size
        qsup = (self.batch_idx + 1) * self.batch_size // self.size
        self.batch_idx += 1
        #print(qinf, qsup)
        if qinf == qsup:
            return self.X[inf:sup]
        else:
            return np.concatenate((self.X[inf:], self.X[:sup]), axis=0)
    
    def __getattr__(self, attr):
        """Supports numpy array indexing""" # or does it ?
        return self.X[attr]
    
    def iterations(self, n_iter):
        """Change max number of iterations"""
        self.n_iter = n_iter

class ImgDataset():
    """
    Class for handling an image dataset.
    Supports batch iteration.
    """
    # TODO memory management, dynamic loading ?
    # TODO create interface for unifying all types of datasets
    # TODO dynamic dataset from images ? May slow down training too much
    
    def __init__(self, 
                 path,
                 batch_size=64,
                 n_iter=None,
                 color=False):
        self.path = path
        self.batch_size = batch_size
        self.color = color
        assert(self.scan_size()) # stop if dataset is too big
        self.load_data()
        self.size = len(self.X)
        if n_iter:
            self.n_iter = n_iter # max number of iterations
        else:
            self.n_iter = int(len(self.X)/self.batch_size)
    
    def __iter__(self):
        self.batch_idx = 0
        return self
    
    def __next__(self): # TODO chech this
        if self.batch_idx >= self.n_iter:
            raise StopIteration
        inf = self.batch_idx * self.batch_size % self.size
        sup = (self.batch_idx + 1) * self.batch_size % self.size
        qinf = self.batch_idx * self.batch_size // self.size
        qsup = (self.batch_idx + 1) * self.batch_size // self.size
        self.batch_idx += 1
        #print(qinf, qsup)
        if qinf == qsup:
            return self.X[inf:sup]
        else:
            return np.concatenate((self.X[inf:], self.X[:sup]), axis=0)
    
    def iterations(self, n_iter):
        """Change max number of iterations"""
        self.n_iter = n_iter
        
    def scan_size(self):
        """
        Returns True if the dataset is smaller than 1GB.
        """
        max_memory = 10e9/4 # because 32-bit floats will be used
        memory = 0
        for f in sorted(glob(op.join(self.path, '*.jpg'))):
            img = cv2.imread(f, int(self.color))
            m = 1
            for dim in img.shape:
                m *= dim
            memory += m
        print('size is %s bytes' % memory)
        return memory <= max_memory
    
    def load_data(self):
        """
        Loads the image data into X.
        """
        filenames = sorted(glob(op.join(self.path, '*.jpg')))
        length = len(filenames)
        # we assume all images have the same dimensions
        shape = cv2.imread(filenames[0], int(self.color)).shape
        if not self.color:
            shape += (1,) # add additionnal channel for black and white
        X = []
        for f in tqdm(filenames):
            img = cv2.imread(f, int(self.color))
            if not self.color:
                img = np.expand_dims(img, axis=-1)
            # change range of image to [-1, 1]
            # TODO : different procedure for colored images
            img = img.astype('float32')
            mx = np.max(img)
            mn = np.min(img)
            m = mx/2 + mn/2
            r = mx/2 - mn/2
            if r:
                img = (img - m)/r
                # add to dataset
                X.append(img)
        self.X = np.array(X)
            




    






























        
