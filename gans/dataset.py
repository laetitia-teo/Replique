import numpy as np
import tensorflow as tf
import cv2
from glob import glob
import os.path as op
import os
from tqdm import tqdm
import psutil
import queue
import gc

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
                 color=True):
        self.path = path
        self.filenames = sorted(glob(op.join(self.path, '*.jpg')))
        self.batch_size = batch_size
        self.color = color
        assert(self.scan_size()) # stop if dataset is too big
        self.check_files()
        self.load_data()
        self.size = len(self.X)
        if n_iter:
            self.n_iter = n_iter # max number of iterations
        else:
            self.n_iter = int(len(self.X)/self.batch_size)
    
    def __iter__(self):
        self.batch_idx = 0
        return self
    
    def __next__(self): # TODO check this
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
    
    def check_files(self):
        """
        Performs a check on the images, and deletes the ones that cannot be
        opened.
        """
        for f in self.filenames:
            img = cv2.imread(f, int(self.color))
            if img is None:
                os.remove(f)
    
    def iterations(self, n_iter):
        """Change max number of iterations"""
        self.n_iter = n_iter
        
    def scan_size(self):
        """
        Returns True if the dataset is smaller than 1GB.
        """
        max_memory = 10e9/4 # because 32-bit floats will be used
        memory = 0
        for f in self.filenames:
            img = cv2.imread(f, int(self.color))
            if img is not None:
                m = 1
                for dim in img.shape:
                    m *= dim
                memory += m
            else:
                print('error opening %s' % f)
        print('size is %s bytes' % memory)
        return memory <= max_memory
    
    def load_data(self, from_idx):
        """
        Loads the image data into X.
        """
        length = len(self.filenames)
        # we assume all images have the same dimensions
        shape = cv2.imread(filenames[0], int(self.color)).shape
        if not self.color:
            shape += (1,) # add additionnal channel for black and white
        X = []
        for f in tqdm(self.filenames[:5000]):
            if psutil.virtual_memory()[2] >= 60.0:
                break # preserve memory
            img = cv2.imread(f, int(self.color))
            if img is not None:
                if not self.color:
                    img = np.expand_dims(img, axis=-1)
                # change range of image to [-1, 1]
                # TODO : different procedure for colored images
                if not self.color:
                    img = img.astype('float32')
                    mx = np.max(img)
                    mn = np.min(img)
                    m = mx/2 + mn/2
                    r = mx/2 - mn/2
                else:
                    mx = np.amax(np.amax(img, axis=0), axis=0)
                    mn = np.amin(np.amin(img, axis=0), axis=0)
                    m = mx/2 + mn/2
                    r = mx/2 - mn/2
                if np.all(r):
                    img = (img - m)/r # works in both cases
                    # add to dataset
                    X.append(img)
        self.X = np.array(X)
    
    def load_chunk(self, idx):
        """
        Loads a chunk, a certain number of batches.
        """
        for f in self.filenames[idx:]:
            ...
    
class ImgDataset2():
    """
    Class for handling an image dataset.
    Supports batch iteration.
    Loads chunks of data in memory before yielding them to the iterator.
    """
    # TODO memory management, dynamic loading ?
    # TODO create interface for unifying all types of datasets
    # TODO dynamic dataset from images ? May slow down training too much
    
    def __init__(self, 
                 path,
                 data_path,
                 batch_size=64,
                 n_iter=None,
                 color=True):
        self.path = path
        self.data_path = data_path
        self.max_memory = 65.0
        self.filenames = sorted(glob(op.join(self.path, '*.jpg')))
        self.batch_size = batch_size
        self.color = color
        self.data_filenames = sorted(glob(op.join(self.data_path, '*.npy')))
        self.X = None
        if not self.data_filenames:
            print('loading files')
            self.check_files()
            self.load_data_in_folder()
        self.size = len(self.data_filenames)
        if n_iter:
            self.n_iter = n_iter # max number of iterations
        else:
            self.n_iter = self.size
    
    def __iter__(self):
        self.batch_idx = 0
        # remaining batches in X :
        self.remain = self.load_chunk(self.batch_idx) # loads first chunk
        return self
    
    def __next__(self): # TODO check this
        if self.batch_idx >= self.n_iter:
            raise StopIteration
        print('remain : %s' % self.remain)
        if not self.remain:
            self.X = None
            gc.collect()
            self.remain = self.load_chunk(self.batch_idx)
        self.batch_idx += 1
        self.remain -= 1
        return self.X.get()
    
    def stop_loading(self):
        """
        Criterion for stopping the loading of images.
        """
        return psutil.virtual_memory()[2] >= self.max_memory
    
    def check_files(self):
        """
        Performs a check on the images, and deletes the ones that cannot be
        opened.
        """
        print('checking files')
        for f in self.filenames:
            img = cv2.imread(f, int(self.color))
            if img is None:
                os.remove(f)
    
    def iterations(self, n_iter):
        """Change max number of iterations"""
        self.n_iter = n_iter
        
    def scan_size(self):
        """
        Returns True if the dataset is smaller than 1GB.
        """
        max_memory = 10e9/4 # because 32-bit floats will be used
        memory = 0
        for f in self.filenames:
            img = cv2.imread(f, int(self.color))
            if img is not None:
                m = 1
                for dim in img.shape:
                    m *= dim
                memory += m
            else:
                print('error opening %s' % f)
        print('size is %s bytes' % memory)
        return memory <= max_memory
    
    def load_data_in_folder(self):
        """
        Reads the image data, performs operations, and saves the resulting data
        in self.data_path, as batch tensors of length batch_size.
        """
        print('loading files in data folder')
        n = len(self.filenames)
        idx_max = n // self.batch_size
        for idx in range(0, idx_max-1):
            data = []
            for f in self.filenames[idx:idx+64]:
                img = cv2.imread(f, int(self.color))
                if not self.color:
                    img = np.expand_dims(img, axis=-1)
                data.append(img)
            data = np.array(data)
            data = data.astype('float32')
            data = (data - 127.5)/127.5
            np.save(op.join(self.data_path, str(idx)), data)
        # TODO last batch ?
        self.data_filenames = sorted(glob(op.join(self.data_path, '*.npy')))
    
    def load_chunk(self, start): # TODO parallelize this whole process
        """
        Loads a chunk, a certain number of batches, and stores them in self.X
        Returns the number of loaded chunks. 
        """
        self.X = queue.Queue()
        n = 0 # number of loaded batches
        print('stop loading : %s' % self.stop_loading())
        print('start + n : %s' % str(start + n))
        while (not self.stop_loading()) and (start + n) < self.size:
            print('load')
            self.X.put(np.load(self.data_filenames[start+n]))
            n += 1
        print('return chunk')
        return n
    
class ImgDataset3():
    """
    Class for handling an image dataset.
    Supports batch iteration.
    """
    # TODO memory management, dynamic loading ?
    # TODO create interface for unifying all types of datasets
    # TODO dynamic dataset from images ? May slow down training too much
    
    def __init__(self, 
                 path,
                 data_path,
                 batch_size=64,
                 n_iter=None,
                 color=True):
        self.path = path
        self.data_path = data_path
        self.max_memory = 65.0
        self.filenames = sorted(glob(op.join(self.path, '*.jpg')))
        self.batch_size = batch_size
        self.color = color
        self.data_filenames = sorted(glob(op.join(self.data_path, '*.npy')))
        self.X = None
        if not self.data_filenames:
            print('loading files')
            self.check_files()
            self.load_data_in_folder()
        elif not self.correct_batch_size_in_files():
            print('loading files')
            self.check_files()
            self.load_data_in_folder()
        self.size = len(self.data_filenames)
        if n_iter:
            self.n_iter = n_iter # max number of iterations
        else:
            self.n_iter = self.size
    
    def __iter__(self):
        self.batch_idx = 0
        return self
    
    def __next__(self): # TODO check this
        if self.batch_idx >= self.n_iter:
            raise StopIteration
        tensor = np.load(self.data_filenames[self.batch_idx])
        self.batch_idx += 1
        return tensor
    
    def stop_loading(self):
        """
        Criterion for stopping the loading of images.
        """
        return psutil.virtual_memory()[2] >= self.max_memory
    
    def check_files(self):
        """
        Performs a check on the images, and deletes the ones that cannot be
        opened.
        """
        print('checking files')
        for f in tqdm(self.filenames):
            img = cv2.imread(f, int(self.color))
            if img is None:
                os.remove(f)
    
    def correct_batch_size_in_files(self):
        """
        Checks that our existing files have the correct batch size.
        """
        print('checking correct file sizes')
        all_ok = True
        for f in self.data_filenames:
            all_ok *= (np.load(f).shape[0] == self.batch_size)
            if not all_ok:
                break
            print(all_ok)
        return all_ok
    
    def iterations(self, n_iter):
        """Change max number of iterations"""
        self.n_iter = n_iter
        
    def scan_size(self):
        """
        Returns True if the dataset is smaller than 1GB.
        """
        max_memory = 10e9/4 # because 32-bit floats will be used
        memory = 0
        for f in self.filenames:
            img = cv2.imread(f, int(self.color))
            if img is not None:
                m = 1
                for dim in img.shape:
                    m *= dim
                memory += m
            else:
                print('error opening %s' % f)
        print('size is %s bytes' % memory)
        return memory <= max_memory
    
    def load_data_in_folder(self):
        """
        Reads the image data, performs operations, and saves the resulting data
        in self.data_path, as batch tensors of length batch_size.
        """
        if self.data_filenames:
            print('removing existing data files')
            for f in tqdm(self.data_filenames):
                os.remove(f)
        print('loading files in data folder')
        n = len(self.filenames)
        idx_max = n // self.batch_size
        for idx in tqdm(range(0, idx_max-1)):
            data = []
            for f in self.filenames[idx:idx+self.batch_size]:
                img = cv2.imread(f, int(self.color))
                if not self.color:
                    img = np.expand_dims(img, axis=-1)
                data.append(img)
            data = np.array(data)
            data = data.astype('float32')
            data = (data - 127.5)/127.5
            np.save(op.join(self.data_path, str(idx)), data)
        # TODO last batch ?
        self.data_filenames = sorted(glob(op.join(self.data_path, '*.npy')))
    
            
    




    






























        
