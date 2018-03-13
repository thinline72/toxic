from __future__ import absolute_import

import math, random
import pickle
import bcolz
import numpy as np
from keras.utils import Sequence
from .sampling_utils import stratified_kfold_sampling

def load_array(rootdir, mode='r'):
    return bcolz.open(rootdir, mode=mode)[:]

def load_carray(rootdir, mode='r'):
    return bcolz.open(rootdir, mode=mode)

def load_npy_files(folder, names, shape, cpu_cores):
    return np.stack(Parallel(n_jobs=cpu_cores)(delayed(load_npy)(folder, name, shape) for name in names))
        
def load_npy(folder, name, shape):
    try:
        arr = np.load(file = '{}{}'.format(os.path.join(folder, name),'.npy'))
        if arr.shape != shape:
            return np.zeros(shape)
        else:
            return arr
    except:
        print("Cannot read {}{}".format(folder, name))
        return np.zeros(shape)
    
def load_embs(embs_dir='/src/DL/commons/word_embeddings/', embs_name='crawl-300d-2M'):
    '''
    Load embeddings from bcolz (vectors) and pkl (words) fromat.
    Available embs_names:
        - crawl-300d-2M
        - glove-300d-840B
        - glove-50d-6B
        - glove-twitter-200d-27B
    '''
    vecs_path = embs_dir+embs_name+'-vectors.bcolz'
    words_path = embs_dir+embs_name+'-words.pkl'
    
    vectors = load_array(vecs_path)
    inx2word = pickle.load(open(words_path, 'rb'), encoding='utf-8')
    word2inx = {word: inx for inx, word in enumerate(inx2word)}
    
    return vectors, inx2word, word2inx
    
class FeatureSequence(Sequence):
    
    def __init__(self, X, Y, batch_size, shuffle=False):
        
        self.X, self.Y = X, Y
        self.batch_size = batch_size
        
        self.inx = np.arange(0, self.Y.shape[0])
        self.shuffle = shuffle
        if self.shuffle:
            np.random.shuffle(self.inx)

    def __len__(self):
        return math.ceil(self.inx.shape[0] / self.batch_size)

    def __getitem__(self, i):
        batch_inx = self.inx[i*self.batch_size:(i+1)*self.batch_size]
        
        return self.X[batch_inx], self.Y[batch_inx]
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.inx)
        
class PseudoFeatureSequence(Sequence):
    
    def __init__(self, X, Y, batch_size, 
                 test_X, test_X_meta, test_Y, test_batch_size,
                 shuffle=False):
        
        self.X, self.Y = X, Y
        self.batch_size = batch_size
        self.inx = np.arange(0, self.Y.shape[0])
        
        self.test_X, self.test_X_meta, self.test_Y = test_X, test_X_meta, test_Y
        self.test_batch_size = test_batch_size
        self.test_inx = np.arange(0, self.test_Y.shape[0])
        
        self.shuffle = shuffle
        if self.shuffle:
            np.random.shuffle(self.inx)
            np.random.shuffle(self.test_inx)

    def __len__(self):
        return math.ceil(self.inx.shape[0] / self.batch_size)

    def __getitem__(self, i):
        batch_inx = self.inx[i*self.batch_size:(i+1)*self.batch_size]
        test_batch_inx = self.test_inx[i*self.test_batch_size:(i+1)*self.test_batch_size]
        
        return (np.concatenate([self.X[batch_inx], self.test_X[test_batch_inx]]), 
                np.concatenate([self.Y[batch_inx], self.test_Y[test_batch_inx]]))
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.inx)
            np.random.shuffle(self.test_inx)
            
            
class StratifiedFeatureSequence(Sequence):
    
    def __init__(self, X, Y, batch_size, seed=random.randint(0, 1e6)):
        self.X, self.Y = X, Y
        self.seed = seed
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(self.Y) / self.batch_size)
        _, self.batches = stratified_kfold_sampling(self.Y, self.num_batches, self.seed)

    def __len__(self):
        return self.num_batches

    def __getitem__(self, i):
        return self.X[self.batches[i]], self.Y[self.batches[i]]
    
    def on_epoch_end(self):
        pass
    
class FeatureMetaSequence(Sequence):
    
    def __init__(self, X, X_meta, Y, batch_size, shuffle=False):
        
        self.X, self.X_meta, self.Y = X, X_meta, Y
        self.batch_size = batch_size
        
        self.inx = np.arange(0, self.Y.shape[0])
        self.shuffle = shuffle
        if self.shuffle:
            np.random.shuffle(self.inx)

    def __len__(self):
        return math.ceil(self.inx.shape[0] / self.batch_size)

    def __getitem__(self, i):
        batch_inx = self.inx[i*self.batch_size:(i+1)*self.batch_size]
        
        return [self.X[batch_inx], self.X_meta[batch_inx]], self.Y[batch_inx]
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.inx)
        
class PseudoFeatureMetaSequence(Sequence):
    
    def __init__(self, X, X_meta, Y, batch_size, 
                 test_X, test_X_meta, test_Y, test_batch_size,
                 shuffle=False):
        
        self.X, self.X_meta, self.Y = X, X_meta, Y
        self.batch_size = batch_size
        self.inx = np.arange(0, self.Y.shape[0])
        
        self.test_X, self.test_X_meta, self.test_Y = test_X, test_X_meta, test_Y
        self.test_batch_size = test_batch_size
        self.test_inx = np.arange(0, self.test_Y.shape[0])
        
        self.shuffle = shuffle
        if self.shuffle:
            np.random.shuffle(self.inx)
            np.random.shuffle(self.test_inx)

    def __len__(self):
        return math.ceil(self.inx.shape[0] / self.batch_size)

    def __getitem__(self, i):
        batch_inx = self.inx[i*self.batch_size:(i+1)*self.batch_size]
        test_batch_inx = self.test_inx[i*self.test_batch_size:(i+1)*self.test_batch_size]
        
        return ([np.concatenate([self.X[batch_inx], self.test_X[test_batch_inx]]),
                 np.concatenate([self.X_meta[batch_inx], self.test_X_meta[test_batch_inx]])],
                np.concatenate([self.Y[batch_inx], self.test_Y[test_batch_inx]]))
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.inx)
            np.random.shuffle(self.test_inx)