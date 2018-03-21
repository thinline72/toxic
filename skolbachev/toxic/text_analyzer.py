from __future__ import absolute_import

import multiprocessing
import collections
import numpy as np
from joblib import Parallel, delayed
from keras.preprocessing.sequence import pad_sequences

class TextAnalyzer(object):
    
    def __init__(self, word2inx, vectors, max_len, reverse=False,
                 min_word_hits=1, min_doc_hits=1, max_doc_freq=1.0,
                 process_oov_words=False, oov_window=5, oov_min_doc_hits=1):
        self.word2inx = word2inx
        self.vectors = vectors
        
        self.PAD_TOKEN = "_PAD_"
        self.UNK_TOKEN = "_UNK_"
        self.max_len = max_len
        self.reverse = reverse

        self.emb_size = vectors.shape[1]
        self.pad_vec = -1*np.ones(self.emb_size)
        self.unk_vec = np.zeros(self.emb_size)
        
        self.doc_counts = 0
        self.word_hits = collections.OrderedDict()
        self.doc_hits = collections.OrderedDict()
        
        self.min_word_hits = min_word_hits
        self.min_doc_hits = min_doc_hits
        self.max_doc_freq = max_doc_freq

        self.process_oov_words = process_oov_words
        self.oov_window = oov_window
        self.oov_min_doc_hits = oov_min_doc_hits
        self._oov_vec = collections.OrderedDict()
        self._oov_vec_sum = collections.OrderedDict()

        self.inx2emb = [self.PAD_TOKEN, self.UNK_TOKEN]
        self.emb_vectors = [self.pad_vec, self.unk_vec]
        self._embs = collections.OrderedDict()
    
    def fit_on_docs(self, docs):
        doc_len = []
        doc_ulen = []
        docs_seq = []
        
        self.doc_counts = len(docs)
        
        for doc in docs:
            for w in doc:
                if w in self.word_hits: self.word_hits[w] += 1
                else: self.word_hits[w] = 1
            for w in set(doc):
                if w in self.doc_hits: self.doc_hits[w] += 1
                else: self.doc_hits[w] = 1

            doc_len.append(len(doc))
            doc_ulen.append(len(set(doc)))
            
            for i, w in enumerate(doc):
                if w in self.word2inx:
                    self._embs[w] = self.vectors[self.word2inx[w]]
                elif self.process_oov_words:
                    emb = self.unk_vec
                    for pw in doc[max(i-self.oov_window,0):i]: 
                        if pw in self.word2inx:
                            emb+= self.vectors[self.word2inx[pw]]
                            if w in self._oov_vec_sum: self._oov_vec_sum[w] += 1
                            else: self._oov_vec_sum[w] = 1
                    for pw in doc[i+1:i+self.oov_window]: 
                        if pw in self.word2inx:
                            emb+= self.vectors[self.word2inx[pw]]
                            if w in self._oov_vec_sum: self._oov_vec_sum[w] += 1
                            else: self._oov_vec_sum[w] = 1

                    self._oov_vec[w] = emb
        
        if self.process_oov_words:
            for w, s in self._oov_vec_sum.items():
                if (self.doc_hits[w] >= self.oov_min_doc_hits):
                    self._embs[w] = self._oov_vec[w] / s
        
        for w,v in self._embs.items():
            if (self.word_hits[w] >= self.min_word_hits and \
                self.doc_hits[w] >= self.min_doc_hits and \
                self.doc_hits[w]/self.doc_counts <= self.max_doc_freq):
                self.inx2emb.append(w)
                self.emb_vectors.append(v)
                 
        self.emb2inx = {word: inx for inx, word in enumerate(self.inx2emb)}
        self.emb_vectors = np.stack(self.emb_vectors)
        
        for doc in docs:
            if self.reverse:
                docs_seq.append([self.emb2inx[w] if w in self.emb2inx else self.emb2inx[self.UNK_TOKEN] for w in doc[::-1]])
            else:
                docs_seq.append([self.emb2inx[w] if w in self.emb2inx else self.emb2inx[self.UNK_TOKEN] for w in doc])
            
        self.summary()
        return [pad_sequences(docs_seq, maxlen=self.max_len), np.stack([np.array(doc_len), np.array(doc_ulen)], axis=-1)]
    
    def transform_docs(self, docs):
        doc_len = []
        doc_ulen = []
        docs_seq = []

        for doc in docs:
            doc_len.append(len(doc))
            doc_ulen.append(len(set(doc)))
            if self.reverse:
                docs_seq.append([self.emb2inx[w] if w in self.emb2inx else self.emb2inx[self.UNK_TOKEN] for w in doc[::-1]])
            else:
                docs_seq.append([self.emb2inx[w] if w in self.emb2inx else self.emb2inx[self.UNK_TOKEN] for w in doc])
            
        return [pad_sequences(docs_seq, maxlen=self.max_len), np.stack([np.array(doc_len), np.array(doc_ulen)], axis=-1)]
        
    def summary(self):
        print("Docs: {}".format(self.doc_counts))
        print("Selected words: {}".format(len(self.inx2emb)))
        print("Processed OOV words: {}".format(len(self._oov_vec_sum.keys() & set(self.inx2emb))))