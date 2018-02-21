from __future__ import absolute_import

from keras.layers import *
from keras.layers.core import Activation
from keras.models import *
from keras.constraints import *
from keras.regularizers import *

from .attentions import *

def getBiCuDNNGRUx2Model(input_shape, classes, num_words, emb_size, emb_matrix,
                         attention=0, dense=False, emb_trainable=False):

    x_input = Input(shape=(input_shape,))
    
    emb = Embedding(num_words, emb_size, weights=[emb_matrix], trainable=emb_trainable, name='embs')(x_input)
    emb = SpatialDropout1D(0.3)(emb)
        
    rnn1 = Bidirectional(CuDNNGRU(64, return_sequences=True))(emb)
    rnn2 = Bidirectional(CuDNNGRU(64, return_sequences=True))(rnn1)
    x = concatenate([rnn1, rnn2])

    if attention == 1: x = AttentionWeightedAverage()(x)
    elif attention == 2: x = Attention()(x)
    else: x = GlobalMaxPooling1D()(x)
    x = Dropout(0.3)(x)
    
    if dense: 
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.3)(x)
    
    x_output = Dense(classes, activation='sigmoid')(x)
    return Model(inputs=x_input, outputs=x_output)

def getBiCuDNNGRUHowardModel(input_shape, classes, num_words, emb_size, emb_matrix,
                             attention=0, dense=False, emb_trainable=False):

    x_input = Input(shape=(input_shape,))
    
    emb = Embedding(num_words, emb_size, weights=[emb_matrix], trainable=emb_trainable, name='embs')(x_input)
    emb = SpatialDropout1D(0.3)(emb)
        
    rnn_forw, rnn_last_forw = CuDNNGRU(100, return_sequences=True, return_state=True)(emb)
    rnn_back, rnn_last_back = CuDNNGRU(100, return_sequences=True, return_state=True, go_backwards=True)(emb)
    
    rnn = concatenate([rnn_forw, rnn_back])
    rnn_max = GlobalMaxPool1D()(rnn)
    rnn_avg = GlobalAvgPool1D()(rnn)
    rnn_last = concatenate([rnn_last_forw, rnn_last_back])
    
    x = concatenate([rnn_last, rnn_max, rnn_avg])
    x = Dropout(0.3)(x)
    
    if dense: 
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.3)(x)
    
    x_output = Dense(classes, activation='sigmoid')(x)
    return Model(inputs=x_input, outputs=x_output)

def crop(dimension, start, end):
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]
    return Lambda(func)