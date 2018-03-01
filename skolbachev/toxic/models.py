from __future__ import absolute_import

from keras.layers import *
from keras.layers.core import Activation
from keras.models import *
from keras.constraints import *
from keras.regularizers import *

from .attentions import *

def getModel0(input_shape, classes, num_words, emb_size, emb_matrix, emb_dropout=0.5,
              attention=0, dense=False, emb_trainable=False):

    x_input = Input(shape=(input_shape,))
    
    emb = Embedding(num_words, emb_size, weights=[emb_matrix], trainable=emb_trainable, name='embs')(x_input)
    emb = SpatialDropout1D(emb_dropout)(emb)
        
    rnn1 = Bidirectional(CuDNNGRU(64, return_sequences=True))(emb)
    rnn2 = Bidirectional(CuDNNGRU(64, return_sequences=True))(rnn1)
    x = concatenate([rnn1, rnn2])

    if attention == 1: x = AttentionWeightedAverage()(x)
    elif attention == 2: x = Attention()(x)
    else: x = GlobalMaxPooling1D()(x)
    
    if dense: 
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.3)(x)
    
    x_output = Dense(classes, activation='sigmoid')(x)
    return Model(inputs=x_input, outputs=x_output)

def getModel1(input_shape, classes, num_words, emb_size, emb_matrix, emb_dropout=0.5,
              attention=0, dense=False, emb_trainable=False):

    x_input = Input(shape=(input_shape,))
    
    emb = Embedding(num_words, emb_size, weights=[emb_matrix], trainable=emb_trainable, name='embs')(x_input)
    emb = SpatialDropout1D(emb_dropout)(emb)
        
    rnn, rnn_fw, rnn_bw = Bidirectional(CuDNNGRU(100, return_sequences=True, return_state=True))(emb)
    
    rnn_max = GlobalMaxPool1D()(rnn)
    rnn_avg = GlobalAvgPool1D()(rnn)
    rnn_last = concatenate([rnn_fw, rnn_bw])
    
    x = concatenate([rnn_max, rnn_avg, rnn_last])
    
    if dense: 
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.3)(x)
    
    x_output = Dense(classes, activation='sigmoid')(x)
    return Model(inputs=x_input, outputs=x_output)

def getModel2(input_shape, classes, num_words, emb_size, emb_matrix, emb_dropout=0.5,
              attention=0, dense=False, emb_trainable=False):

    x_input = Input(shape=(input_shape,))
    
    emb = Embedding(num_words, emb_size, weights=[emb_matrix], trainable=emb_trainable, name='embs')(x_input)
    emb = SpatialDropout1D(emb_dropout)(emb)
        
    rnn = Bidirectional(CuDNNGRU(75, return_sequences=True))(emb)
    
    cnn1 = Conv1D(filters=50, kernel_size=3, activation='relu', padding='same')(rnn)
    cnn2 = Conv1D(filters=50, kernel_size=4, activation='relu', padding='same')(rnn)
    cnn3 = Conv1D(filters=50, kernel_size=5, activation='relu', padding='same')(rnn)
    
    x = concatenate([rnn, cnn1, cnn2, cnn3])
    
    if attention == 1: x = AttentionWeightedAverage()(x)
    elif attention == 2: x = Attention()(x)
    else: x = GlobalMaxPooling1D()(x)
    
    if dense: 
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.3)(x)
    
    x_output = Dense(classes, activation='sigmoid')(x)
    return Model(inputs=x_input, outputs=x_output)

def getModel3(input_shape, classes, num_words, emb_size, emb_matrix, emb_dropout=0.5,
              attention=0, dense=False, emb_trainable=False):

    x_input = Input(shape=(input_shape,))
    
    emb = Embedding(num_words, emb_size, weights=[emb_matrix], trainable=emb_trainable, name='embs')(x_input)
    emb = SpatialDropout1D(emb_dropout)(emb)
    
    rnn1 = Bidirectional(CuDNNGRU(64, return_sequences=True))(emb)
    rnn2 = Bidirectional(CuDNNGRU(64, return_sequences=False))(rnn1)
    x = rnn2
    
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