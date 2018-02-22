from keras.models import Model
from keras.layers.convolutional import Conv1D
from keras.layers.embeddings import Embedding
from keras.layers import Input, Dense, Dropout, Lambda
from keras.layers.pooling import MaxPooling1D
from vdcnn_layers import ConvBlockVDCNN
from keras_metrics import auc_keras
import tensorflow as tf

num_filters_default = [64,128,256,512] # from VDCNN paper


def VDCNN_model(input_shape,num_classes,num_words,emb_size,emb_matrix,num_filters=num_filters_default,top_k=8,emb_trainable=False):

    inputs = Input(shape=(input_shape, ), dtype='int32', name='inputs')

    embedded_sent = Embedding(num_words, emb_size, weights=[emb_matrix], trainable=emb_trainable, name='embs')(inputs)

    conv = Conv1D(filters=64, kernel_size=3, strides=2, padding="same")(embedded_sent)

    for i in range(len(num_filters)):
        conv = ConvBlockVDCNN(conv.get_shape().as_list()[1:], num_filters[i])(conv)
        conv = MaxPooling1D(pool_size=3, strides=2, padding="same")(conv)
        
    def k_max_pooling(x):
        x = tf.transpose(x, [0, 2, 1])
        k_max = tf.nn.top_k(x, k=top_k)
        return tf.reshape(k_max[0], (-1, num_filters[-1] * top_k))

    k_max = Lambda(k_max_pooling,output_shape=(num_filters[-1] * top_k,))(conv)

    # fully-connected layers
    fc1 = Dropout(0.2)(Dense(4096, activation='relu', kernel_initializer='he_normal')(k_max))
    fc2 = Dropout(0.2)(Dense(2048, activation='relu', kernel_initializer='he_normal')(fc1))
    fc3 = Dense(num_classes, activation='sigmoid')(fc2)

    model = Model(inputs=inputs, outputs=fc3)
    return model
