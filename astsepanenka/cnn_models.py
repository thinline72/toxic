from keras.layers import Dense, LSTM, Bidirectional,Flatten
from keras.layers import Conv2D, MaxPool2D, Reshape,Embedding
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, Concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D


def BiLSTM_2DCNN(maxlen,max_features,embed_size,embedding_matrix,lstm_units=256):
    conv_filters = 32
    sequence_input = Input(shape=(maxlen,), dtype='int32')
    embedded_sequences = Embedding(max_features,embed_size,input_length=maxlen,weights=[embedding_matrix],trainable=False)(sequence_input)
    x = SpatialDropout1D(0.2)(embedded_sequences)
    x = Bidirectional(LSTM(lstm_units,return_sequences=True))(embedded_sequences)
    x = Dropout(0.1)(x)
    x = Reshape((2 * maxlen,lstm_units, 1))(x)
    x = Conv2D(conv_filters, (3, 3))(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Flatten()(x)
    preds = Dense(6, activation='sigmoid')(x)
    model = Model(sequence_input, preds)
    return model


def Conv2D_block(reshape,sequence_length,embedding_dim):
    filter_sizes = [3,4,5]
    num_filters = 32

    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='he_uniform', activation='relu')(reshape)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='he_uniform', activation='relu')(reshape)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='he_uniform', activation='relu')(reshape)

    maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)

    concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
    flatten = Flatten()(concatenated_tensor)
    return flatten


def Art_CNN(maxlen,max_features,embed_size,embedding_matrix):
    sequence_input = Input(shape=(maxlen,), dtype='int32')
    embedding = Embedding(max_features, embed_size, weights=[embedding_matrix],input_length=maxlen,trainable=False)(sequence_input)
    x = SpatialDropout1D(0.2)(embedding)
    reshape = Reshape((maxlen,embed_size,1))(x)
    x = Conv2D_block(reshape,maxlen,embed_size)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.2)(x)
    preds = Dense(6, activation='sigmoid')(x)
    model = Model(sequence_input, preds)
    return model