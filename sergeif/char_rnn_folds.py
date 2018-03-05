# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import tensorflow as tf

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Model
from keras.layers import Dense, Embedding, Input, Concatenate, Conv1D, Activation, TimeDistributed, Flatten, RepeatVector, Permute,multiply
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout, GRU, GlobalAveragePooling1D, MaxPooling1D, SpatialDropout1D, BatchNormalization
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re,gc
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score,log_loss

#from utils import to_categorical, get_comment_ids, get_conv_shape
#from vdcnn import build_model

max_features = 73
maxlen = 512

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

#ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'\"/\\|_@#$%^&*~`+ =<>()[]{}"
ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789,.!?:_@#$&"

def preproc_str(s):
    chs = s.encode('ascii', errors='ignore').decode('ascii', errors='ignore').lower()
    return chs #''.join([ch if ch in ALPHABET else ' ' for ch in chs])

train['comment_text'] = train.comment_text.fillna('').apply(preproc_str)
test['comment_text'] = test.comment_text.fillna('').apply(preproc_str)

print(train['comment_text'].values[239])
print(test['comment_text'].values[239])

list_sentences_train = train["comment_text"].fillna("").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("").values

tokenizer = Tokenizer()
tokenizer.fit_on_texts(list(list_sentences_train) + list(list_sentences_test))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
print(tokenizer.word_counts.items())
print('padding sequences')
X_train = {}
X_test = {}
X_train['text'] = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen, padding='post', truncating='post')
X_test['text'] = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen, padding='post', truncating='post')

def flatten(l): return [item for sublist in l for item in sublist]
max_features = np.unique(flatten(X_train['text'])).shape[0] + 1
print('max_features_train:', max_features)
max_features_test = np.unique(flatten(X_test['text'])).shape[0] + 1
print('max_features_test:', max_features_test)

def _train_model(model, batch_size, train_x, train_y, val_x, val_y):
    best_loss = -1
    best_weights = None
    best_epoch = 0

    current_epoch = 0

    while True:
        model.fit(train_x, train_y, batch_size=batch_size, epochs=1)
        y_pred = model.predict(val_x, batch_size=batch_size)

        total_loss = 0
        for j in range(6):
            loss = log_loss(val_y[:, j], y_pred[:, j])
            total_loss += loss

        total_loss /= 6.
        auc = roc_auc_score(val_y, y_pred)

        print("Epoch {0} loss {1} best_loss {2} roc_auc {3}".format(current_epoch, total_loss, best_loss, auc))

        if (np.isnan(total_loss)):
            break

        current_epoch += 1
        if total_loss < best_loss or best_loss == -1:
            best_loss = total_loss
            best_weights = model.get_weights()
            best_epoch = current_epoch
        else:
            if current_epoch - best_epoch == 5:
                break

    model.set_weights(best_weights)
    return model

def get_model_cnn(X_train):
    inp = Input(shape=(maxlen, ), name="text")
    x = Embedding(max_features, 16)(inp)
    x = SpatialDropout1D(0.2)(x)
    c1 = Conv1D(64, 3, activation="relu")(x)
    c1 = GlobalMaxPool1D()(c1)
    c2 = Conv1D(64, 5, activation="relu")(x)
    c2 = GlobalMaxPool1D()(c2)
    c3 = Conv1D(64, 9, activation="relu")(x)
    c3 = GlobalMaxPool1D()(c3)
    x = Dropout(0.2)(Concatenate()([c1,c2,c3]))
    x = Dropout(0.2)(Dense(128, activation="relu")(x))
    x = Dropout(0.2)(Dense(128, activation="relu")(x))
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=[inp], outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model        

from keras.layers import Dense, Embedding, Input
from keras.layers import Bidirectional, Dropout, SpatialDropout1D, CuDNNGRU
from keras.models import Model
from keras.optimizers import RMSprop

def get_model_rnn(X_train):
    input_layer = Input(shape=(maxlen,), name="text")
    embedding_layer = Embedding(max_features, 16)(input_layer)
    #x = SpatialDropout1D(0.2)(embedding_layer)
    x = embedding_layer
    x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
    c1 = Dropout(0.3)(GlobalMaxPool1D()(x))
    output_layer = Dense(6, activation="sigmoid")(c1)

    model = Model(inputs=[input_layer], outputs=output_layer)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model
      


print('start modeling')
batch_size = 128
epochs = 20
scores = []
predict = np.zeros((test.shape[0],6))
oof_predict = np.zeros((train.shape[0],6))

kf = KFold(n_splits=10, shuffle=True, random_state=1256)
for train_index, test_index in kf.split(X_train['text']):
    kfold_X_train = {}
    kfold_X_valid = {}
    y_train,y_test = y[train_index], y[test_index]
    for c in ['text']:
        kfold_X_train[c] = X_train[c][train_index]
        kfold_X_valid[c] = X_train[c][test_index]

    model = get_model_rnn(X_train)

    # Stage 2: Build Model
    #model = build_model(num_filters = [64,128,256,512], rep_filters = [4,4,2,2], num_classes=6, sequence_max_length=maxlen, num_quantized_chars=max_features, embedding_size=16, learning_rate=0.01, top_k=3, dense_size=1536)
    model = _train_model(model, batch_size, kfold_X_train, y_train, kfold_X_valid, y_test)

    predict += model.predict(X_test, batch_size=batch_size) * 0.1
    oof_predict[test_index] = model.predict(kfold_X_valid, batch_size=batch_size)
    cv_score = roc_auc_score(y_test, oof_predict[test_index])
    scores.append(cv_score)
    print('score: ',cv_score)

print('Total CV score is {}'.format(np.mean(scores)))    

sample_submission = pd.DataFrame.from_dict({'id': test['id']})
oof = pd.DataFrame.from_dict({'id': train['id']})
for c in list_classes:
    oof[c] = np.zeros(len(train))
    sample_submission[c] = np.zeros(len(test))
    
sample_submission[list_classes] = predict
sample_submission.to_csv("test_crnn.csv", index=False)

oof[list_classes] = oof_predict
oof.to_csv('train_crnn.csv', index=False)

