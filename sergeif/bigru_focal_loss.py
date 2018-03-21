# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import tensorflow as tf
#with tf.device('/cpu:0'):
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Model
from keras.layers import Dense, Embedding, Input, Concatenate, Conv1D, Activation, TimeDistributed, Flatten, RepeatVector, Permute,multiply, CuDNNLSTM
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout, GRU, GlobalAveragePooling1D, MaxPooling1D, SpatialDropout1D, BatchNormalization, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam, Nadam, SGD
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re,gc
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score,log_loss
import nltk
from keras import backend as K


cfg = K.tf.ConfigProto()
cfg.gpu_options.allow_growth = True
K.set_session(K.tf.Session(config=cfg))

max_features = 30000
maxlen = 200

sia = SentimentIntensityAnalyzer()

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

X_train = {}
X_test = {}
X_train2 = {}
X_test2 = {}
embedding_matrix = np.array([])
embedding_matrix2 = np.array([])
import pickle
with open(r"Xtrain3.pickle", "rb") as output_file:
    X_train = pickle.load(output_file)
with open(r"Xtest3.pickle", "rb") as output_file:
    X_test = pickle.load(output_file)
with open(r"embmatrix3.pickle", "rb") as output_file:
    embedding_matrix = pickle.load(output_file)
with open(r"Xtrain4.pickle", "rb") as output_file:
    X_train2 = pickle.load(output_file)
with open(r"Xtest4.pickle", "rb") as output_file:
    X_test2 = pickle.load(output_file)
with open(r"embmatrix4.pickle", "rb") as output_file:
    embedding_matrix2 = pickle.load(output_file)

X_test['text2'] = X_test2['text']
X_train['text2'] = X_train2['text']

max_features = embedding_matrix.shape[0]    
max_features2 = embedding_matrix2.shape[0]

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values    

embed_size = 200
embed_size2 = 300

def focal_loss(y_true, y_pred, alpha, gamma=0.5):
    alpha = K.variable(alpha)
    pt = K.abs(1. - y_true - y_pred)
    pt = K.clip(pt, K.epsilon(), 1. - K.epsilon())
    return K.mean(-alpha * K.pow(1. - pt, gamma) * K.log(pt), axis=-1)

def _train_model(model, batch_size, train_x, train_y, val_x, val_y):
    best_loss = -1
    best_weights = None
    best_epoch = 0

    current_epoch = 0

    while True:
        #K.set_value(model.optimizer.lr, 0.001 * (0.85**current_epoch)) 
        model.fit(train_x, train_y, batch_size=batch_size, epochs=1)
        y_pred = model.predict(val_x, batch_size=batch_size)

        total_loss = 0
        for j in range(6):
            loss = log_loss(val_y[:, j], y_pred[:, j])
            total_loss += loss

        total_loss /= 6.

        for j in range(6):
            loss = roc_auc_score(val_y[:, j], y_pred[:, j])
            print(list_classes[j],loss)
        auc = roc_auc_score(val_y, y_pred)

        print("Epoch {0} loss {1} best_loss {2} roc_auc {3}".format(current_epoch, total_loss, best_loss, auc))

        if (np.isnan(total_loss)):
            break

        current_epoch += 1
        if auc > best_loss or best_loss == -1:
            best_loss = auc
            best_weights = model.get_weights()
            best_epoch = current_epoch
        else:
            if current_epoch - best_epoch == 3:
                break

    model.set_weights(best_weights)
    return model

from keras.engine.topology import Layer
from keras.layers import Bidirectional, Dropout, SpatialDropout1D, CuDNNGRU
from keras.optimizers import RMSprop


def get_model_cnn(X_train):
    global embed_size
    inp = Input(shape=(maxlen, ), name="text")
    inp2 = Input(shape=(maxlen, ), name="text2")
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    y = Embedding(max_features2, embed_size2, weights=[embedding_matrix2], trainable=False)(inp2)
    x = SpatialDropout1D(0.4)(x)
    y = SpatialDropout1D(0.4)(y)
    #rnn1 = Bidirectional(CuDNNGRU(200, return_sequences = True))(x)
    rnn1 = CuDNNGRU(64, return_sequences = True)(x)
    #rnn2 = Bidirectional(CuDNNGRU(200, return_sequences = True))(BatchNormalization()(rnn1))
    #rnn3 = Bidirectional(CuDNNGRU(200, return_sequences = True))(y)
    rnn3 = CuDNNGRU(64, return_sequences = True)(y)
    #rnn4 = Bidirectional(CuDNNGRU(200, return_sequences = True))(BatchNormalization()(rnn3))
    #rnn3 = Bidirectional(CuDNNGRU(128, return_sequences = True))(rnn2)
    #x = Conv1D(128, kernel_size = 3, padding = "valid", kernel_initializer = "he_uniform")(Concatenate()([rnn1, rnn3]))
    #avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(Concatenate()([rnn1, rnn3]))
    #x = Concatenate()([avg_pool, max_pool])
    x = Dense(32, activation='relu')(max_pool)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=[inp,inp2], outputs=x)
    opt = Nadam(lr=0.001)
    model.compile(loss=lambda y_true, y_pred: focal_loss(y_true, y_pred, 1.6, 2),
        #loss='binary_crossentropy', 
        optimizer='adam', metrics=['accuracy'])
    return model        

print('start modeling')

batch_size = 128
epochs = 5
scores = []
predict = np.zeros((test.shape[0],6))
oof_predict = np.zeros((train.shape[0],6))


kf = KFold(n_splits=10, shuffle=True, random_state=1256)
ifold = 0
for train_index, test_index in kf.split(X_train['num_vars']):
    kfold_X_train = {}
    kfold_X_valid = {}
    y_train,y_test = y[train_index], y[test_index]
    for c in ['text','text2','num_vars']:
        kfold_X_train[c] = X_train[c][train_index]
        kfold_X_valid[c] = X_train[c][test_index]

    model = get_model_cnn(X_train)
    #print(model.summary())
    model = _train_model(model, batch_size, kfold_X_train, y_train, kfold_X_valid, y_test)

    predict += model.predict(X_test, batch_size=batch_size) * 0.1
    oof_predict[test_index] = model.predict(kfold_X_valid, batch_size=batch_size)
    cv_score = roc_auc_score(y_test, oof_predict[test_index])
    scores.append(cv_score)
    print('score: ',cv_score)
    K.clear_session()

print('Total CV score is {}'.format(np.mean(scores)))    

sample_submission = pd.DataFrame.from_dict({'id': test['id']})
oof = pd.DataFrame.from_dict({'id': train['id']})
for c in list_classes:
    oof[c] = np.zeros(len(train))
    sample_submission[c] = np.zeros(len(test))
    
sample_submission[list_classes] = predict
sample_submission.to_csv("test_grucnn_fl3.csv", index=False)

oof[list_classes] = oof_predict
oof.to_csv('train_grucnn_fl3.csv', index=False)