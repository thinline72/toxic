import tensorflow as tf
import numpy as np 
import pandas as pd 
from keras.models import Model
from keras.layers import Dense, Embedding, Input, Concatenate, Conv1D, Activation, TimeDistributed, Flatten, RepeatVector, Permute,multiply
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout, GRU, GlobalAveragePooling1D, MaxPooling1D, SpatialDropout1D, BatchNormalization
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

max_features = 160000
maxlen = 200

sia = SentimentIntensityAnalyzer()

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

import sys
import regex as re

FLAGS = re.MULTILINE | re.DOTALL

def hashtag(text):
    text = text.group()
    hashtag_body = text[1:]
    if hashtag_body.isupper():
        result = " {} ".format(hashtag_body.lower())
    else:
        result = " ".join(["<hashtag>"] + re.split(r"(?=[A-Z])", hashtag_body, flags=FLAGS))
    return result

def allcaps(text):
    text = text.group()
    return text.lower() + ' <allcaps>'

def tokenize(text):
    # Different regex parts for smiley faces
    eyes = r"[8:=;]"
    nose = r"['`\-]?"
    text = re.sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>", text, flags=FLAGS)
    text = re.sub(r"@\w+", "<user>", text, flags=FLAGS)
    text = re.sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>", text, flags=FLAGS)
    text = re.sub(r"{}{}p+".format(eyes, nose), "<lolface>", text, flags=FLAGS)
    text = re.sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>", text, flags=FLAGS)
    text = re.sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>", text, flags=FLAGS)
    text = re.sub(r"/"," / ", text, flags=FLAGS)
    text = re.sub(r"<3","<heart>", text, flags=FLAGS)
    text = re.sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>", text, flags=FLAGS)
    text = re.sub(r"#\S+", hashtag, text, flags=FLAGS)
    text = re.sub(r"([!?.]){2,}", r"\1 <repeat>", text, flags=FLAGS)
    text = re.sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>", text, flags=FLAGS)
    text = re.sub(r"([A-Z]){2,}", allcaps, text, flags=FLAGS)

    return text.lower()
    
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()    

def tokenize_sentences(sentence):
    sentence = sentence.encode('ascii', errors='ignore').decode('ascii', errors='ignore')
    sentence = tokenize(sentence)
    tokens = tknzr.tokenize(sentence)
    result = tokens
    return ' '.join(result).replace('  ', ' ').strip()
    
print('preprocessing')
list_sentences_train = train["comment_text"].fillna("").apply(tokenize_sentences).values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("").apply(tokenize_sentences).values

print('loading embeddings vectors')
def get_coefs(word,*arr): 
    return word, np.asarray(arr, dtype='float32')
embeddings_index = {}
for o in open('../input/glove.twitter.27B.200d.txt', encoding="utf8"):
    w,v = get_coefs(*o.strip().split(' '))
    if len(v) == 200:
        embeddings_index[w] = v

print('mean text len:',train["comment_text"].str.count('\S+').mean())
print('max text len:',train["comment_text"].str.count('\S+').max())

tokenizer = Tokenizer(filters='"#$%&*+-/;=@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(list(list_sentences_train)) # 
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
print('padding sequences')
X_train = {}
X_test = {}
X_train['text'] = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen, padding='post', truncating='post')
X_test['text'] = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen, padding='post', truncating='post')

print('numerical variables')
train['num_words'] = train.comment_text.str.count('\S+')
test['num_words'] = test.comment_text.str.count('\S+')
train['num_comas'] = train.comment_text.str.count('\.')
test['num_comas'] = test.comment_text.str.count('\.')
train['num_bangs'] = train.comment_text.str.count('\!')
test['num_bangs'] = test.comment_text.str.count('\!')
train['num_quotas'] = train.comment_text.str.count('\"')
test['num_quotas'] = test.comment_text.str.count('\"')
train['avg_word'] = train.comment_text.str.len() / (1 + train.num_words)
test['avg_word'] = test.comment_text.str.len() / (1 + test.num_words)
#print('sentiment')
#train['sentiment'] = train.comment_text.apply(lambda s : sia.polarity_scores(s)['compound'])
#test['sentiment'] = test.comment_text.apply(lambda s : sia.polarity_scores(s)['compound'])
scaler = MinMaxScaler()
X_train['num_vars'] = scaler.fit_transform(train[['num_words','num_comas','num_bangs','num_quotas','avg_word']])
X_test['num_vars'] = scaler.transform(test[['num_words','num_comas','num_bangs','num_quotas','avg_word']])

embed_size = 200

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()

print('create embedding matrix')
word_index = tokenizer.word_index
nb_words = len(word_index)
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

del embeddings_index
gc.collect()  

def _train_model(model, batch_size, train_x, train_y, val_x, val_y):
    best_loss = -1
    best_weights = None
    best_epoch = 0

    current_epoch = 0

    while True:
        K.set_value(model.optimizer.lr, 0.001 * (0.85**current_epoch)) 
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
        if auc > best_loss or best_loss == -1:
            best_loss = auc
            best_weights = model.get_weights()
            best_epoch = current_epoch
        else:
            if current_epoch - best_epoch == 5:
                break

    model.set_weights(best_weights)
    return model

from keras.engine.topology import Layer
from keras.layers import Bidirectional, Dropout, SpatialDropout1D, CuDNNGRU
from keras.optimizers import RMSprop


def get_model_cnn(X_train):
    global embed_size
    inp = Input(shape=(maxlen, ), name="text")
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=True)(inp)
    x = SpatialDropout1D(0.4)(x)
    c2 = Conv1D(128, 3, activation="relu")(x)
    c2 = MaxPooling1D(maxlen-3+1)(c2)

    c3 = Conv1D(128, 4, activation="relu")(x)
    c3 = MaxPooling1D(maxlen-4+1)(c3)

    c4 = Conv1D(128, 5, activation="relu")(x)
    c4 = MaxPooling1D(maxlen-5+1)(c4)

    x = Concatenate(axis=1)([c2,c3,c4])
    x = Conv1D(128, 3, activation="relu")(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.3)(x)
    #fc1 = Dropout(0.4)(Dense(128, activation='relu', kernel_initializer='he_normal')(x))
    #fc2 = Dropout(0.4)(Dense(32, activation='relu', kernel_initializer='he_normal')(fc1))
    #fc3 = Dropout(0.4)(Dense(100, activation='relu', kernel_initializer='he_normal')(fc2))
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=[inp], outputs=x)
    opt = Nadam(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
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
    for c in ['text','num_vars']:
        kfold_X_train[c] = X_train[c][train_index]
        kfold_X_valid[c] = X_train[c][test_index]

    model = get_model_cnn(X_train)
    #print(model.summary())
    #model = _train_model(model, batch_size, kfold_X_train, y_train, kfold_X_valid, y_test)
    model.fit(kfold_X_train, y_train, batch_size=batch_size, epochs=6)
    model.save_weights('model_all_weights_'+str(ifold)+'.h5')
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
sample_submission.to_csv("test_simple_cnn.csv", index=False)

oof[list_classes] = oof_predict
oof.to_csv('train_simple_cnn.csv', index=False)