from __future__ import absolute_import

import nltk

# TODO
def tokenize(text):
    return [nltk.stem.WordNetLemmatizer().lemmatize(token) 
            for token in nltk.tokenize.TweetTokenizer(False, False, False).tokenize(text)]