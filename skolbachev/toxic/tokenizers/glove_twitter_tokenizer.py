from __future__ import absolute_import

import sys
import nltk
import regex

"""
preprocess-twitter.py from https://gist.github.com/ppope/0ff9fa359fb850ecf74d061f3072633a

python preprocess-twitter.py "Some random text with #hashtags, @mentions and http://t.co/kdjfkdjf (links). :)"

Script for preprocessing tweets by Romain Paulus
with small modifications by Jeffrey Pennington
with translation to Python by Motoki Wu

Translation of Ruby script to create features for GloVe vectors for Twitter data.
http://nlp.stanford.edu/projects/glove/preprocess-twitter.rb
"""

FLAGS = regex.MULTILINE | regex.DOTALL

def hashtag(text):
    text = text.group()
    hashtag_body = text[1:]
    if hashtag_body.isupper():
        result = " {} ".format(hashtag_body.lower())
    else:
        result = " ".join([" _1_hashtag_2_ "] + regex.split(r"(?=[A-Z])", hashtag_body, flags=FLAGS))
    return result

def allcaps(text):
    text = text.group()
    return text.lower() + " _1_allcaps_2_ "


def tokenize(text):
    # Different regex parts for smiley faces
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    # function so code less repetitive
    def re_sub(pattern, repl):
        return regex.sub(pattern, repl, text, flags=FLAGS)

    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", " _1_url_2_ ")
    text = re_sub(r"@\w+", " _1_user_2_ ")
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), " _1_smile_2_ ")
    text = re_sub(r"{}{}p+".format(eyes, nose), " _1_lolface_2_ ")
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), " _1_sadface_2_ ")
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), " _1_neutralface_2_ ")
    text = re_sub(r"/"," / ")
    text = re_sub(r"<3"," _1_heart_2_ ")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", " _1_number_2_ ")
    text = re_sub(r"#\S+", hashtag)
    text = re_sub(r"([!?.]){2,}", r"\1  <repeat> ")
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2  _1_elong_2_ ")

    ## -- I just don't understand why the Ruby script adds <allcaps> to everything so I limited the selection.
    # text = re_sub(r"([^a-z0-9()<>'`\-]){2,}", allcaps)
    text = re_sub(r"([A-Z]){2,}", allcaps)

    return [token.replace('_1_', '<').replace('_2_', '>').lower() for token in nltk.word_tokenize(text)]