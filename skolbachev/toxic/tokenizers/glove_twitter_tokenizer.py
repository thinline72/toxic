from __future__ import absolute_import

import sys
import regex as re
import nltk

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

def tokenize(text, vocab=None, lower=True, strip=True, lemmatize=True):
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

    tokens = nltk.tokenize.TweetTokenizer().tokenize(text)
    if lower:
        tokens = [token.lower() for token in tokens]
    
    if vocab is not None and (strip or lemmatize):
        new_tokens = []
        for token in tokens:
            new_token = token
            if strip and new_token not in vocab:
                new_token = re.sub(r'(.)\1{2,}', r'\1', new_token)
                if new_token not in vocab:
                    new_token = re.sub(r'(.)\1{1,}', r'\1', new_token)
            if lemmatize and new_token not in vocab:
                new_token = nltk.stem.WordNetLemmatizer().lemmatize(new_token)
            new_tokens.append(new_token)
        
        tokens = new_tokens
            
    return tokens