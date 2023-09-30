import re
import string
import nltk
import numpy as np
from gensim.models import KeyedVectors
from flask import current_app


def tokenize(text):
    russian_stopwords = set(nltk.corpus.stopwords.words("russian"))
    punctuation = re.compile(
        r"[" + string.punctuation + string.ascii_letters + string.digits + "]"
    )
    words = nltk.word_tokenize(text)
    words = [word for word in words if word not in russian_stopwords]

    return [word for word in words if not re.search(punctuation, word)]


def fasttext(text):
    tokens = tokenize(text)
    model = KeyedVectors.load(current_app.config["FASTTEXT"])

    def use_model(model_input):
        return model[str(model_input)]

    text_array = np.array(list(map(use_model, tokens)))
    res = text_array.mean(axis=0)  # TODO tfidf
    return res
