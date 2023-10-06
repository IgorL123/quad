import re
import string
import nltk
import numpy as np
from gensim.models import KeyedVectors
from flask import current_app
from typing import List
import spacy
from spacy.lang.ru.examples import sentences


def tokenize_spacy(text: str) -> List[str]:
    # TODO переделать и ускорить
    stopwords = set(nltk.corpus.stopwords.words("russian"))
    stopwords = set(nltk.corpus.stopwords.words("english")) | stopwords
    punctuation = re.compile(r"[" + string.punctuation + string.digits + "]")
    nlp = spacy.load("ru_core_news_sm")
    doc = nlp(text)
    return [
        _.lemma_
        for _ in doc
        if _.lemma_ not in stopwords and not re.search(punctuation, _.lemma_)
    ]


def tokenize(text: str) -> List[str]:
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
