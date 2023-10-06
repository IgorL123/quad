import pytest
from app.backend.core.pipeline import Pipeline
from app.backend.core.utils import tokenize_spacy
from typing import List

config = {
    "model": "test",
    "time_sleep": 1,
    "chain_type": "stuff",
    "source": "wiki",
    "chunk_size": 512,
    "chunk_overlap": 16,
}

config_prod = {
    "model": "fredt5",
    "do_sample": True,
    "chain_type": "stuff",
    "source": "wiki",
    "chunk_size": 512,
    "chunk_overlap": 16,
}

sample_query_1 = "Who is the President of USA?"
sample_query_2 = "Who is Vladimir Putin?"

sample_query_3 = "Кто сейчас является президентом США?"
sample_query_4 = "Кто такой Владимир Путин?"


def test_documents_fetching():
    pipe = Pipeline(config)
    docs = pipe.get_documents(sample_query_3)
    assert isinstance(docs, List)
    assert len(docs) > 0


def test_tokenizer():
    res = tokenize_spacy(sample_query_3)
    assert len(res) == 3

    res = tokenize_spacy(sample_query_1)
    assert len(res) == 2


def test_fake_pipeline():
    pipe = Pipeline(config)
    res = pipe.query(sample_query_1)
    assert len(res) > 2

    res = pipe.query(sample_query_2)
    assert len(res) > 2


def test_fredt5_pipeline():
    pipe = Pipeline(config_prod)
    res = pipe.query(sample_query_1)
    assert len(res) > 2
