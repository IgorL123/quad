import pytest
from .test_backend import client
from app.backend.core.models import FakeModel
from app.backend.core.vectordb import create_client, stop_client
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from app.backend.core.utils import fasttext
from numpy import float32, issubdtype


def test_fake_model():
    fake = FakeModel(return_value=True, time_sleep=1)
    prompt_template = "Tell me a {adjective} joke"
    prompt = PromptTemplate(
        input_variables=["adjective"], template=prompt_template
    )
    llm = LLMChain(llm=fake, prompt=prompt)

    completion = llm.predict(adjective="funny")
    assert len(completion.split()) == 5
    assert completion.find("me") != -1
    assert completion.find("a") != -1
    assert completion.find("joke") != -1
    assert completion.find("funny") != -1
    assert completion.find("Tell") != -1


def test_fasttext(client):
    doc = "Съешь же ещё этих мягких французских булок, да выпей чаю"
    emb = fasttext(doc)
    assert len(emb) == 300
    assert issubdtype(emb.dtype, float32)


def test_chroma(client):

    doc1 = "Natural Language Processing (NLP) is a domain of research whose objective is to analyze and understand it"
    doc2 = "Съешь же ещё этих мягких французских булок, да выпей чаю"
    doc3 = "Обычно панграммы используют для презентации шрифтов, чтобы можно было в одной фразе рассмотреть все глифы"
    test_text = "We introduce this question answering NLP project"

    collection = create_client()
    collection.add(
        documents=[
            doc1,
            doc2,
            doc3,
        ],
        metadatas=[
            {"testing": True},
            {"testing": True},
            {"testing": True},
        ],
        ids=[
            "1",
            "2",
            "3",
        ]
    )

    texts = collection.query(
        query_texts=[test_text],
        n_results=1,
    )
    assert texts["documents"][0][0] == doc1

    texts = collection.get(
        where={"testing": True}
    )
    assert len(texts["documents"]) == 3
    collection.delete(
        where={"testing": True}
    )
    texts = collection.get(
        where={"testing": True}
    )
    assert len(texts["documents"]) == 0

    stop_client()
