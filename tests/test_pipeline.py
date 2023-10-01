import pytest
from app.backend.core.pipeline import Pipeline

config = {
    "model": "test",
    "time_sleep": 1,
    "chain_type": "stuff",
    "source": "wiki",
    "chunk_size": 512,
    "chunk_overlap": 16,

}


def test_fake_pipeline():

    pipe = Pipeline(config)
    res = pipe.query("Who is the President of USA?")
    assert res == 0

    res = pipe.query("Who is Vladimir Putin?")
    assert res == 1

