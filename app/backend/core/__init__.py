from .vectordb import create_client
from .pipeline import Pipeline


config = {
    "model": "fredt5",
    "do_sample": True,
    "chain_type": "stuff",
    "source": "wiki",
    "chunk_size": 512,
    "chunk_overlap": 16,
}

test_config = {
    "model": "test",
    "time_sleep": 1,
    "chain_type": "stuff",
    "source": "wiki",
    "chunk_size": 512,
    "chunk_overlap": 16,
}

main = Pipeline(config)
