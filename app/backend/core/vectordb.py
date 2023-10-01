import chromadb
from chromadb.config import Settings
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils import embedding_functions
from .utils import fasttext
import numpy as np
from typing import cast

client = None


class FasttextEmbeddings(EmbeddingFunction):
    """
    Cant work with english - only russian
    """

    def __call__(self, texts: Documents) -> Embeddings:
        return cast(Embeddings, [fasttext(text).astype(np.float32) for text in texts])


def create_client(retriever=None):
    global client
    """
    Create Croma client and main collection
    :return:
    """
    settings = Settings(anonymized_telemetry=False)
    client = chromadb.Client(settings)

    # emb_fn = FasttextEmbeddings()
    emb_fn = embedding_functions.DefaultEmbeddingFunction()
    collection = client.create_collection(
        name="main_app_collection",
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"},
        get_or_create=True,
    )

    return collection


def stop_client():
    global client
    if client:
        client.stop()
