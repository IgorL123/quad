import chromadb
from chromadb.config import Settings
from chromadb import Documents, EmbeddingFunction, Embeddings
from .utils import fasttext


class FasttextEmbeddings(EmbeddingFunction):
    def __call__(self, texts: Documents) -> Embeddings:
        embeddings = []
        for doc in texts:
            embeddings.append(fasttext(doc))
        return embeddings


def create_client():
    """
    Create Croma client and main collection
    :return:
    """
    settings = Settings(anonymized_telemetry=False)
    client = chromadb.Client(settings)

    collection = client.create_collection(
        name="main_app_collection",
        embedding_function=FasttextEmbeddings,
        metadata={"hnsw:space": "cosine"},
        get_or_create=True,
    )

    return collection
