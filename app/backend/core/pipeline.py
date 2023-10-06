from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import VectorDBQA
from .models import FredT5, FakeModel
from .utils import tokenize_spacy
from typing import List
from .vectordb import create_client
from langchain.vectorstores import Chroma
from langchain.embeddings.spacy_embeddings import SpacyEmbeddings

# TODO добавить разделение на языки ru/en
# TODO добавить answers cache
# TODO переписать ебучий vectorstore


class Pipeline:
    """
    Retrieval QA pipeline. RAG pattern
    """

    def __init__(self, config: dict):
        self.config = config

        # TODO fix create_client()
        # self.collection = create_client()

        emb_fn = SpacyEmbeddings()
        self.collection = Chroma("main_app2", embedding_function=emb_fn)
        self.last_id = 0

        if self.config["model"] == "test":
            model = FakeModel(return_value=True, time_sleep=self.config["time_sleep"])
        elif self.config["model"] == "fredt5":
            model = FredT5(do_sample=self.config["do_sample"])
        else:
            raise NotImplementedError

        self.chain = VectorDBQA.from_chain_type(
            llm=model,
            chain_type=self.config["chain_type"],
            vectorstore=self.collection,
        )

    def get_documents(self, query: str) -> List[List[Document]]:
        if self.config["source"] != "wiki":
            raise NotImplementedError

        sent = tokenize_spacy(query)

        # TODO распаралелить
        documents = []
        for _ in sent:
            loader = WikipediaLoader(_, lang="ru")
            data = loader.load()
            documents.append(data)

        if len(documents) == 0:
            raise Exception("Not found")
        return documents

    def collecting(self, documents: List[List[Document]]):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config["chunk_size"],
            chunk_overlap=self.config["chunk_overlap"],
        )

        # TODO распидарасить
        for doc in documents:
            splits = text_splitter.split_documents(doc)
            self.collection.add_documents(splits)
            # self.collection.add_documents(documents=all_splits, ids=list(range(self.last_id, len(all_splits))))

    def query(self, text: str) -> str:
        self.collecting(self.get_documents(text))
        res = self.chain.run(text)
        return res
