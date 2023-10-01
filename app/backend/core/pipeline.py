from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import VectorDBQA
from .models import FredT5, FakeModel
from .utils import tokenize, lemmatize
from typing import List
from .vectordb import create_client
from langchain.vectorstores import Chroma


class Pipeline:
    """
    Naive Retrieval QA pipeline. RAG pattern
    """

    def __init__(self, config: dict):
        self.config = config
        # self.collection = create_client()
        self.collection = Chroma("main_app2")
        self.last_id = 0

        if self.config["model"] == "test":
            model = FakeModel(return_value=True, time_sleep=self.config["time_sleep"])
        elif self.config["model"] == "fredt5":
            model = FredT5(do_sample=True)
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

        sent = lemmatize(tokenize(query))

        documents = []
        for word in sent:
            loader = WikipediaLoader(word)
            data = loader.load()
            documents.append(data)

        return documents

    def collecting(self, documents: List[List[Document]]):
        all_splits = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config["chunk_size"],
            chunk_overlap=self.config["chunk_overlap"],
        )

        for doc in documents:
            all_splits.append(text_splitter.split_documents(doc))

        #self.collection.add_documents(documents=all_splits, ids=list(range(self.last_id, len(all_splits))))
        self.collection.add_texts(all_splits)

    def query(self, text: str) -> str:
        self.collecting(self.get_documents(text))
        res = self.chain.run(text)
        return res
