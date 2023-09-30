from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from models import FredT5
from langchain.chains import ConversationChain


class Pipeline:
    """
    Naive Retrieval QA pipeline
    """

    def __init__(self):
        self.source = "wiki"

    @staticmethod
    def run():
        loader = WikipediaLoader("AGI")
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=32)
        all_splits = text_splitter.split_documents(data)

        print(len(all_splits))

        q = "Who work in field of AGI?"

        llm = FredT5(do_sample=True)
        conversation = ConversationChain(llm=llm)
        res = conversation.run(q)
        print(res)

    @staticmethod
    async def arun():
        pass


if __name__ == "__main__":
    Pipeline.run()
