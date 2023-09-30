from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from models import FredT5
import re
import string
import nltk
import numpy as np
from langchain.chains import ConversationChain


def tokenize(text):
    russian_stopwords = set(nltk.corpus.stopwords.words("russian"))

    punctuation = re.compile(
        r"[" + string.punctuation + string.ascii_letters + string.digits + "]"
    )

    words = nltk.word_tokenize(text)

    words = [word for word in words if word not in russian_stopwords]

    return [word for word in words if not re.search(punctuation, word)]


def fasttext(text, model, *args):
    tokens = tokenize(text)

    def use_model(model_input):
        return model[str(model_input)]

    text_array = np.array(list(map(use_model, tokens)))
    res = text_array.mean(axis=0)
    return res


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


if __name__ == "__main__":
    Pipeline.run()
