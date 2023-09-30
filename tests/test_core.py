import pytest
from app.backend.core.models import FakeModel
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


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


