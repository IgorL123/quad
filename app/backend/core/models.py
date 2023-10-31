from typing import Any, List, Mapping, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from time import sleep
from random import shuffle


def preload():
    tokenizer = AutoTokenizer.from_pretrained("SiberiaSoft/SiberianFredT5-instructor")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "SiberiaSoft/SiberianFredT5-instructor"
    )
    model.eval()

    return model, tokenizer


class FredT5(LLM):
    """
    Based on SiberianFredT5-instructor from SiberiaSoft
    """

    do_sample: bool
    """
    temperature: float
    max_new_tokens: int
    top_p: float
    top_k: int
    repetition_penalty: float
    no_repeat_ngram_size: int
    """

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        model, tokenizer = preload()

        data = tokenizer(
            "<SC6>" + prompt + "\nОтвет: <extra_id_0>", return_tensors="pt"
        )
        data = {k: v.to(model.device) for k, v in data.items()}
        output_ids = model.generate(
            **data,
            do_sample=self.do_sample,
            temperature=0.2,
            max_new_tokens=512,
            top_p=0.95,
            top_k=5,
            repetition_penalty=1.03,
            no_repeat_ngram_size=2,
        )[0]
        out = tokenizer.decode(output_ids.tolist())
        out = out.replace("<s>", "").replace("</s>", "")
        return out[17:]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"do_sample": self.do_sample}


class FakeModel(LLM):
    """
    For testing & debugging
    """

    return_value: bool
    time_sleep: int

    @property
    def _llm_type(self) -> str:
        return "test"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        value = None
        sleep(self.time_sleep)

        if self.return_value:
            value = prompt.split()
            shuffle(value)
            value = " ".join(value)

        return value

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"return_value": self.return_value, "time_sleep": self.time_sleep}
