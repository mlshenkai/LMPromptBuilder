# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/5/19 2:37 PM
# @File: llama_model
# @Email: mlshenkai@163.com
"""
llama langchain
"""
from typing import Optional, List, Any, Dict
import torch
from langchain.callbacks.manager import CallbackManagerForLLMRun
from pydantic import root_validator
from langchain.llms.base import LLM
from src.LMBuilder.LLM.models import LlamaModelPeft


class Llama(LLM):
    client: Any
    model_type: str = "llama"
    model_name: str = None
    peft_name: str = None
    args: Any
    use_cuda = torch.cuda.is_available()
    cuda_device = -1
    kwargs: Any = {}

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        llama_peft = LlamaModelPeft(
            model_type=values["model_type"],
            model_name=values["model_name"],
            peft_name=values["peft_name"],
            args=values["args"],
            use_cuda=values["use_cuda"],
            cuda_device=values["cuda_device"],
            **values["kwargs"]
        )
        values["client"] = llama_peft
        cls.client = llama_peft
        return values

    def _default_params(self) -> Dict[str, Any]:
        return {
            "client": self.client,
            "model_type": self.model_type,
            "model_name": self.model_name,
            "peft_name": self.peft_name,
            "args": self.args,
            "use_cuda": self.use_cuda,
            "cuda_device": self.cuda_device,
            "kwargs": self.kwargs,
        }

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        response = self.client.predict([prompt], max_length=self.args.max_seq_length)[0]
        return response


    @property
    def _llm_type(self) -> str:
        return "Llama"
