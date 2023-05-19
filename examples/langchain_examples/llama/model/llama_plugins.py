# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/5/19 6:37 PM
# @File: llama_plugins
# @Email: mlshenkai@163.com
import os
os.environ["SERPAPI_API_KEY"] = "947087c28df96809d878a9f86547b740f46d33b129ec4ad8313ce0b2a84eaca3"
from examples.langchain_examples.llama.model.llama_model import Llama
from src.models.peft.config.model_args import LlamaArgs
from langchain.agents import load_tools, initialize_agent, AgentType
llm = Llama(args=LlamaArgs())

tools = load_tools(["serpapi"], llm=llm)

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

agent.run("今年的英雄联盟MSI胜者组决赛T1和JDG的比赛结果如何?")

