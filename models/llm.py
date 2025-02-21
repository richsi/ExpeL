from typing import Callable, List
import time

from langchain.chat_models import ChatOpenAI
from langchain.schema import ChatMessage
import openai
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class GPTWrapper:
    def __init__(self, llm_name: str, openai_api_key: str, long_ver: bool):
        self.model_name = llm_name
        if long_ver:
            llm_name = 'gpt-3.5-turbo-16k'
        self.llm = ChatOpenAI(
            model=llm_name,
            temperature=0.0,
            openai_api_key=openai_api_key,
        )

    def __call__(self, messages: List[ChatMessage], stop: List[str] = [], replace_newline: bool = True) -> str:
        kwargs = {}
        if stop != []:
            kwargs['stop'] = stop
        for i in range(6):
            try:
                output = self.llm(messages, **kwargs).content.strip('\n').strip()
                break
            except openai.error.RateLimitError:
                print(f'\nRetrying {i}...')
                time.sleep(1)
        else:
            raise RuntimeError('Failed to generate response')

        if replace_newline:
            output = output.replace('\n', '')
        return output


class MistralWrapper:
  def __init__(self):
    self.model_name = "mistralai/Mistral-7B-Instruct-v0.1" 
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    self.llm = AutoModelForCausalLM.from_pretrained(self.model_name)
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.llm.to(self.device)

  def __call__(self, messages: List[ChatMessage], stop: List[str] = [], replace_newline: bool = True) -> str:
    prompt = self._format_messages(messages)
    for i in range(6):
      try:
          # tokenize prompt
          input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
          output_ids = self.llm.generate(
            input_ids,
            max_length=input_ids.shape[1] + 200,
            do_sample=False
          )
          output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
          response = output[len(prompt):].strip()
          break
      except Exception as e:
        print(f"\nRetrying {i} due to error: {e}")
        time.sleep(1)
    
    else:
      raise RuntimeError("Failed to generate response - Mistral")

    if stop:
      for stop_seq in stop:
        if stop_seq in response:
          response = response.split(stop_seq)[0]
          break
    if replace_newline:
      response = response.replace('\n', '')
    return response


  def _format_messages(self, messages: List[ChatMessage]) -> str:
    conversation = ""
    for msg in messages:
        if hasattr(msg, "role"):
            role = msg.role
        elif isinstance(msg, HumanMessage):
            role = "Human"
        elif isinstance(msg, AIMessage):
            role = "AI"
        elif isinstance(msg, SystemMessage):
            role = "System"
        else:
            role = "Unknown"
        conversation += f"{role}: {msg.content}\n"
    return conversation


class LlamaWrapper:
  def __init__(self):
    self.model_name = "meta-llama/Llama-3.2-1B"
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    self.llm = AutoModelForCausalLM.from_pretrained(self.model_name)
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.llm.to(self.device)

  def __call__(self, messages: List[ChatMessage], stop: List[str] = [], replace_newline: bool = True) -> str:
    prompt = self._format_messages(messages)
    for i in range(6):
      try:
          # tokenize prompt
          input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
          output_ids = self.llm.generate(
            input_ids,
            max_length=input_ids.shape[1] + 200,
            do_sample=False
          )
          output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
          response = output[len(prompt):].strip()
          break
      except Exception as e:
        print(f"\nRetrying {i} due to error: {e}")
        time.sleep(1)
    
    else:
      raise RuntimeError("Failed to generate response - Mistral")

    if stop:
      for stop_seq in stop:
        if stop_seq in response:
          response = response.split(stop_seq)[0]
          break
    if replace_newline:
      response = response.replace('\n', '')
    return response


  def _format_messages(self, messages: List[ChatMessage]) -> str:
    conversation = ""
    for msg in messages:
        conversation += f"System: {msg.content}\n"
    return conversation


def LLM_CLS(llm_name: str, openai_api_key: str, long_ver: bool) -> Callable:
    if 'gpt' in llm_name:
        return GPTWrapper(llm_name, openai_api_key, long_ver)
    elif 'mistral' in llm_name:
        return MistralWrapper()
    elif 'llama' in llm_name:
        return LlamaWrapper()
    else:
        raise ValueError(f"Unknown LLM model name: {llm_name}")
