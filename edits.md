Changes made to implement Mistral:

expel.yaml
- changed llm name to mistral / llama

llm.py
- modify LLM_CLS
- create MistralWrapper(), LlamaWrapper()

expel/utils.py
- implemented llama support in token_counter()