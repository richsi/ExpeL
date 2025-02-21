Changes made to implement Mistral:

expel.yaml
- changed llm name to mistral

train.py 
- if mistral, dont add openai_api_key

llm.py
- modify LLM_CLS
- create MistralWrapper()

