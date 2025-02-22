Changes made to implement Mistral:

expel.yaml
- changed llm name to mistral / llama

llm.py
- modify LLM_CLS
- create MistralWrapper(), LlamaWrapper()

expel/utils.py
- implemented llama support in token_counter()

Steps to train with hugginface models:
1. pip install transformers
2. Run “huggingface-cli login” in command line
3. Get hugging face access token from profile -> access tokens -> create new token
Enter the token
4. If unable to authenticate through git credential, reauthenticate by: git config --global credential.helper store
Repeat steps 2 and 4

