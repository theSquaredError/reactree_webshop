from guidance import system, user, assistant, gen
from guidance.models import Transformers

from dotenv import load_dotenv
import os
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
# Could also do LlamaCpp or many other models
model_id = "meta-llama/Llama-3.1-8B-Instruct"
llama_lm = Transformers(model_id, use_auth_token=hf_token)

