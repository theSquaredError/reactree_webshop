import os
os.environ["GUIDANCE_DISABLE_METRICS"] = "1"
import warnings
warnings.filterwarnings("ignore")
from guidance import select
from guidance.models import LlamaCpp
from guidance.chat import ChatTemplate

class Llama3Template(ChatTemplate):

    def user(self, content):
        return f"<|start_header_id|>user<|end_header_id|>\n{content}<|eot_id|>"

    def assistant(self, content):
        return f"<|start_header_id|>assistant<|end_header_id|>\n{content}<|eot_id|>"

lm = LlamaCpp(
    model="/Users/vikas/Downloads/llama3-local/model.gguf",
    chat_template = Llama3Template,
    temperature=0
)

lm += "The sentiment is " + select(
    ["positive", "negative", "neutral"],
    name="sentiment"
)

print(lm["sentiment"])

https://github.com/theSquaredError/reactree_webshop.git