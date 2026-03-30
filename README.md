# Reactree-MCTS
The current workflow includes:
1. The agent is given user instruction query
2. Agent is asked to act as a shopping agent and has to buy the product matching the user instruction.
3. It has design it's own search query and navigate through page.

There are three expand modes in reactree:
    1. sequence - follow one by one, if one fails whole sequence fails
    2. fallback - (attempt subgoals in order until one succeeds)
    3. parallel: achieve subgoals in parallel (this enables tasks to continue independtly, even if one subgoal fails)

Now in webshop, best search query need not include 


### Think out to improve search query which is actually put in the search box
changed the prompt to adjust for search query, 


### Current Challenges:
- parallel can be used to explore multiple products 
- fallback can be used if any one of the condition is failed, then backtrack
- sequence can be used to in situation's where 

### Current Status:
1. In the first search query the agent is almost giving whole user instruction and then further deciding to decompose the goal.



### Sample decomposition(human)
User query: "Find me non slip, easy clean computer armoires with pu leather with color: gray, and size: 90x40cm, and price lower than 60.00 dollars"

search inst: computer armoire gray 90x40cm under 60 pu leather 

<!-- ### model details

{'quantize.imatrix.file': './Meta-Llama-3.1-8B-Instruct-GGUF_imatrix.dat', 'quantize.imatrix.chunks_count': '68', 'tokenizer.chat_template': "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}", 'tokenizer.ggml.eos_token_id': '128009', 'general.type': 'model', 'tokenizer.ggml.bos_token_id': '128000', 'tokenizer.ggml.pre': 'smaug-bpe', 'tokenizer.ggml.model': 'gpt2', 'llama.embedding_length': '4096', 'llama.vocab_size': '128256', 'llama.attention.head_count_kv': '8', 'general.finetune': 'Instruct', 'general.file_type': '14', 'llama.block_count': '32', 'general.size_label': '8B', 'llama.attention.layer_norm_rms_epsilon': '0.000010', 'llama.feed_forward_length': '14336', 'general.quantization_version': '2', 'llama.rope.dimension_count': '128', 'general.license': 'llama3.1', 'llama.attention.head_count': '32', 'quantize.imatrix.entries_count': '224', 'llama.context_length': '131072', 'general.architecture': 'llama', 'general.basename': 'models-meta-llama-Meta-Llama-3.1', 'llama.rope.freq_base': '500000.000000', 'quantize.imatrix.dataset': 'group_40.txt', 'general.name': 'Models Meta Llama Meta Llama 3.1 8B Instruct'} -->