from webshop_solution.llm.ollama_inference import ollama_infer
from webshop_solution.prompts.webshop_prompt import get_webshop_reactree_prompt
import openai

class WebShopLlmAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_name = cfg.llm_agent.model_name
        self.prompt = None
        self.backend = cfg.llm_agent.backend  # 'openai' or 'ollama'

    def reset(self, nl_inst_info, init_obs):
        self.prompt = self.load_prompt(nl_inst_info, init_obs)

    def plan_next_step(self, skill_set):
        if self.backend == "openai":
            text = self._openai_generate(self.prompt)
        elif self.backend == "ollama":
            text = ollama_infer(self.prompt, model=self.model_name)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        # Parse for Think, Act, Expand
        if text.startswith("Think:"):
            reasoning = text[len("Think:"):].strip()
            return {'next_step_class': 'Think', 'next_step': reasoning}
        elif text.startswith("Act:"):
            action = text[len("Act:"):].strip()
            return {'next_step_class': 'Act', 'next_step': action}
        elif text.startswith("Expand:"):
            control_flow = self._parse_control_flow(text)
            subgoals = self._parse_subgoals(text)
            return {'next_step_class': 'Expand', 'next_step': {'control_flow': control_flow, 'conditions': ', '.join(subgoals)}}
        else:
            return {'next_step_class': 'Error', 'next_step': text}

    def _openai_generate(self, prompt):
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role": "system", "content": prompt}],
            temperature=0.7,
            max_tokens=512,
        )
        return response['choices'][0]['message']['content']

    def load_prompt(self, nl_inst_info, init_obs):
        nl_inst = nl_inst_info['nl_inst']
        prompt = get_webshop_reactree_prompt(nl_inst)
        prompt += f"\nObservation: {init_obs}\n"
        return prompt
        return (
            f"You are a WebShop agent. Your task is to: {nl_inst}\n"
            f"Observation: {init_obs}\n"
            "You can Think, Act, or Expand. If you Expand, specify control flow and subgoals.\n"
        )

    def add_obs(self, obs_text):
        self.prompt += f"\nObservation: {obs_text}"

    def _parse_control_flow(self, text):
        for line in text.splitlines():
            if "- control flow:" in line:
                return line.split(":")[1].strip()
        return "sequence"

    def _parse_subgoals(self, text):
        for line in text.splitlines():
            if "- subgoals:" in line:
                return [s.strip() for s in line.split(":")[1].split(",") if s.strip()]
        return []