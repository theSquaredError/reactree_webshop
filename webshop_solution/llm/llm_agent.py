from __future__ import annotations

import os

os.environ.setdefault("GUIDANCE_DISABLE_METRICS", "1")

import guidance
from guidance.chat import ChatTemplate
from guidance.models import LlamaCpp

from webshop_solution.prompts.webshop_prompt import get_webshop_reactree_prompt


class Llama3Template(ChatTemplate):
    def get_role_start(self, role: str) -> str:
        if role == "system":
            return "<|start_header_id|>system<|end_header_id|>\n"
        if role == "user":
            return "<|start_header_id|>user<|end_header_id|>\n"
        if role == "assistant":
            return "<|start_header_id|>assistant<|end_header_id|>\n"
        raise ValueError(f"Unsupported role: {role}")

    def get_role_end(self, role: str) -> str:
        return "<|eot_id|>\n"

    def user(self, content):
        return f"<|start_header_id|>user<|end_header_id|>\n{content}<|eot_id|>"

    def assistant(self, content):
        return f"<|start_header_id|>assistant<|end_header_id|>\n{content}<|eot_id|>"


class WebShopLlmAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_name = cfg.llm_agent.model_name
        self.prompt = ""

        self.base_llm = LlamaCpp(
            model=self.model_name,  # absolute path to .gguf
            chat_template=Llama3Template,
            n_ctx=getattr(cfg.llm_agent, "n_ctx", 4096),
            temperature=cfg.llm_agent.temperature,
            echo=False,
        )
        self.llm = self.base_llm.copy()

    def reset(self, nl_inst_info, init_obs_text: str):
        self.prompt = self.load_prompt(nl_inst_info, init_obs_text)

        # Guidance LlamaCpp does not expose reset() in many versions.
        # Start each episode from a fresh copy of the base model state.
        self.llm = self.base_llm.copy()

        with guidance.user():
            self.llm += self.prompt

    def load_prompt(self, nl_inst_info, init_obs_text: str) -> str:
        nl_inst = nl_inst_info.get("nl_inst", "")
        message = nl_inst_info.get("message", "")

        system_prompt = get_webshop_reactree_prompt(nl_inst).strip()
        policy = (
            "You must choose exactly one next step in this format:\n"
            "1) Think: <short reasoning>\n"
            "2) Act: <one action from available actions>\n"
            "3) Expand:\n"
            "   - control flow: sequence|fallback|parallel\n"
            "   - subgoals: <comma-separated subgoals>\n"
            "Do not output anything else.\n"
        )

        if message:
            context = f"Parent context:\n{message}\n"
        else:
            context = ""

        return (
            f"{system_prompt}\n\n"
            f"{policy}\n"
            f"{context}"
            f"Current observation:\n{init_obs_text}\n"
        )

    def add_obs(self, obs_text: str):
        with guidance.user():
            self.llm += f"Observation:\n{obs_text}\n"

    def plan_next_step(self, skill_set):
        try:
            with guidance.assistant():
                self.llm += guidance.select(
                    ["Act: ", "Think: ", "Expand:\n"],
                    name="choice",
                )

                if self.llm["choice"] == "Think: ":
                    self.llm += (
                        guidance.gen(
                            stop="\n",
                            name="reasoning",
                            max_tokens=200,
                            temperature=0,
                        )
                        + "\nOK.\n"
                    )
                    next_step_info = {
                        "next_step_class": "Think",
                        "next_step": self.llm["reasoning"].strip(),
                    }

                elif self.llm["choice"] == "Act: ":
                    self.llm += guidance.select(skill_set, name="nl_skill") + "\n"
                    next_step_info = {
                        "next_step_class": "Act",
                        "next_step": self.llm["nl_skill"].strip(),
                    }

                elif self.llm["choice"] == "Expand:\n":
                    self.llm += (
                        "- control flow: "
                        + guidance.select(
                            ["sequence", "fallback", "parallel"],
                            name="control_flow",
                        )
                        + "\n- subgoals: "
                        + guidance.gen(
                            stop="\n",
                            name="conditions",
                            max_tokens=200,
                            temperature=0,
                        )
                        + "\nOK.\n"
                    )
                    next_step_info = {
                        "next_step_class": "Expand",
                        "next_step": {
                            "control_flow": self.llm["control_flow"].strip(),
                            "conditions": self.llm["conditions"].strip(),
                        },
                    }

                else:
                    next_step_info = {
                        "next_step_class": "Error",
                        "next_step": f"Unexpected choice: {self.llm.get('choice')}",
                    }

            return next_step_info

        except Exception as e:
            with guidance.user():
                self.llm += (
                    "You should only output Think, Act, or Expand format.\n"
                )
            return {"next_step_class": "Error", "next_step": str(e)}