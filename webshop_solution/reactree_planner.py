import random
from typing import Dict, Any
import gym
from webshop_solution.config import default_config
# from webshop_solution.envs.webshop_env import WebShopEnv
from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
from webshop_solution.llm.llm_agent import WebShopLlmAgent
from webshop_solution.tree.agent_node import AgentNode
from web_agent_site.utils import DEBUG_PROD_SIZE

class ReactreeWebshopPlanner:
    def __init__(self, num_products: int = 100, seed: int = 0, cfg=None):
        self.cfg = cfg or default_config()
        self.cfg.planner.num_products = num_products
        self.cfg.planner.random_seed = seed
        random.seed(seed)
        self.env = gym.make('WebAgentTextEnv-v0', observation_mode='text', num_products=DEBUG_PROD_SIZE)


    def plan_and_run(self, instruction_text: str = None, top_k: int = 1, verbose: bool = True) -> Dict[str, Any]:
        # top_k kept for compatibility with previous interface.
        _ = top_k

        obs = self.env.reset(instruction_text=instruction_text)
        init_obs_text = obs[0] if isinstance(obs, tuple) else obs

        if instruction_text is None:
            if "[SEP] Instruction: [SEP]" in init_obs_text:
                parts = init_obs_text.split("[SEP] Instruction: [SEP]")
                if len(parts) > 1:
                    instruction_text = parts[1].split("[SEP]")[0].strip()
            if not instruction_text:
                instruction_text = self.env.instruction_text

        if verbose:
            print("[planner] Instruction:", instruction_text)

        llm_agent = WebShopLlmAgent(self.cfg)
        root = AgentNode(
            cfg=self.cfg,
            content={"nl_inst": instruction_text, "task_type": "webshop"},
            depth=1,
            llm_agent=llm_agent,
            env=self.env,
        )

        trajectory = []
        result = root.run(
            cur_step_id=1,
            cur_decision_id=1,
            log=None,
            init_obs_text=init_obs_text,
            trajectory=trajectory,
        )

        return {
            "instruction": instruction_text,
            "trajectory": result.get("trajectory", trajectory),
            "success": result.get("success", False),
            "terminate": result.get("terminate"),
            "step_id": result.get("step_id", 0),
            "decision_id": result.get("decision_id", 0),
        }