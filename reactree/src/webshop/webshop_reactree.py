from reactree.src.reactree import Reactree
from .webshop_agent_node import WebShopAgentNode
import os

class WebShopReactree(Reactree):
    """ReAcTree runner for WebShop text environment."""

    def run(self, task_d=None, log=None):
        nl_inst, init_obs_text = self._reset_webshop(task_d)
        nl_inst_info = {"nl_inst": nl_inst, "task_type": "webshop"}
        self.root_node = WebShopAgentNode(self.cfg, nl_inst_info, 1, self.llm_agent, self.env)
        return self.root_node.run(1, 1, log, init_obs_text=init_obs_text)

    def collect(self, task_d=None):
        nl_inst, init_obs_text = self._reset_webshop(task_d)
        collect_dir = self.cfg.dataset.collect_dir
        os.makedirs(collect_dir, exist_ok=True)
        task_id = "webshop"
        if isinstance(task_d, dict):
            task_id = str(task_d.get("task_id", task_id))
        collect_file_base_name = f"{task_id}_webshop"
        self.root_node = WebShopAgentNode(self.cfg, {"nl_inst": nl_inst, "task_type": "webshop"}, 1, self.llm_agent, self.env)
        self.root_node.collect(1, 1, collect_dir, collect_file_base_name, init_obs_text=init_obs_text)

    def _reset_webshop(self, task_d):
        reset_kwargs = {}
        if isinstance(task_d, dict):
            if task_d.get("session") is not None:
                reset_kwargs["session"] = task_d["session"]
            if task_d.get("instruction_text") is not None:
                reset_kwargs["instruction_text"] = task_d["instruction_text"]
        if reset_kwargs:
            reset_result = self.env.reset(**reset_kwargs)
        else:
            reset_result = self.env.reset()
        init_obs_text = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        nl_inst = getattr(self.env, "instruction_text", None)
        if not nl_inst:
            if isinstance(task_d, dict):
                nl_inst = task_d.get("instruction_text", "")
            else:
                nl_inst = ""
        return nl_inst, init_obs_text