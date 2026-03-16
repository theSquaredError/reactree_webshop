import datetime
import os
import re


class Reactree():
    def __init__(self, cfg, llm_agent, env):
        self.cfg = cfg
        self.llm_agent = llm_agent
        self.env = env
        self.max_steps = cfg.llm_agent.max_steps
        self.max_decisions = cfg.llm_agent.max_decisions
        self.cur_step_id = 1
        self.cur_decision_id = 1
    def run(self, task_d):
        raise NotImplementedError()
    def collect(self):
        raise NotImplementedError()

class TreeNode:
    def __init__(self, cfg, content, depth):
        self.cfg = cfg
        self.content = content
        self.depth = depth
        self.children = []
        self.parent = None

    def add_child(self, child_node):
        child_node.parent = self
        self.children.append(child_node)
    
    def run(self, cur_step_id, cur_decision_id, log):
        raise NotImplementedError()

class ControlFlowNode(TreeNode):
    def run(self, cur_step_id, cur_decision_id, log, traj_d=None):
        if self.depth > self.cfg.llm_agent.max_depth:
            terminate_info = {'success': False, 'terminate': 'max_depth', 'step_id': cur_step_id, 'decision_id': cur_decision_id}
            log.info('Max depth')
            return terminate_info

        if self.content == 'sequence':
            for child in self.children:
                if traj_d is None: # for wah --> src.wah.wah_reactree.AgentNode.run()
                    terminate_info = child.run(cur_step_id, cur_decision_id, log)
                else: # for alfred --> src.alfred.alfred_reactree.AlfredAgentNode.run()
                    terminate_info = child.run(traj_d, cur_step_id, cur_decision_id, log)
                cur_step_id, cur_decision_id = terminate_info['step_id'], terminate_info['decision_id']
                if not terminate_info['success']:
                    terminate_info = {'success': False, 'step_id': cur_step_id, 'decision_id': cur_decision_id}
                    return terminate_info
            terminate_info = {'success': True, 'step_id': cur_step_id, 'decision_id': cur_decision_id}
            return terminate_info
        elif self.content == 'fallback':
            for child in self.children:
                if traj_d is None: # for wah --> src.wah.wah_reactree.AgentNode.run()
                    terminate_info = child.run(cur_step_id, cur_decision_id, log)
                else:
                    terminate_info = child.run(traj_d, cur_step_id, cur_decision_id, log)
                cur_step_id, cur_decision_id = terminate_info['step_id'], terminate_info['decision_id']
                if terminate_info['success']:
                    terminate_info = {'success': True, 'step_id': cur_step_id, 'decision_id': cur_decision_id}
                    return terminate_info
            terminate_info = {'success': False, 'step_id': cur_step_id, 'decision_id': cur_decision_id}
            return terminate_info
        elif self.content == 'parallel':
            is_success = True
            for child in self.children:
                if traj_d is None: # for wah --> src.wah.wah_reactree.AgentNode.run()
                    terminate_info = child.run(cur_step_id, cur_decision_id, log)
                else:
                    terminate_info = child.run(traj_d, cur_step_id, cur_decision_id, log)
                cur_step_id, cur_decision_id = terminate_info['step_id'], terminate_info['decision_id']
                if not terminate_info['success']:
                    is_success = False
            terminate_info = {'success': is_success, 'step_id': cur_step_id, 'decision_id': cur_decision_id}
            return terminate_info
        else:
            raise NotImplementedError()
    
    def collect(self, cur_step_id, cur_decision_id, collect_dir, collect_file_base_name, traj_d=None):
        if self.content == 'sequence':
            for child in self.children:
                if traj_d is None: # for wah --> src.wah.wah_reactree.AgentNode.collect()
                    terminate_info = child.collect(cur_step_id, cur_decision_id, collect_dir, collect_file_base_name)
                else: # for alfred --> src.alfred.alfred_reactree.AlfredAgentNode.collect()
                    terminate_info = child.collect(traj_d, cur_step_id, cur_decision_id, collect_dir, collect_file_base_name)
                cur_step_id, cur_decision_id = terminate_info['step_id'], terminate_info['decision_id']
                if not terminate_info['success']:
                    terminate_info = {'success': False, 'step_id': cur_step_id, 'decision_id': cur_decision_id}
                    return terminate_info
            terminate_info = {'success': True, 'step_id': cur_step_id, 'decision_id': cur_decision_id}
            return terminate_info
        elif self.content == 'fallback':
            for child in self.children:
                if traj_d is None: # for wah --> src.wah.wah_reactree.AgentNode.collect()
                    terminate_info = child.collect(cur_step_id, cur_decision_id, collect_dir, collect_file_base_name)
                else: # for alfred --> src.alfred.alfred_reactree.AlfredAgentNode.collect()
                    terminate_info = child.collect(traj_d, cur_step_id, cur_decision_id, collect_dir, collect_file_base_name)
                cur_step_id, cur_decision_id = terminate_info['step_id'], terminate_info['decision_id']
                if terminate_info['success']:
                    terminate_info = {'success': True, 'step_id': cur_step_id, 'decision_id': cur_decision_id}
                    return terminate_info
            terminate_info = {'success': False, 'step_id': cur_step_id, 'decision_id': cur_decision_id}
            return terminate_info
        elif self.content == 'parallel':
            is_success = True
            for child in self.children:
                if traj_d is None: # for wah --> src.wah.wah_reactree.AgentNode.collect()
                    terminate_info = child.collect(cur_step_id, cur_decision_id, collect_dir, collect_file_base_name)
                else: # for alfred --> src.alfred.alfred_reactree.AlfredAgentNode.collect()
                    terminate_info = child.collect(traj_d, cur_step_id, cur_decision_id, collect_dir, collect_file_base_name)
                cur_step_id, cur_decision_id = terminate_info['step_id'], terminate_info['decision_id']
                if not terminate_info['success']:
                    is_success = False
            terminate_info = {'success': is_success, 'step_id': cur_step_id, 'decision_id': cur_decision_id}
            return terminate_info
        else:
            raise NotImplementedError()
        
    def collect_llm(self, cur_step_id, cur_decision_id, collect_dir, collect_file_base_name, trial_id, traj_d=None):
        if self.content == 'sequence':
            for child in self.children:
                if traj_d is None: # for wah --> src.wah.wah_reactree.AgentNode.collect_llm()
                    terminate_info = child.collect_llm(cur_step_id, cur_decision_id, collect_dir, collect_file_base_name, trial_id)
                else: # for alfred --> src.alfred.alfred_reactree.AlfredAgentNode.collect_llm()
                    terminate_info = child.collect_llm(traj_d, cur_step_id, cur_decision_id, collect_dir, collect_file_base_name)
                cur_step_id, cur_decision_id = terminate_info['step_id'], terminate_info['decision_id']
                if not terminate_info['success']:
                    terminate_info = {'success': False, 'step_id': cur_step_id, 'decision_id': cur_decision_id}
                    return terminate_info
            terminate_info = {'success': True, 'step_id': cur_step_id, 'decision_id': cur_decision_id}
            return terminate_info
        elif self.content == 'fallback':
            for child in self.children:
                if traj_d is None: # for wah --> src.wah.wah_reactree.AgentNode.collect_llm()
                    terminate_info = child.collect_llm(cur_step_id, cur_decision_id, collect_dir, collect_file_base_name, trial_id)
                else: # for alfred --> src.alfred.alfred_reactree.AlfredAgentNode.collect_llm()
                    terminate_info = child.collect_llm(traj_d, cur_step_id, cur_decision_id, collect_dir, collect_file_base_name)
                cur_step_id, cur_decision_id = terminate_info['step_id'], terminate_info['decision_id']
                if terminate_info['success']:
                    terminate_info = {'success': True, 'step_id': cur_step_id, 'decision_id': cur_decision_id}
                    return terminate_info
            terminate_info = {'success': False, 'step_id': cur_step_id, 'decision_id': cur_decision_id}
            return terminate_info
        elif self.content == 'parallel':
            is_success = True
            for child in self.children:
                if traj_d is None: # for wah --> src.wah.wah_reactree.AgentNode.collect_llm()
                    terminate_info = child.collect_llm(cur_step_id, cur_decision_id, collect_dir, collect_file_base_name, trial_id)
                else: # for alfred --> src.alfred.alfred_reactree.AlfredAgentNode.collect_llm()
                    terminate_info = child.collect_llm(traj_d, cur_step_id, cur_decision_id, collect_dir, collect_file_base_name)
                cur_step_id, cur_decision_id = terminate_info['step_id'], terminate_info['decision_id']
                if not terminate_info['success']:
                    is_success = False
            terminate_info = {'success': is_success, 'step_id': cur_step_id, 'decision_id': cur_decision_id}
            return terminate_info
        else:
            raise NotImplementedError()

    def make_message(self):
        return None

class AgentNode(TreeNode):
    def __init__(self, cfg, content, depth, llm_agent, env):
        super().__init__(cfg, content, depth)
        self.llm_agent = llm_agent
        self.env = env
    def run(self, cur_step_id, cur_decision_id, log):
        raise NotImplementedError()
    def collect(self, cur_step_id, cur_decision_id, collect_dir, collect_file_base_name):
        raise NotImplementedError()


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
        self.root_node = WebShopAgentNode(self.cfg, {"nl_inst": nl_inst, "task_type": "webshop"}, 1, None, self.env)
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


class WebShopAgentNode(AgentNode):
    def run(self, cur_step_id, cur_decision_id, log, init_obs_text=None):
        message = self.make_message()
        nl_inst_info = self.content
        nl_inst = nl_inst_info["nl_inst"]
        nl_inst_info["message"] = message

        if init_obs_text is None:
            init_obs_text = self.env.observation
        self.llm_agent.reset(nl_inst_info, init_obs_text)

        if message:
            self._log(log, message)
        self._log(log, f"Your task is to: {nl_inst}")
        self._log(log, init_obs_text)

        while True:
            if cur_step_id > self.cfg.llm_agent.max_steps:
                self._log(log, "Max steps")
                return {"success": False, "terminate": "max_step", "step_id": cur_step_id, "decision_id": cur_decision_id}
            if cur_decision_id > self.cfg.llm_agent.max_decisions:
                self._log(log, "Max decisions")
                return {"success": False, "terminate": "max_decision", "step_id": cur_step_id, "decision_id": cur_decision_id}

            skill_set = self._get_possible_skill_set()
            try:
                next_step_info = self.llm_agent.plan_next_step(skill_set)
                next_step_class = next_step_info["next_step_class"]
                next_step = next_step_info["next_step"]
                self._log(log, f"{next_step_class}: {next_step}")
            except Exception as error_message:
                self._log(log, f"Plan Next Step Error: {error_message}")
                return {"success": False, "terminate": "plan_next_step_error", "step_id": cur_step_id, "decision_id": cur_decision_id}

            if next_step_class == "Think":
                cur_decision_id += 1
            elif next_step_class == "Act":
                if next_step == "done":
                    return {"success": True, "terminate": "done", "step_id": cur_step_id, "decision_id": cur_decision_id}
                if next_step == "failure":
                    return {"success": False, "terminate": "failure", "step_id": cur_step_id, "decision_id": cur_decision_id}

                obs_text, done, reward = self._step_webshop(next_step)
                self._log(log, obs_text)
                self.llm_agent.add_obs(obs_text)
                cur_step_id += 1
                cur_decision_id += 1
                if done:
                    return {
                        "success": reward > 0,
                        "terminate": "env_done",
                        "step_id": cur_step_id,
                        "decision_id": cur_decision_id,
                    }
            elif next_step_class == "Expand":
                cur_decision_id += 1
                control_flow = next_step["control_flow"]
                subgoals = [s.strip() for s in next_step["conditions"].split(",") if s.strip()]
                if not subgoals:
                    return {"success": False, "terminate": "empty_expand", "step_id": cur_step_id, "decision_id": cur_decision_id}

                control_flow_node = ControlFlowNode(self.cfg, control_flow, self.depth + 1)
                self.add_child(control_flow_node)
                for subgoal in subgoals:
                    subgoal_info = {"nl_inst": subgoal, "task_type": nl_inst_info.get("task_type", "webshop")}
                    agent_node = WebShopAgentNode(self.cfg, subgoal_info, self.depth + 2, self.llm_agent, self.env)
                    control_flow_node.add_child(agent_node)
                return self.children[0].run(cur_step_id, cur_decision_id, log)
            elif next_step_class == "Error":
                cur_step_id += 1
                cur_decision_id += 1
            else:
                raise NotImplementedError()

    def collect(self, cur_step_id, cur_decision_id, collect_dir, collect_file_base_name, init_obs_text=None):
        message = self.make_message()
        nl_inst_info = self.content
        nl_inst = nl_inst_info["nl_inst"]
        nl_inst_info["message"] = message
        if init_obs_text is None:
            init_obs_text = self.env.observation

        collect_file_name = f"{collect_file_base_name}_dec{str(cur_decision_id).zfill(3)}_depth{str(self.depth).zfill(2)}"
        collect_file_path = os.path.join(collect_dir, f"{collect_file_name}.txt")
        if os.path.exists(collect_file_path):
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            collect_file_path = os.path.join(collect_dir, f"{collect_file_name}_{timestamp}.txt")

        with open(collect_file_path, "w") as file:
            if message:
                file.write(f"{message}\n")
                print(f"\n{message}")
            print(f"Your task is to: {nl_inst}\n")
            print(init_obs_text)
            file.write(f"Your task is to: {nl_inst}\n{init_obs_text}\n")

        while True:
            decision = input('Type decision ("r" or "a" or "e"): ').strip().lower()
            if decision == "r":
                reasoning_text = input("Think: ")
                cur_decision_id += 1
                with open(collect_file_path, "a") as file:
                    file.write(f"Think: {reasoning_text}\nOK.\n")
            elif decision == "a":
                action_text = input("Act: ").strip()
                if action_text in ["done", "failure"]:
                    with open(collect_file_path, "a") as file:
                        file.write(f"Act: {action_text}\n")
                    return {
                        "success": action_text == "done",
                        "terminate": action_text,
                        "step_id": cur_step_id,
                        "decision_id": cur_decision_id,
                    }
                obs_text, done, reward = self._step_webshop(action_text)
                print(obs_text)
                with open(collect_file_path, "a") as file:
                    file.write(f"Act: {action_text}\n{obs_text}\n")
                cur_step_id += 1
                cur_decision_id += 1
                if done:
                    return {
                        "success": reward > 0,
                        "terminate": "env_done",
                        "step_id": cur_step_id,
                        "decision_id": cur_decision_id,
                    }
            elif decision == "e":
                cur_decision_id += 1
                control_flow = input('Control flow ("sequence" or "fallback" or "parallel"): ').strip().lower()
                subgoals_text = input('Subgoals (connect with ","): ')
                with open(collect_file_path, "a") as file:
                    file.write(f"Expand:\n- control flow: {control_flow}\n- subgoals: {subgoals_text}\n")

                subgoals = [s.strip() for s in subgoals_text.split(",") if s.strip()]
                control_flow_node = ControlFlowNode(self.cfg, control_flow, self.depth + 1)
                self.add_child(control_flow_node)
                for subgoal in subgoals:
                    subgoal_info = {"nl_inst": subgoal, "task_type": nl_inst_info.get("task_type", "webshop")}
                    agent_node = WebShopAgentNode(self.cfg, subgoal_info, self.depth + 2, None, self.env)
                    control_flow_node.add_child(agent_node)
                return self.children[0].collect(cur_step_id, cur_decision_id, collect_dir, collect_file_base_name)
            else:
                print("Error. Type again.")

    def make_message(self):
        if self.cfg.llm_agent.message_type == "goal_information":
            return self.make_message_goal_information()
        return None

    def make_message_goal_information(self):
        if self.parent is None:
            return ""

        control_flow_node = self.parent
        superagent_node = control_flow_node.parent
        sibling_nodes = control_flow_node.children

        supergoal = superagent_node.content["nl_inst"]
        control_flow = control_flow_node.content
        sibling_goals = [sibling_node.content["nl_inst"] for sibling_node in sibling_nodes]

        if control_flow == "sequence":
            control_flow_phrase = "in sequence"
        elif control_flow == "fallback":
            control_flow_phrase = "using a fallback strategy"
        elif control_flow == "parallel":
            control_flow_phrase = "in parallel"
        else:
            raise NotImplementedError()

        if len(sibling_goals) == 1:
            sibling_goals_phrase = sibling_goals[0]
        else:
            sibling_goals_phrase = ", ".join(sibling_goals[:-1]) + ", and " + sibling_goals[-1]
        return (
            f"Your primary goal is to: {supergoal}\n"
            f"To achieve this, you should perform your sibling tasks {control_flow_phrase}. "
            f"At this level, your sibling tasks are: {sibling_goals_phrase}."
        )

    def _get_possible_skill_set(self):
        available_actions = self.env.get_available_actions()
        clickables = available_actions.get("clickables", [])
        skill_set = [f"click[{clickable}]" for clickable in clickables]

        if available_actions.get("has_search_bar", False):
            for query in self._make_search_queries(self.content.get("nl_inst", "")):
                skill_set.append(f"search[{query}]")

        skill_set.append("done")
        skill_set.append("failure")
        return list(dict.fromkeys(skill_set))

    def _make_search_queries(self, instruction):
        normalized = instruction.strip().lower()
        if not normalized:
            return ["product"]

        cleaned = re.sub(r"[^a-z0-9 ]+", " ", normalized)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        stemmed = re.sub(r"^(find|find me|i need|i want|looking for|search for)\s+", "", cleaned).strip()
        short = " ".join(stemmed.split()[:7]).strip()

        candidates = [cleaned, stemmed, short]
        candidates = [c for c in candidates if c]
        if not candidates:
            return ["product"]
        return list(dict.fromkeys(candidates[:3]))

    def _step_webshop(self, action_text):
        try:
            obs, reward, done, _ = self.env.step(action_text)
        except Exception as exc:
            return f"Action failed: {exc}", False, 0.0
        obs_text = f"{obs}\n[reward={reward}]"
        return obs_text, done, reward

    def _log(self, log, text):
        if log is not None and hasattr(log, "info"):
            log.info(text)
        else:
            print(text)
