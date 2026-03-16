from reactree.src.reactree import AgentNode, ControlFlowNode

class WebShopAgentNode(AgentNode):
    def run(self, cur_step_id, cur_decision_id, log, init_obs_text=None, trajectory=None):
        if trajectory is None:
            trajectory = []
        message = self.make_message()
        nl_inst_info = self.content
        nl_inst = nl_inst_info["nl_inst"]
        nl_inst_info["message"] = message

        if init_obs_text is None:
            init_obs_text = self.env.observation
        self.llm_agent.reset(nl_inst_info, init_obs_text)

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
                trajectory.append({
                    "step_id": cur_step_id,
                    "decision_id": cur_decision_id,
                    "subgoal": self.content.get("nl_inst", ""),
                    "action": "error",
                    "observation": str(error_message),
                    "success": False,
                    "terminate": "plan_next_step_error"
                })
                return {"success": False, "terminate": "plan_next_step_error", "step_id": cur_step_id, "decision_id": cur_decision_id, "trajectory": trajectory}

            if next_step_class == "Think":
                trajectory.append({
                    "step_id": cur_step_id,
                    "decision_id": cur_decision_id,
                    "subgoal": self.content.get("nl_inst", ""),
                    "action": "Think",
                    "observation": next_step,
                    "success": None,
                    "terminate": None
                })
                cur_decision_id += 1
            elif next_step_class == "Act":
                obs_text, done, reward = self._step_webshop(next_step)
                trajectory.append({
                    "step_id": cur_step_id,
                    "decision_id": cur_decision_id,
                    "subgoal": self.content.get("nl_inst", ""),
                    "action": next_step,
                    "observation": obs_text,
                    "success": reward > 0,
                    "terminate": "env_done" if done else None
                })
                self.llm_agent.add_obs(obs_text)
                cur_step_id += 1
                cur_decision_id += 1
                if done:
                    return {
                        "success": reward > 0,
                        "terminate": "env_done",
                        "step_id": cur_step_id,
                        "decision_id": cur_decision_id,
                        "trajectory": trajectory
                    }
            elif next_step_class == "Expand":
                trajectory.append({
                    "step_id": cur_step_id,
                    "decision_id": cur_decision_id,
                    "subgoal": self.content.get("nl_inst", ""),
                    "action": "Expand",
                    "observation": next_step,
                    "success": None,
                    "terminate": None
                })
                cur_decision_id += 1
                control_flow = next_step["control_flow"]
                subgoals = [s.strip() for s in next_step["conditions"].split(",") if s.strip()]
                control_flow_node = ControlFlowNode(self.cfg, control_flow, self.depth + 1)
                self.add_child(control_flow_node)
                for subgoal in subgoals:
                    subgoal_info = {"nl_inst": subgoal, "task_type": self.content.get("task_type", "webshop")}
                    agent_node = WebShopAgentNode(self.cfg, subgoal_info, self.depth + 2, self.llm_agent, self.env)
                    control_flow_node.add_child(agent_node)
                return self.children[0].run(cur_step_id, cur_decision_id, log, trajectory=trajectory)
            elif next_step_class == "Error":
                trajectory.append({
                    "step_id": cur_step_id,
                    "decision_id": cur_decision_id,
                    "subgoal": self.content.get("nl_inst", ""),
                    "action": "Error",
                    "observation": next_step,
                    "success": False,
                    "terminate": "error"
                })
                cur_step_id += 1
            else:
                raise NotImplementedError()

    def _get_possible_skill_set(self):
        available_actions = self.env.get_available_actions()
        clickables = available_actions.get("clickables", [])
        skill_set = [f"click[{clickable}]" for clickable in clickables]
        if available_actions.get("has_search_bar", False):
            skill_set.append(f"search[{self.content.get('nl_inst', '')}]")
        skill_set.append("done")
        skill_set.append("failure")
        return list(dict.fromkeys(skill_set))

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

    def make_message(self):
        return None