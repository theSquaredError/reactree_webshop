from __future__ import annotations

from datetime import datetime

from webshop_solution.tree.control_nodes import ControlFlowNode
from webshop_solution.tree.node import Node


class AgentNode(Node):
    def __init__(self, cfg, content, depth, llm_agent, env):
        super().__init__(depth=depth)
        self.cfg = cfg
        self.content = content
        self.llm_agent = llm_agent
        self.env = env
        # Default verbose on unless cfg.planner.verbose is explicitly False.
        self.verbose = getattr(getattr(cfg, "planner", object()), "verbose", True)

    def _log(self, msg: str):
        if not self.verbose:
            return
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}][agent_node][id={self.node_id}][depth={self.depth}] {msg}", flush=True)

    def run(self, cur_step_id, cur_decision_id, log=None, init_obs_text=None, trajectory=None):
        if trajectory is None:
            trajectory = []

        message = self.make_message()
        nl_inst = self.content.get("nl_inst", "")

        if init_obs_text is None:
            init_obs_text = self.env.observation

        self._log(
            f"run() start step={cur_step_id} decision={cur_decision_id} "
            f"subgoal={nl_inst!r}"
        )
        self.llm_agent.reset({"nl_inst": nl_inst, "message": message}, init_obs_text)
        self._log(f"llm reset complete; init_obs_chars={len(str(init_obs_text))}")

        while True:
            self._log(f"loop step={cur_step_id} decision={cur_decision_id}")
            if cur_step_id > self.cfg.planner.max_steps:
                self._log("terminate=max_step")
                return self._terminate(False, "max_step", cur_step_id, cur_decision_id, trajectory)

            if cur_decision_id > self.cfg.planner.max_decisions:
                self._log("terminate=max_decision")
                return self._terminate(False, "max_decision", cur_step_id, cur_decision_id, trajectory)

            skill_set = self._get_possible_skill_set()
            self._log(f"available_skills={len(skill_set)}")
            next_step_info = self.llm_agent.plan_next_step(skill_set)
            next_step_class = next_step_info.get("next_step_class")
            next_step = next_step_info.get("next_step")
            self._log(f"next_step_class={next_step_class} next_step={next_step!r}")

            if next_step_class == "Think":
                trajectory.append(
                    self._mk_entry(
                        cur_step_id,
                        cur_decision_id,
                        "Think",
                        self.env.observation,
                        None,
                        None,
                        llm_action="Think",
                        llm_reasoning=str(next_step),
                    )
                )
                self.llm_agent.add_obs("OK.")
                self._log("handled Think; appended synthetic obs='OK.'")
                cur_decision_id += 1
                continue

            if next_step_class == "Act":
                action = str(next_step)
                self._log(f"Act received action={action!r}")

                if action == "done":
                    trajectory.append(
                        self._mk_entry(
                            cur_step_id,
                            cur_decision_id,
                            action,
                            self.env.observation,
                            True,
                            "done",
                            llm_action=action,
                            llm_reasoning=None,
                        )
                    )
                    self._log("terminate=done")
                    return self._terminate(True, "done", cur_step_id, cur_decision_id + 1, trajectory)

                if action == "failure":
                    trajectory.append(
                        self._mk_entry(
                            cur_step_id,
                            cur_decision_id,
                            action,
                            self.env.observation,
                            False,
                            "failure",
                            llm_action=action,
                            llm_reasoning=None,
                        )
                    )
                    self._log("terminate=failure")
                    return self._terminate(False, "failure", cur_step_id, cur_decision_id + 1, trajectory)

                self._log("executing env.step(...)")
                obs_text, done, reward = self._step_webshop(action)
                self._log(
                    f"env result done={done} reward={reward} "
                    f"obs_chars={len(str(obs_text))}"
                )
                trajectory.append(
                    self._mk_entry(
                        cur_step_id,
                        cur_decision_id,
                        action,  # executed env action
                        self.env.observation,  # canonical env state after env.step
                        reward > 0,
                        "env_done" if done else None,
                        llm_action=action,
                        llm_reasoning=None,
                    )
                )
                self.llm_agent.add_obs(obs_text)
                cur_step_id += 1
                cur_decision_id += 1
                if done:
                    self._log("terminate=env_done")
                    return self._terminate(reward > 0, "env_done", cur_step_id, cur_decision_id, trajectory)
                continue

            if next_step_class == "Expand":
                control_flow = next_step.get("control_flow", "sequence")
                subgoals = [s.strip() for s in next_step.get("conditions", "").split(",") if s.strip()]
                self._log(f"Expand control_flow={control_flow} subgoals={subgoals}")
                trajectory.append(
                    self._mk_entry(
                        cur_step_id,
                        cur_decision_id,
                        "Expand",
                        self.env.observation,
                        None,
                        None,
                        llm_action=f"Expand(control_flow={control_flow}, subgoals={subgoals})",
                        llm_reasoning=None,
                    )
                )

                if not subgoals:
                    self._log("terminate=empty_expand")
                    return self._terminate(False, "empty_expand", cur_step_id, cur_decision_id + 1, trajectory)

                control = ControlFlowNode(self.cfg, control_flow, self.depth + 1)
                self.add_child(control)
                for subgoal in subgoals:
                    child = AgentNode(
                        cfg=self.cfg,
                        content={"nl_inst": subgoal, "task_type": self.content.get("task_type", "webshop")},
                        depth=self.depth + 2,
                        llm_agent=self.llm_agent,
                        env=self.env,
                    )
                    control.add_child(child)
                self._log("delegating to ControlFlowNode.run(...)")
                return control.run(cur_step_id, cur_decision_id + 1, trajectory=trajectory, log=log)

            trajectory.append(
                self._mk_entry(
                    cur_step_id,
                    cur_decision_id,
                    "Error",
                    self.env.observation,
                    False,
                    "plan_next_step_error",
                    llm_action=f"Error({next_step_class})",
                    llm_reasoning=str(next_step),
                )
            )
            self._log(f"terminate=plan_next_step_error raw={next_step!r}")
            return self._terminate(False, "plan_next_step_error", cur_step_id, cur_decision_id, trajectory)

    def make_message(self):
        if self.parent is None or not isinstance(self.parent, ControlFlowNode):
            return ""

        super_node = self.parent.parent
        if super_node is None or not hasattr(super_node, "content"):
            return ""

        supergoal = super_node.content.get("nl_inst", "")
        sibling_goals = [child.content.get("nl_inst", "") for child in self.parent.children if hasattr(child, "content")]
        flow = self.parent.control_flow
        if flow == "sequence":
            flow_phrase = "in sequence"
        elif flow == "fallback":
            flow_phrase = "using fallback"
        else:
            flow_phrase = "in parallel"
        siblings_text = ", ".join([g for g in sibling_goals if g])
        return (
            f"Your primary goal is: {supergoal}. "
            f"Sibling goals should be executed {flow_phrase}: {siblings_text}."
        )

    def _get_possible_skill_set(self):
        available = self.env.get_available_actions()
        clickables = available.get("clickables", [])
        skill_set = [f"click[{name}]" for name in clickables]

        # if available.get("has_search_bar", False):
        #     query = self.content.get("nl_inst", "").strip()
        #     if query:
        #         skill_set.append(f"search[{query}]")

        if available.get("has_search_bar", False):
            skill_set.append("search[query]")

        # Keep explicit markers for subgoal-level control.
        skill_set.append("done")
        skill_set.append("failure")
        return list(dict.fromkeys(skill_set))

    def _step_webshop(self, action_text):
        if not (
            action_text.startswith("click[") and action_text.endswith("]")
        ) and not (
            action_text.startswith("search[") and action_text.endswith("]")
        ):
            return f"Invalid WebShop action format: {action_text}", False, 0.0

        try:
            obs, reward, done, _ = self.env.step(action_text)
        except Exception as exc:
            return f"Action execution failed: {exc}", False, 0.0

        obs_text = f"{obs}\n[reward={reward}]"
        return obs_text, done, reward

    def _mk_entry(self, step_id, decision_id, action, observation, success, terminate, llm_action=None, llm_reasoning=None):
        parent_control_flow = None
        if isinstance(self.parent, ControlFlowNode):
            parent_control_flow = self.parent.control_flow

        return {
            "llm_action": llm_action if llm_action is not None else action,
            "llm_reasoning": llm_reasoning,
            "agent_node_id": self.node_id,
            "agent_depth": self.depth,
            "parent_control_flow": parent_control_flow,
            "step_id": step_id,
            "decision_id": decision_id,
            "subgoal": self.content.get("nl_inst", ""),
            "action": action,
            "observation": observation,
            "success": success,
            "terminate": terminate,
        }

    def _terminate(self, success, terminate, step_id, decision_id, trajectory):
        return {
            "success": success,
            "terminate": terminate,
            "step_id": step_id,
            "decision_id": decision_id,
            "trajectory": trajectory,
        }