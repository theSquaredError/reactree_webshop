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
                control_flow = next_step.get("control_flow", "sequence")
                subgoals = [s.strip() for s in next_step.get("conditions", "").split(",") if s.strip()]
                self._log(f"Expand control_flow={control_flow} subgoals={subgoals}")
                trajectory.append(
                    self._mk_entry(
                        cur_step_id, cur_decision_id, "Expand",
                        self.env.observation, None, None,
                        llm_action=f"Expand(control_flow={control_flow}, subgoals={subgoals})",
                    )
                )

                if not subgoals:
                    self._log("terminate=empty_expand")
                    return self._terminate(False, "empty_expand", cur_step_id, cur_decision_id + 1, trajectory)

                # ── MCTS path: parallel expand with ≥2 subgoals ──────────────
                mcts_cfg = getattr(self.cfg, "mcts", None)
                session_id = self.content.get("session_id")
                use_mcts = (
                    control_flow == "parallel"
                    and len(subgoals) >= 2
                    and mcts_cfg is not None
                    and getattr(mcts_cfg, "enabled", True)
                    and session_id is not None      # need integer session for replay
                )

                if use_mcts:
                    from webshop_solution.mcts.webshop_state import WebShopState
                    from webshop_solution.mcts.search_mcts import SearchQueryMCTS

                    # Snapshot env state right now (before any search)
                    base_state = WebShopState.capture(
                        env=self.env,
                        session_id=session_id,
                        instruction_text=self.content.get("nl_inst", ""),
                        max_steps=self.cfg.planner.max_steps,
                    )

                    # Each subgoal treated as a search query candidate
                    queries = [_to_search_query(s) for s in subgoals]
                    self._log(f"MCTS over {len(queries)} search candidates: {queries}")

                    mcts = SearchQueryMCTS(
                        env=self.env,
                        budget=mcts_cfg.budget,
                        max_rollout_steps=mcts_cfg.max_rollout_steps,
                        c=mcts_cfg.c,
                    )
                    best_action, mcts_root = mcts.run(base_state, queries)
                    ranked, preference_pairs = SearchQueryMCTS.build_preference_pairs(mcts_root)
                    self._log(f"MCTS best={best_action!r} ranked={ranked}")

                    # Annotate the last trajectory entry with MCTS preference data
                    trajectory[-1]["mcts_ranked"] = ranked
                    trajectory[-1]["mcts_preference_pairs"] = preference_pairs

                    # Execute the winning search in the real env and continue the loop
                    obs_text, done, reward = self._step_webshop(best_action)
                    trajectory.append(
                        self._mk_entry(
                            cur_step_id, cur_decision_id + 1,
                            best_action, self.env.observation,
                            reward > 0, "env_done" if done else None,
                            llm_action=best_action,
                            llm_reasoning=f"MCTS-selected from {len(queries)} candidates",
                        )
                    )
                    self.llm_agent.add_obs(obs_text)
                    cur_step_id += 1
                    cur_decision_id += 2
                    if done:
                        self._log("terminate=env_done (after mcts)")
                        return self._terminate(reward > 0, "env_done", cur_step_id, cur_decision_id, trajectory)
                    continue   # back to top of AgentNode loop; LLM now sees search results
                # ── end MCTS path ─────────────────────────────────────────────

                # Original Expand path (sequence / fallback / non-MCTS parallel)
                control = ControlFlowNode(self.cfg, control_flow, self.depth + 1)
                self.add_child(control)
                for subgoal in subgoals:
                    child = AgentNode(
                        cfg=self.cfg,
                        content={
                            "nl_inst": subgoal,
                            "task_type": self.content.get("task_type", "webshop"),
                            "session_id": self.content.get("session_id"),   # ← propagate
                        },
                        depth=self.depth + 2,
                        llm_agent=self.llm_agent,
                        env=self.env,
                    )
                    control.add_child(child)
                self._log("delegating to ControlFlowNode.run(...)")
                return control.run(cur_step_id, cur_decision_id + 1, trajectory=trajectory, log=log)
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