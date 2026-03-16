from webshop_solution.tree.node import Node


class ControlFlowNode(Node):
    def __init__(self, cfg, control_flow: str, depth: int):
        super().__init__(depth=depth)
        self.cfg = cfg
        self.control_flow = control_flow

    def run(self, cur_step_id, cur_decision_id, trajectory, log=None):
        if self.depth > self.cfg.planner.max_depth:
            return {
                "success": False,
                "terminate": "max_depth",
                "step_id": cur_step_id,
                "decision_id": cur_decision_id,
                "trajectory": trajectory,
            }

        if self.control_flow == "sequence":
            for child in self.children:
                result = child.run(cur_step_id, cur_decision_id, log=log, trajectory=trajectory)
                cur_step_id = result["step_id"]
                cur_decision_id = result["decision_id"]
                if result.get("terminate") == "env_done":
                    return result
                if not result.get("success", False):
                    return result
            return {
                "success": True,
                "terminate": "sequence_done",
                "step_id": cur_step_id,
                "decision_id": cur_decision_id,
                "trajectory": trajectory,
            }

        if self.control_flow == "fallback":
            last_result = None
            for child in self.children:
                result = child.run(cur_step_id, cur_decision_id, log=log, trajectory=trajectory)
                cur_step_id = result["step_id"]
                cur_decision_id = result["decision_id"]
                last_result = result
                if result.get("terminate") == "env_done":
                    return result
                if result.get("success", False):
                    return result
            return last_result or {
                "success": False,
                "terminate": "fallback_failed",
                "step_id": cur_step_id,
                "decision_id": cur_decision_id,
                "trajectory": trajectory,
            }

        if self.control_flow == "parallel":
            any_success = False
            last_result = None
            for child in self.children:
                result = child.run(cur_step_id, cur_decision_id, log=log, trajectory=trajectory)
                cur_step_id = result["step_id"]
                cur_decision_id = result["decision_id"]
                last_result = result
                if result.get("terminate") == "env_done":
                    return result
                any_success = any_success or result.get("success", False)
            if last_result is None:
                return {
                    "success": False,
                    "terminate": "parallel_empty",
                    "step_id": cur_step_id,
                    "decision_id": cur_decision_id,
                    "trajectory": trajectory,
                }
            last_result["success"] = any_success
            last_result["terminate"] = "parallel_done"
            return last_result

        return {
            "success": False,
            "terminate": f"invalid_control_flow:{self.control_flow}",
            "step_id": cur_step_id,
            "decision_id": cur_decision_id,
            "trajectory": trajectory,
        }
    