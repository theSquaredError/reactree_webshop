from webshop_solution.memory.working_memory import WorkingMemory


class TreeExecutor:
    def __init__(self, env):
        self.env = env
        self.memory = WorkingMemory()

    def run(self, root, instruction_text=None):
        observation = self.env.reset(instruction_text=instruction_text)
        self.memory.observation = observation[0] if isinstance(observation, tuple) else observation
        return root.run(cur_step_id=1, cur_decision_id=1, trajectory=[])