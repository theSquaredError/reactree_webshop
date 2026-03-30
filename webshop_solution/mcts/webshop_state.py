class WebShopState:
    """
    Minimal snapshot of a WebShop env state.
    Clone = env.reset(session_id) + replay action_history.
    This is possible because WebShop is a deterministic simulator.
    """

    def __init__(
        self,
        session_id: int,        # integer passed to env.reset(session=...) for deterministic replay
        instruction_text: str,
        action_history: list,   # all primitive env actions executed so far (from env.prev_actions)
        obs: str,               # current observation (informational; restored by replay)
        done: bool = False,
        reward: float = 0.0,
        step: int = 0,
        max_steps: int = 15,
    ):
        self.session_id = session_id
        self.instruction_text = instruction_text
        self.action_history = list(action_history)
        self.obs = obs
        self.done = done
        self.reward = reward
        self.step = step
        self.max_steps = max_steps

    def is_terminal(self):
        return self.done or self.step >= self.max_steps

    def restore_env(self, env):
        """
        Rewind `env` to this state by reset + replay.
        After this call, env.observation == state after last replayed action,
        and env.prev_actions == self.action_history.
        """
        env.reset(session=self.session_id, instruction_text=self.instruction_text)
        reward, done = 0.0, False
        for action in self.action_history:
            _, reward, done, _ = env.step(action)
        return reward, done

    @classmethod
    def capture(cls, env, session_id: int, instruction_text: str, max_steps: int = 15):
        """Snapshot current env state. Call this right before running MCTS."""
        return cls(
            session_id=session_id,
            instruction_text=instruction_text,
            action_history=list(env.prev_actions),   # WebAgentTextEnv tracks this
            obs=env.observation,
            done=False,
            reward=0.0,
            step=len(env.prev_actions),
            max_steps=max_steps,
        )