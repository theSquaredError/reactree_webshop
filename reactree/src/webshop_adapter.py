#filepath: reactree/src/webshop_adapter.py
import gym
from typing import Any, Dict, Optional
from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
# Adapter to expose a minimal Reactree-friendly interface around the
# WebAgentTextEnv registered as 'WebAgentTextEnv-v0' in this repo.
class ReactreeWebshopAdapter:
    """
    Minimal adapter that exposes:
      - reset(session: Optional[str]=None, instruction_text: Optional[str]=None) -> Dict
      - step(action: str) -> (state_dict, reward, done, info)
      - get_available_actions() -> Dict
      - get_state() -> Any
    """

    def __init__(self, *, observation_mode: str = "text", num_products: Optional[int] = None, **kwargs):
        make_args = {"observation_mode": observation_mode}
        if num_products is not None:
            make_args["num_products"] = num_products
        make_args.update(kwargs)
        # create env via gym registry used elsewhere in the repo
        self.env = gym.make("WebAgentTextEnv-v0", **make_args)

    def reset(self, session: Optional[str] = None, instruction_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Reset underlying env. If the env supports session/instruction args, they are passed.
        Returns a packed dict with 'observation', 'instruction_text', 'raw_state', 'available_actions'.
        """
        # Try to call reset with common signatures; fallback to no-arg reset
        try:
            obs = self.env.reset(session=session, instruction_text=instruction_text)
            # some envs return (obs, info)
            if isinstance(obs, tuple):
                observation = obs[0]
            else:
                observation = obs
        except TypeError:
            # fallback to no-arg reset
            out = self.env.reset()
            if isinstance(out, tuple):
                observation = out[0]
            else:
                observation = out

        return self._pack(observation)

    def step(self, action: str):
        """
        Step env with an action string (e.g. 'search[shoes]' or 'click[Buy Now]').
        Returns (packed_state, reward, done, info)
        """
        out = self.env.step(action)
        # standard gym tuple (obs, reward, done, info)
        if len(out) == 4:
            observation, reward, done, info = out
        else:
            # be defensive
            observation = out[0]
            reward = out[1] if len(out) > 1 else 0.0
            done = out[2] if len(out) > 2 else False
            info = out[3] if len(out) > 3 else {}

        return self._pack(observation), reward, done, info

    def get_available_actions(self) -> Dict[str, Any]:
        """Return current available actions as provided by the env (if supported)."""
        return getattr(self.env, "get_available_actions", lambda: {})()

    def get_state(self) -> Any:
        """Return underlying env state if present (best-effort)."""
        return getattr(self.env, "state", getattr(self.env, "server", None))

    def _pack(self, observation: Any) -> Dict[str, Any]:
        instruction = getattr(self.env, "instruction_text", None)
        packed = {
            "observation": observation,
            "instruction_text": instruction,
            "raw_state": self.get_state(),
            "available_actions": self.get_available_actions(),
        }
        return packed


if __name__ == "__main__":
    # quick smoke test (similar to run_envs/run_web_agent_text_env.py)
    adapter = ReactreeWebshopAdapter(observation_mode="text", num_products=10)
    s = adapter.reset()
    print("instruction:", s.get("instruction_text"))
    print("available:", s.get("available_actions"))
    acts = s.get("available_actions", {}).get("clickables", [])
    if acts:
        ns, r, d, info = adapter.step(f"click[{acts[0]}]")
        print("took action", acts[0], "reward", r, "done", d)
