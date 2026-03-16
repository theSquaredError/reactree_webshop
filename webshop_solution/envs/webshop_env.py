import gym
from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

class WebShopEnv:
    def __init__(self, num_products=100):
        self.env = gym.make(
            "WebAgentTextEnv-v0",
            observation_mode="text",
            num_products=num_products,
        )

    @property
    def observation(self):
        return self.env.observation

    @property
    def instruction_text(self):
        return getattr(self.env, "instruction_text", "")

    def reset(self, session=None, instruction_text=None):
        return self.env.reset(session=session, instruction_text=instruction_text)

    def step(self, action):
        return self.env.step(action)

    def get_available_actions(self):
        return self.env.get_available_actions()