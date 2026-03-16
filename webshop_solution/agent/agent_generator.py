class ActionGenerator:

    def __init__(self, llm):
        self.llm = llm

    def decide_action(self, goal, observation):

        prompt = f"""
Goal: {goal}

Current webpage:
{observation}

You must output ONLY one action.

Valid formats:
search[query]
choose[item]

Action:
"""

        action = self.llm.generate(prompt)

        return action.strip()