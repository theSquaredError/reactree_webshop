class WorkingMemory:

    def __init__(self):

        self.history = []
        self.observation = None

    def update(self, action, observation):

        self.history.append((action, observation))
        self.observation = observation