import numpy as np
import json

class ExperienceBuffer:
    def __init__(self, states, actions, rewards):
        self.states = states
        self.rewards = rewards
        self.actions = actions

    def save(self, file):  # json
        out = {  # list because apparently json doesn't know what to do with numpy array :?
            'states': self.states.tolist(),
            'rewards': self.rewards.tolist(),
            'actions': self.actions.tolist(),
        }
        json.dump(out, file)

    def load(self, file):
        return ExperienceBuffer(
            states=np.array(file['experience']['states']),
            actions=np.array(file['experience']['actions']),
            rewards=np.array(file['experience']['rewards'])
        )
