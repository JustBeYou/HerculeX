import numpy as np
from .experience_buffer import ExperienceBuffer

class ExperienceCollector:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.current_episode_states = []
        self.current_episode_actions = []

    def init_episode(self):
        self.current_episode_states = []
        self.current_episode_actions = []

    def record_decision(self, state, action):
        self.current_episode_states.append(state[0]) # we are interested in just the board
        self.current_episode_actions.append(action)

    def complete_episode(self, reward):
        self.states += self.current_episode_states
        self.actions += self.current_episode_actions
        self.rewards += [reward for _ in range(len(self.current_episode_states))]  # every actions gets the same reward

        self.init_episode()  # reinitialize

    def to_buffer(self):
        return ExperienceBuffer(
            states= self.states,
            actions= self.actions,
            rewards= self.rewards
        )

