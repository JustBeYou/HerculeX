import numpy as np
from .experience_buffer import ExperienceBuffer

class ExperienceCollector:
    def __init__(self):
        self.game_states = []
        self.search_probabilities = []
        self.winner = []
        self.current_episode_game_states = []
        self.current_episode_search_probabilities = []

    def init_episode(self):
        self.current_episode_game_states = []
        self.current_episode_search_probabilities = []

    def record_decision(self, game_state, search_probabilities):
        self.current_episode_game_states.append(game_state)  # we are interested in just the board
        self.current_episode_search_probabilities.append(search_probabilities)

    def complete_episode(self, reward):
        self.game_states += self.current_episode_game_states
        self.search_probabilities += self.current_episode_search_probabilities
        self.winner.append(reward)

        self.init_episode()  # reinitialize

    def to_buffer(self):
        return ExperienceBuffer(
            game_states=self.game_states,
            search_probabilities=self.search_probabilities,
            winner=self.winner
        )

