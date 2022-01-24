import numpy as np
import json


def load(file):
    return ExperienceBuffer(
        game_states=np.array(file['experience']['game_states']),
        search_probabilities=np.array(file['experience']['search_probabilities']),
        winner=np.array(file['experience']['winner'])
    )


class ExperienceBuffer:
    def __init__(self, game_states, search_probabilities, winner):
        self.game_states = game_states
        self.search_probabilities = search_probabilities
        self.winner = winner

    def save(self, file):  # npz
        '''
        out = {  # list because apparently json doesn't know what to do with numpy array :?
            'game_states': self.game_states,
            'search_probabilities': self.search_probabilities,
            'winner': self.winner,
        }
        json.dump(out, file)'''
        np.savez(file, game_states=self.game_states, search_probabilities=self.search_probabilities, winner=self.winner)
