import numpy as np
import random
from .AbstractAgent import AbstractAgent

import os
import binascii

SEED = int(binascii.hexlify(os.urandom(8)), 16)
random.seed(SEED)

class RandomAgent(AbstractAgent):
    def __init__(self, board_size=11, **kwargs) -> None:
        # caching this array improves performance by 20%
        self.actions = np.arange(board_size ** 2)
        self.model = None
        self.id = "randomvirgin"

    def get_action(self, state, connected_stones, history, info=None):
        board, player = state
        board = board.copy() ### WTF!!!
        player = player.copy()
        valid_actions = self.actions[board.flatten() == player.EMPTY]
        choice = int(random.random() * len(valid_actions))

        return valid_actions[choice]

    def load(self, path):
        print(f"Loading {path} for a random agent has no sense.")

    def save(self, path):
        print(f"Saving {path} for a random agent has no sense.")
