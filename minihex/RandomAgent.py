import numpy as np
import random

class RandomAgent:
    def __init__(self, board_size=11) -> None:
        # caching this array improves performance by 20%
        self.actions = np.arange(board_size ** 2)

    def get_action(self, state, info=None):
        board, player = state
        valid_actions = self.actions[board.flatten() == player.EMPTY]
        choice = int(random.random() * len(valid_actions))
        return valid_actions[choice]