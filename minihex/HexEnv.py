import gym
from .HexGame import Player, HexGame
import numpy as np

class HexEnv(gym.Env):
    metadata = {"render.modes": ["ansi"]}

    def __init__(self, opponent_policy, reward_function,
                 player_color=Player.BLACK,
                 active_player=Player.BLACK,
                 board=None,
                 regions=None,
                 board_size=5,
                 debug=False):
        self.opponent_policy = opponent_policy
        self.reward_function = reward_function

        if board is None:
            board = Player.EMPTY * np.ones((board_size, board_size))

        self.initial_board = board
        self.active_player = active_player
        self.player = player_color
        self.simulator = None
        self.winner = None
        self.previous_opponent_move = None
        self.debug = debug

        # cache initial connection matrix (approx +100 games/s)
        self.initial_regions = regions

    @property
    def opponent(self):
        return Player((self.player + 1) % 2)

    def reset(self):
        if self.initial_regions is None:
            self.simulator = HexGame(self.active_player,
                                     self.initial_board.copy(),
                                     self.player,
                                     debug=self.debug)
            regions = self.simulator.regions.copy()
            self.initial_regions = regions
        else:
            regions = self.initial_regions.copy()
            self.simulator = HexGame(self.active_player,
                                     self.initial_board.copy(),
                                     self.player,
                                     connected_stones=regions,
                                     debug=self.debug)

        self.previous_opponent_move = None

        if self.player != self.active_player:
            info_opponent = {
                'state': self.simulator.board,
                'last_move_opponent': None,
                'last_move_player': None
            }
            self.opponent_move(info_opponent)

        info = {
            'state': self.simulator.board,
            'last_move_opponent': self.previous_opponent_move,
            'last_move_player': None
        }

        return (self.simulator.board, self.active_player), info

    def step(self, action):
        if not self.simulator.done:
            self.winner = self.simulator.make_move(action)

        opponent_action = None

        if not self.simulator.done:
            info_opponent = {
                'state': self.simulator.board,
                'last_move_opponent': action,
                'last_move_player': self.previous_opponent_move
            }
            opponent_action = self.opponent_move(info_opponent)

        reward = self.reward_function(self)

        info = {
            'state': self.simulator.board,
            'last_move_opponent': opponent_action,
            'last_move_player': action
        }

        return ((self.simulator.board, self.active_player), reward,
                self.simulator.done, info)

    def render(self, mode='ansi', close=False):
        board = self.simulator.board
        print(" " * 6, end="")
        for j in range(board.shape[1]):
            print(" ", j + 1, " ", end="")
            print("|", end="")
        print("")
        print(" " * 5, end="")
        print("-" * (board.shape[1] * 6 - 1), end="")
        print("")
        for i in range(board.shape[1]):
            print(" " * (1 + i * 3), i + 1, " ", end="")
            print("|", end="")
            for j in range(board.shape[1]):
                if board[i, j] == Player.EMPTY:
                    print("  O  ", end="")
                elif board[i, j] == Player.BLACK:
                    print("  B  ", end="")
                else:
                    print("  W  ", end="")
                print("|", end="")
            print("")
            print(" " * (i * 3 + 1), end="")
            print("-" * (board.shape[1] * 7 - 1), end="")
            print("")

    def opponent_move(self, info):
        state = (self.simulator.board, self.opponent)
        opponent_action = self.opponent_policy(state,
                                               info)
        self.winner = self.simulator.make_move(opponent_action)
        self.previous_opponent_move = opponent_action
        return opponent_action