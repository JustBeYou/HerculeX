import numpy as np
from enum import IntEnum

# TODO: implement swap rule maybe

class Player(IntEnum):
    BLACK = 0
    WHITE = 1
    EMPTY = 2

class HexGame(object):
    def __init__(self, active_player, board,
                 focus_player, connected_stones=None, debug=False):
        self.board = board
        # track number of empty fields for speed
        self.empty_fields = np.count_nonzero(board == Player.EMPTY)

        if debug:
            self.make_move = self.make_move_debug
        else:
            self.make_move = self.fast_move

        if connected_stones is None:
            self.regions = np.stack([
                np.pad(np.zeros_like(self.board), 1),
                np.pad(np.zeros_like(self.board), 1)
            ], axis=0)
            self.regions[Player.WHITE][:, 0] = 1
            self.regions[Player.BLACK][0, :] = 1
            self.regions[Player.WHITE][:, self.board_size + 1] = 2
            self.regions[Player.BLACK][self.board_size + 1, :] = 2
        else:
            self.regions = connected_stones

        self.region_counter = np.zeros(2)
        self.region_counter[Player.BLACK] = np.max(self.regions[Player.BLACK]) + 1
        self.region_counter[Player.WHITE] = np.max(self.regions[Player.WHITE]) + 1

        if connected_stones is None:
            for y, row in enumerate(board):
                for x, value in enumerate(row):
                    if value == Player.BLACK:
                        self.active_player = Player.BLACK
                        self.flood_fill((y, x))
                    elif value == Player.WHITE:
                        self.active_player = Player.WHITE
                        self.flood_fill((y, x))

        self.active_player = active_player
        self.player = focus_player
        self.done = False
        self.winner = None

        self.actions = np.arange(self.board_size ** 2)

    @property
    def board_size(self):
        return self.board.shape[1]

    def is_valid_move(self, action):
        coords = self.action_to_coordinate(action)
        return self.board[coords[0], coords[1]] == Player.EMPTY

    def make_move_debug(self, action):
        if not self.is_valid_move(action):
            raise IndexError(("Illegal move "
                             f"{self.action_to_coordinate(action)}"))

        return self.fast_move(action)

    def fast_move(self, action):
        y, x = self.action_to_coordinate(action)
        self.board[y, x] = self.active_player
        self.empty_fields -= 1

        self.flood_fill((y, x))

        winner = None
        regions = self.regions[self.active_player]
        if regions[-1, -1] == 1:
            self.done = True
            winner = Player(self.active_player)
            self.winner = winner
        elif self.empty_fields <= 0:
            self.done = True
            winner = None

        self.active_player = (self.active_player + 1) % 2
        return winner

    def coordinate_to_action(self, coords):
        return np.ravel_multi_index(coords, (self.board_size, self.board_size))

    def action_to_coordinate(self, action):
        y = action // self.board_size
        x = action - self.board_size * y
        return (y, x)

    def get_possible_actions(self):
        return self.actions[self.board.flatten() == Player.EMPTY]

    def flood_fill(self, position):
        regions = self.regions[self.active_player]

        y, x = (position[0] + 1, position[1] + 1)
        neighborhood = regions[(y - 1):(y + 2), (x - 1):(x + 2)].copy()
        neighborhood[0, 0] = 0
        neighborhood[2, 2] = 0
        adjacent_regions = sorted(set(neighborhood.flatten().tolist()))

        # region label = 0 is always present, but not a region
        adjacent_regions.pop(0)

        if len(adjacent_regions) == 0:
            regions[y, x] = self.region_counter[self.active_player]
            self.region_counter[self.active_player] += 1
        else:
            new_region_label = adjacent_regions.pop(0)
            regions[y, x] = new_region_label
            for label in adjacent_regions:
                regions[regions == label] = new_region_label