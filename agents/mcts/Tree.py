import numpy as np
from minihex.HexGame import HexGame, Player

class Tree:
    def __init__(self, root, const, board_size):
        self.root = root
        self.const = const  # constant used when calculating best value from the tree
        self.nodes = []
        self.board_size = board_size

        self.add_node(root)

    def __len__(self):
        return len(self.nodes)

    def add_node(self, node):
        self.nodes.append(node)

    def get_best_leaf(self):
        curr_node = self.root

        history = []
        reward, done = 0, 0

        while not curr_node.is_leaf():
            max_val = -999999

            N_parent = 0

            for action, edge in curr_node.edges:
                N_parent += edge.N

            for idx, (action, edge) in enumerate(curr_node.edges):
                # TODO: Check if this is the right formula and see what to do if node is root
                U = edge.P * self.const * np.sqrt(np.log(N_parent) / (edge.N + 1))
                Q = edge.Q

                if U + Q > max_val:
                    max_val = U + Q
                    best_action = action
                    best_edge = edge

            _, reward, done = self.simulate_next_state(curr_node.state, best_action)
            history.append(best_edge)
            curr_node = best_edge.child

        return curr_node, reward, done, history

    def back_propagation(self, leaf, reward, history):
        for edge in history:
            sign = 1 if edge.player == self.root.player else -1

            edge.N = edge.N + 1
            edge.W = edge.W + reward * sign
            edge.Q = edge.W / edge.N

    def action_to_coordinate(self, action):
        y = action // self.board_size
        x = action - self.board_size * y
        return (y, x)

    def reward_function(self, winner):
        if self.root.player == winner:
            return 1
        elif (self.root.player + 1) % 2 == winner:
            return -1
        return 0

    def simulate_next_state(self, state, action):
        simulator = HexGame(active_player=state[1], board=state[0].copy(), focus_player=None)

        winner = simulator.make_move(action)
        reward = self.reward_function(winner)

        return (simulator.board, Player(simulator.active_player)), reward, simulator.done





