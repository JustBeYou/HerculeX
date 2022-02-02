import numpy as np
from minihex.HexGame import HexGame, Player

class Tree:
    def __init__(self, root, const, board_size):
        self.root = root
        self.const = const  # constant used when calculating best value from the tree
        self.board_size = board_size

        self.nodes_hash = {}
        self.add_node(root)

    def nice_size(self):
        return self.root.nice_size()

    def __len__(self):
        return len(self.nodes_hash)

    def add_node(self, node):
        self.nodes_hash[node.id] = node

    def check_node(self, node):
        if node.id in self.nodes_hash:
           return self.nodes_hash[node.id]

        return None

    def get_best_leaf(self):
        curr_node = self.root

        history = []
        reward, done = 0, 0

        while not curr_node.is_leaf():
            max_val = -np.inf

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

            #_, reward, done = self.simulate_next_state(curr_node.state, best_action)
            history.append(best_edge)
            curr_node = best_edge.child
            reward = curr_node.reward
            done = curr_node.done

        return curr_node, reward, done, history

    def back_propagation(self, leaf, reward, history):
        for edge in history:
            sign = 1 if edge.player == self.root.player else -1

            edge.N = edge.N + 1
            edge.W = edge.W + reward * sign
            edge.Q = edge.W / edge.N

    def reward_function(self, winner):
        if self.root.player == winner:
            return 1
        elif (self.root.player + 1) % 2 == winner:
            return -1
        return 0

    def action_to_coordinate(self, action):
        y = action // self.board_size
        x = action - self.board_size * y
        return (y, x)

    def simulate_next_state(self, simulator, action):
        winner = simulator.make_move(action)
        reward = self.reward_function(winner)

        return (simulator.board, Player(simulator.active_player)), reward, simulator.done
