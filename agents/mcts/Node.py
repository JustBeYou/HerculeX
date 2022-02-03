import numpy as np
from hashlib import sha1
from marshal import dumps


class Node:
    def __init__(self, state, reward, done, connected_stones):
        self.state = (state[0].copy(), state[1].copy())
        self.board = state[0].copy()
        self.player = state[1].copy()
        self.connected_stones = connected_stones.copy()

        self.reward = reward
        self.done = done
        self.edges = []

        self.id = sha1(dumps(self.board)).hexdigest()

    def nice_size(self):
        if len(self.edges) == 0:
            return 1

        t = 0
        for (action, edge) in self.edges:
            t += edge.child.nice_size()
        return t

    def __eq__(self, other):
        return self.id == other.id

    def is_leaf(self):
        return len(self.edges) == 0
