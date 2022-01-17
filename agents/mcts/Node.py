import numpy as np
from hashlib import sha1
from marshal import dumps

class Node:
    def __init__(self, state):
        self.state = state
        self.board = state[0]
        self.player = state[1]

        self.edges = []

        self.id = sha1(dumps(self.board)).hexdigest()

    def __eq__(self, other):
        return self.id == other.id

    def is_leaf(self):
        return len(self.edges) == 0
