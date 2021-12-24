
class Node:
    def __init__(self, state):
        self.state = state
        self.board = state[0]
        self.player = state[1]

        self.edges = []

    def __eq__(self, other):
        return all((self.board == other.board)[0]) and self.player == other.player

    def is_leaf(self):
        return len(self.edges) == 0
