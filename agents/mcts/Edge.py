from .Node import Node


class Edge:
    def __init__(self, parent: Node, child: Node, probability, action):
        self.parent = parent
        self.child = child

        self.action = action
        self.player = child.player

        self.N = 1
        self.W = 0
        self.Q = 0
        self.P = probability
