import numpy as np

from .models.cnn_model import CNNModel
from .models.residual_model import ResidualModel
from .AbstractAgent import AbstractAgent

from .mcts.Tree import Tree
from .mcts.Node import Node
from .mcts.Edge import Edge


class HerculexTheSecond(AbstractAgent):
    def __init__(self, board_size, constant, num_simulations) -> None:
        self.board_size = board_size

        hidden = [
            {'filters': 64, 'kernel_size': (4, 4)}
            , {'filters': 64, 'kernel_size': (4, 4)}
        ]

        self.model = ResidualModel(0.00001, 0.1, (board_size, board_size, 1), board_size ** 2, hidden, 0.9)
        self.actions = np.arange(board_size ** 2)

        self.num_simulations = num_simulations

        self.constant = constant # constant used when calculating the value for each node
        self.collector = None
        self.epsilon = 0.8
        self.tree = None
        self.root = None

    def set_collector(self, collector):
        self.collector = collector

    def build_tree(self, state):
        self.root = Node(state)
        self.tree = Tree(self.root, self.constant, self.board_size)

    def change_root(self, new_root):
        self.tree.root = new_root

    def get_action(self, state, info=None):
        node = Node(state)

        if self.tree is None or node not in self.tree.nodes:
            self.build_tree(state)
        else:
            self.change_root(node)

        for idx in range(self.num_simulations):
            self.simulate()

        policy, rewards = self.get_policy_rewards()
        action, reward = self.choose_action(policy, rewards)

        return action

    def get_policy_rewards(self):
        policy = np.zeros(len(self.actions), dtype=np.integer)
        rewards = np.zeros(len(self.actions), dtype=np.float32)

        for action, edge in self.tree.root.edges:
            policy[action] = edge.stats['N']
            rewards[action] = edge.stats['Q']

        policy = policy / (np.sum(policy) * 1.0)
        return policy, rewards

    def choose_action(self, policy, rewards):
        sampled_value = float(np.random.uniform(0, 1))

        if sampled_value >= self.epsilon:
            actions = np.argwhere(policy == max(policy))
            action = np.random.choice(actions)[0]
        else:
            action_idx = np.random.multinomial(1, policy)
            action = np.where(action_idx == 1)[0][0]
            self.epsilon **= 2

        reward = rewards[action]

        return action, reward

    def simulate(self):
        leaf, reward, done, history = self.tree.get_best_leaf()

        reward = self.evaluate_leaf(leaf, reward, done)

        self.tree.back_propagation(leaf, reward, history)

    def evaluate_leaf(self, leaf, reward, done):
        if not done:

            reward, probabilities, valid_actions = self.get_predictions(leaf.state)
            probabilities = probabilities[valid_actions]

            for idx, action in enumerate(valid_actions):
                new_state, _, _ = self.tree.simulate_next_state(leaf.state, action)

                new_node = Node(new_state)

                # Check if already in tree else add it
                node = [node for node in self.tree.nodes if node == new_node]
                if not node:
                    self.tree.add_node(new_node)
                else:
                    new_node = node[0]

                new_edge = Edge(leaf, new_node, probabilities[idx], action)
                leaf.edges.append((action, new_edge))

        return reward

    def get_predictions(self, state):
        # TODO: State preparation for the model
        input = self.model.transform_input(state)

        predictions = self.model.predict(input)
        reward = predictions[0][0]
        probabilities = predictions[1][0]

        board, player = state
        valid_actions = self.actions[board.flatten() == player.EMPTY]

        mask = np.ones(probabilities.shape, dtype=bool)
        mask[valid_actions] = False
        probabilities[mask] = -100

        # SOFTMAX function in order to convert all probabilities between 0-1 with the sum of 1 including the ones
        # for the actions that are not allowed
        odds = np.exp(probabilities)
        probabilities = odds / np.sum(odds)

        return reward, probabilities, valid_actions

    def save(self, path):
        print("What is there to save other than the weights")

    def load(self, path):
        print("What is there to save other than the weights")