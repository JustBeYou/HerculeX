import numpy as np

from .models.cnn_model import CNNModel
from .models.residual_model import ResidualModel
from .AbstractAgent import AbstractAgent
from minihex.HexGame import HexGame, Player

from .mcts.Tree import Tree
from .mcts.Node import Node
from .mcts.Edge import Edge

from datetime import datetime

class HerculexTheSecond(AbstractAgent):
    def __init__(self, board_size, epsilon, constant, num_simulations, collector, model) -> None:
        self.board_size = board_size

        self.model = model

        self.actions = np.arange(board_size ** 2)

        self.num_simulations = num_simulations

        self.constant = constant # constant used when calculating the value for each node
        self.collector = collector
        self.epsilon = epsilon
        self.tree = None
        self.root = None

    def build_tree(self, state, connected_stones):
        self.root = Node(state, 0, 0, connected_stones)
        self.tree = Tree(self.root, self.constant, self.board_size)

    def change_root(self, new_root):
        self.tree.root = new_root

    def get_action(self, state, connected_stones, info=None):
        node = Node(state, 0, 0, connected_stones)

        if self.tree is None or self.tree.check_node(node) is None:
            self.build_tree(state, connected_stones)
        else:
            node = self.tree.check_node(node)
            self.change_root(node)

        start = datetime.now()
        for idx in range(self.num_simulations):
            self.simulate()
        print('Finished a simulation, elapsed time: ' + str(datetime.now() - start))

        policy, rewards = self.get_policy_rewards()
        action, reward = self.choose_action(policy, rewards, state)

        return action

    def get_policy_rewards(self):
        policy = np.zeros(len(self.actions), dtype=np.integer)
        rewards = np.zeros(len(self.actions), dtype=np.float32)

        for action, edge in self.tree.root.edges:
            policy[action] = edge.N
            rewards[action] = edge.Q

        policy = policy / (np.sum(policy) * 1.0)
        return policy, rewards

    def choose_action(self, policy, rewards, state):
        sampled_value = float(np.random.uniform(0, 1))

        if sampled_value <= self.epsilon:  # exploratory move
            actions = [idx for idx, el in enumerate(policy) if el != 0]
            action = np.random.choice(actions)
        else:
            action_idx = np.random.multinomial(1, policy)
            action = np.where(action_idx == 1)[0][0]

        self.epsilon **= 2
        reward = rewards[action]

        return action, reward

    def simulate(self):
        leaf, reward, done, history = self.tree.get_best_leaf()

        reward = self.evaluate_leaf(leaf, reward, done, history)

        self.tree.back_propagation(leaf, reward, history)

    def evaluate_leaf(self, leaf, reward, done, history):
        if not done:
            reward, probabilities, valid_actions = self.get_predictions(leaf.state, history)
            probabilities = probabilities[valid_actions]

            for idx, action in enumerate(valid_actions):
                simulator = HexGame(active_player=leaf.state[1], board=leaf.state[0].copy(), focus_player=None,
                                    connected_stones=leaf.connected_stones.copy())

                new_state, new_reward, new_done = self.tree.simulate_next_state(simulator, action)

                new_node = Node(new_state, new_reward, new_done, simulator.regions)

                # Check if already in tree else add it
                node = self.tree.check_node(new_node)
                if not node:
                    self.tree.add_node(new_node)
                else:
                    new_node = node

                new_edge = Edge(leaf, new_node, probabilities[idx], action)
                leaf.edges.append((action, new_edge))

        return reward

    def get_predictions(self, state, history):
        game_state = [state[0]]

        if len(history) < 2:
            game_state.append(state[0])
            game_state.append(state[0])
        else:
            for edge in reversed(history[-2:]):  # add the previous two states to the prediction
                game_state.append(edge.parent.board)

        input = self.model.transform_input(game_state, state[1])

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

        self.collector.record_decision(input, probabilities)

        return reward, probabilities, valid_actions

    def save(self, path):
        self.model.save()

    def load(self, path):
        self.model.load()