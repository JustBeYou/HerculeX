import numpy as np

from .models.cnn_model import CNNModel
from .models.residual_model import ResidualModel
from .AbstractAgent import AbstractAgent
from minihex.HexGame import HexGame, Player

from .mcts.Tree import Tree
from .mcts.Node import Node
from .mcts.Edge import Edge

import constants
from time import sleep, time

from random import randint

from time import time_ns

from datetime import datetime

import tensorflow as tf

import gc

class HerculexTheSecond(AbstractAgent):
    def __init__(self, board_size, epsilon, constant, num_simulations, collector, model = None) -> None:
        self.board_size = board_size


        if model is None:
            self.id = f"0_{randint(0, int(1e8))}"
            model = ResidualModel(regularizer=constants.REGULARIZER, learning_rate=constants.LEARNING_RATE,
                                  input_dim=constants.INPUT_DIM, output_dim=constants.OUTPUT_DIM,
                                  hidden_layers=constants.HIDDEN, momentum=constants.MOMENTUM,
                                  id=self.id)
        else:
            self.id = model.id

        self.model = model

        self.actions = np.arange(board_size ** 2)

        self.num_simulations = num_simulations

        self.constant = constant  # constant used when calculating the value for each node
        self.collector = collector
        self.epsilon = epsilon
        self.tree = None
        self.root = None

    def build_tree(self, state, connected_stones):
        self.root = Node(state, 0, 0, connected_stones)
        self.tree = Tree(self.root, self.constant, self.board_size)

    def change_root(self, new_root):
        self.tree.root = new_root

    def reset(self):
        self.tree = None
        self.root = None
        gc.collect()

    def get_action(self, state, connected_stones, history, info=None):
        #print("[DEBUG] START", state[1])
        stime = time()
        node = Node(state, 0, 0, connected_stones)

        if self.tree is None or self.tree.check_node(node) is None:
            self.build_tree(state, connected_stones)
        else:
            node = self.tree.check_node(node)
            self.change_root(node)

        #print("[DEBUG] MID 1", state[1])

        for idx in range(self.num_simulations):
            self.simulate()

        stime2 = time()
        #print("[DEBUG] MID ", stime2 - stime)

        #print("[DEBUG] MID 2", state[1])

        policy, rewards = self.get_policy_rewards()
        action, reward = self.choose_action(policy, rewards)

        #print("[DEBUG] MID fuck", state[1])

        # save actual search probs and gameState
        if len(history) < 2:
            game_state = [state[0].copy(), state[0].copy(), state[0].copy()]
        else:
            game_state = np.zeros(shape=(3, self.board_size, self.board_size))
            for idx, elem_state in enumerate(reversed(history[-2:])):  # add the previous two states to the prediction
                game_state[idx] = elem_state[0].copy()

        #print("[DEBUG] MID 3", state[1])

        input = self.model.transform_input(game_state, state[1].copy())

        if self.collector is not None:
            self.collector.record_decision(np.reshape(input, input.shape[1:]), policy)

        #print(f"[DEBUG] END", state[1], f"{action} {time() - stime2}")
        self.reset()
        return action

    def get_policy_rewards(self):
        policy = np.zeros(len(self.actions), dtype=np.integer)
        rewards = np.zeros(len(self.actions), dtype=np.float32)

        for action, edge in self.tree.root.edges:
            policy[action] = edge.N
            rewards[action] = edge.Q

        policy = policy / (np.sum(policy) * 1.0)
        return policy, rewards

    def choose_action(self, policy, rewards):
        sampled_value = float(np.random.uniform(0, 1))

        if sampled_value <= self.epsilon and not constants.RELEASE:  # exploratory move
            actions = [idx for idx, el in enumerate(policy) if el != 0]
            action = np.random.choice(actions)
            #print("[DEBUG] Explore")
        else:
            action_idx = np.random.multinomial(1, policy)
            action = np.where(action_idx == 1)[0][0]
            #print("[DEBUG] Exploit")

        self.epsilon *= constants.EPSILON_DECAY
        reward = rewards[action]

        #print(f"[DEBUG] Play {action} {reward}")
        return action, reward

    def simulate(self):
        #stime = datetime.now()
        leaf, reward, done, history = self.tree.get_best_leaf()
        #print('[DEBUG] Best leaf time: ', datetime.now()-stime)


        reward = self.evaluate_leaf(leaf, reward, done, history)

        #stime = datetime.now()
        self.tree.back_propagation(leaf, reward, history)
        #print('[DEBUG] Back prop time: ', datetime.now() - stime)

    def evaluate_leaf(self, leaf, reward, done, history):
        if not done:
            reward, probabilities, valid_actions = self.get_predictions([leaf.state[0].copy(), leaf.state[1].copy()], history)

            probabilities = probabilities[valid_actions]

            for idx, action in enumerate(valid_actions):
                simulator = HexGame(active_player=leaf.state[1].copy(), board=leaf.state[0].copy(), focus_player=None,
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

        #print (f"[DEBUG] evaluated {len(self.tree)}")

        return reward

    def get_predictions(self, state, history):
        game_state = None

        if len(history) < 2:
            game_state = [state[0].copy(), state[0].copy(), state[0].copy()]
        else:
            game_state = np.zeros(shape=(3, self.board_size, self.board_size))
            for idx, edge in enumerate(reversed(history[-2:])):  # add the previous two states to the prediction
                game_state[idx] = edge.parent.board.copy()
            game_state[2] = state[0].copy()

        input = self.model.transform_input(game_state, state[1].copy())

        #stime = datetime.now()
        predictions = self.model.predict(input)
        #print('[DEBUG] Is predict fucked? : ', datetime.now() - stime, self.model.id)

        reward = predictions[0][0].numpy()
        probabilities = predictions[1][0].numpy()

        board, player = state
        board = board.copy() #### WTF!!!
        player = player.copy()
        valid_actions = self.actions[board.flatten() == player.EMPTY]

        #array_sum = np.sum(probabilities)
        #if np.isnan(array_sum):
        #    print(probabilities, valid_actions, input)


        mask = np.ones(probabilities.shape, dtype=bool)
        mask[valid_actions] = False
        probabilities[mask] = -100

        # SOFTMAX function in order to convert all probabilities between 0-1 with the sum of 1 including the ones
        # for the actions that are not allowed
        odds = np.exp(probabilities)

        #print(odds)

        probabilities = odds / np.sum(odds)

        return reward, probabilities, valid_actions

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model.load(path)
        self.id = self.model.id