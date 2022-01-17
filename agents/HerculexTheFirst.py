import numpy as np
from .models.cnn_model import CNNModel
from .AbstractAgent import AbstractAgent


class HerculexTheFirst(AbstractAgent):
    def __init__(self, board_size, epsilon, constant, num_simulations, collector, model) -> None:
        self.model = CNNModel(board_size=board_size)
        self.actions = np.arange(board_size ** 2)

        self.collector = collector
        self.epsilon = epsilon

    def get_action(self, state, info=None):
        sampled_value = float(np.random.uniform(0, 1))

        board, player = state
        valid_actions = self.actions[board.flatten() == player.EMPTY]

        if sampled_value < self.epsilon:
            action = valid_actions[int(np.random.random() * len(valid_actions))]

        # TODO: maybe do stratified sampling and see how that affects if this bot turns out okay
        else:  # predict
            actions = self.model.predict(board)[0]
            actions = [act if idx in valid_actions else 0.0 for idx, act in enumerate(actions)]  # make invalid actions have weight 0
            action = np.argmax(actions)

        if self.collector is not None:
            self.collector.record_decision(state=state, action=action)

        self.epsilon **= 2
        return action

    def save(self, path):
        print("What is there to save other than the weights")

    def load(self, path):
        print("What is there to save other than the weights")