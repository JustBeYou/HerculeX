from abc import abstractmethod

class AbstractAgent:

    @abstractmethod
    def get_action(self, state, connected_stones, info=None):
        raise NotImplementedError

    @abstractmethod
    def save(self, path):
        raise NotImplementedError

    @abstractmethod
    def load(self, path):
        raise NotImplementedError