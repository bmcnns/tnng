import random
from uuid import uuid4

from neuralnet import NeuralNet
from parameters import Parameters


class Learner:

    def __init__(self):
        self.id = uuid4()
        self.neuralnet = NeuralNet()
        self.action = random.choice(Parameters.ACTIONS)
        self.referenced_by = []

    def bid(self, observation):
        prediction, _ = self.neuralnet.forward(observation)
        return prediction

    def is_atomic(self):
        return self.action in Parameters.ACTIONS

    def train(self, previous_state, reward, next_state):
        self.neuralnet.backward(previous_state, reward, next_state)

    def add_noise(self, std):
        self.neuralnet.add_noise(std)