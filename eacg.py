import math
import random

from database import Database
from learner import Learner
from mutator import Mutator
from parameters import Parameters
from team import Team

class EACG:
    def __init__(self):
        self.learnerPopulation = [Learner() for _ in range(Parameters.INITIAL_LEARNER_POPULATION_SIZE)]
        self.teamPopulation = [Team(self.learnerPopulation) for _ in range(Parameters.POPULATION_SIZE)]
        self.teamPopulation = []

    def get_root_teams(self):
        return list(filter(lambda x: x.is_root_team(), self.teamPopulation))
