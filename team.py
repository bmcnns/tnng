import random
from uuid import uuid4

from parameters import Parameters


class Team:

    def __init__(self, learner_population):
        self.id = uuid4()
        self.learners = []
        self.referenced_by = []
        self.lucky_breaks = 0

        size = random.randint(2, Parameters.MAX_INITIAL_TEAM_SIZE)

        # This has to the potential to be unsafe if
        # the learner population is not large enough to have 2 distinct actions.
        while len(set(learner.action for learner in self.learners)) < 2:
            self.learners = random.sample(learner_population, k=size)
            for learner in self.learners:
                learner.referenced_by.append(self.id)

    def is_root_team(self):
        return len(self.referenced_by) == 0

    def get_action(self, observation, visited=None):
        # Initialize visited set if not provided
        if visited is None:
            visited = set()

        # Sort learners by their bid in descending order
        sorted_learners = sorted(self.learners, key=lambda x: x.bid(observation), reverse=True)

        # Iterate through sorted learners to find an action
        for highest_bidder in sorted_learners:
            if highest_bidder.id in visited:
                continue  # Skip if already visited (cycle detection)

            visited.add(highest_bidder.id)

            if highest_bidder.is_atomic():
                return highest_bidder.action, highest_bidder, visited
            else:
                # If it's a team, delegate action selection to that team
                return highest_bidder.action.get_action(observation, visited)

        # If no valid action is found, raise an error
        raise RuntimeError("No atomic action found, but one was expected.")
