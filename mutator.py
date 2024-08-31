import random

from learner import Learner
from parameters import Parameters


class Mutator:
    @staticmethod
    def mutateLearner(learner):
        if random.random() < Parameters.ADD_NOISE_PROBABILITY:
            learner.add_noise(Parameters.NOISE_AMOUNT)

    @staticmethod
    def mutateTeam(tnng, team):
        ids = [learner.id for learner in team.learners]

        if random.random() < Parameters.ADD_LEARNER_PROBABILITY:
            if len(team.learners) < Parameters.MAX_LEARNER_COUNT:
                newLearner = random.choice(tnng.learnerPopulation)
                while newLearner.id in ids:
                    newLearner = random.choice(tnng.learnerPopulation)
                newLearner.referenced_by.append(team.id)
                team.learners.append(newLearner)

        if random.random() < Parameters.REMOVE_LEARNER_PROBABILITY:
            # Collect distinct atomic actions
            distinct_atomic_actions = set(learner.action for learner in team.learners if learner.is_atomic())

            # Ensure that at least 2 distinct atomic actions are always present
            if len(distinct_atomic_actions) > 2:
                # Select a random learner to remove
                removed_learner = random.choice(team.learners)

                if removed_learner.is_atomic():
                    # Check if removing this learner's action would still leave 2 distinct actions
                    # We simulate removal and check the new set of distinct actions
                    remaining_actions = set(
                        learner.action for learner in team.learners if learner.is_atomic() and learner != removed_learner.id
                    )

                    if len(remaining_actions) >= 2:
                        team.learners.remove(removed_learner)
                else:
                    # Safe to remove non-atomic learners
                    team.learners.remove(removed_learner)

        if random.random() < Parameters.NEW_LEARNER_PROBABILITY:
            if len(team.learners) < Parameters.MAX_LEARNER_COUNT:
                learner = Learner()
                learner.referenced_by.append(team.id)
                tnng.learnerPopulation.append(learner)
                team.learners.append(learner)

        learner = random.choice(team.learners)

        for learner in team.learners:
            Mutator.mutateLearner(learner)

        if random.random() < Parameters.POINTER_PROBABILITY:
            distinct_atomic_actions = set(learner.action for learner in team.learners if learner.is_atomic())

            if len(distinct_atomic_actions) >= 2:
                if learner.is_atomic():
                    # Check if removing this learner's action would still leave 2 distinct actions
                    # We simulate removal and check the new set of distinct actions
                    remaining_actions = set(
                        other_learner.action for other_learner in team.learners if
                        learner.is_atomic() and learner.id != other_learner.id
                    )

                    if len(remaining_actions) >= 2:
                        pointed_team = random.choice(tnng.teamPopulation)
                        learner.action = pointed_team
                        pointed_team.referenced_by.append(learner.id)

                else:
                    pointed_team = random.choice(tnng.teamPopulation)
                    learner.action = pointed_team
                    pointed_team.referenced_by.append(learner.id)
