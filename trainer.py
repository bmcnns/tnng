import copy
import random
import time
from uuid import uuid4
import gymnasium
import numpy as np
from matplotlib import pyplot as plt

from database import Database
from mutator import Mutator
from parameters import Parameters
from eacg import EACG

from visualization import Debugger

# Global variable to toggle video capture
is_rendering = True


def toggle_rendering(event):
    global is_rendering
    if event.key == 'p':  # 'p' to play/pause
        is_rendering = not is_rendering
        print(f"Rendering {'enabled' if is_rendering else 'disabled'}.")


def run_environment(seed, tnng, root_team, generation, run_id):
    assert Parameters.ENVIRONMENT in ['CartPole-v1', 'LunarLander-v2', 'CliffWalking-v0', 'Taxi-v3'], 'Environment not implemented.'

    # Initialize the environment
    env = gymnasium.make(Parameters.ENVIRONMENT, render_mode="rgb_array")

    # Set the random seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)

    # Reset the environment and get the initial observation
    obs = env.reset(seed=seed)[0]

    step = 0
    training_data = []

    # Initialize the plot with two subplots: one for environment, one for policy graph
    fig, (ax_env, ax_graph) = plt.subplots(1, 2, figsize=(12, 6))

    # Initialize the environment image
    img = ax_env.imshow(np.zeros_like(env.render()))  # Initialize with a blank frame
    ax_env.axis('off')  # Turn off axis labels

    # Connect the keypress event to toggle rendering
    fig.canvas.mpl_connect('key_press_event', toggle_rendering)

    # Initialize the policy graph
    Debugger.plotTeam(root_team, tnng=tnng, ax=ax_graph)

    # Show the plot in non-blocking mode
    plt.show(block=False)

    # Run the environment loop
    while step < Parameters.MAX_NUM_STEPS:
        action, learner, visited = root_team.get_action(obs)  # Get action and learner from the team

        # Only render the current state if rendering is enabled
        if is_rendering:
            frame = env.render()  # Get the current frame in RGB format

            # Ensure that the frame is a valid NumPy array
            if isinstance(frame, np.ndarray):
                img.set_data(frame)  # Update the image data
                fig.canvas.draw()  # Update the canvas
                plt.pause(0.001)  # Small pause to create a real-time effectactionactionactionactionaction
            else:
                raise TypeError("The rendered frame is not a valid NumPy array.")

        # Take a step in the environment
        previous_state = obs

        obs, rew, term, trunc, info = env.step(action)

        next_state = obs

        # Train the learner with the transition data
        learner.train(previous_state, rew, next_state)

        # Increment step counter
        step += 1

        # Collect the training data for this step/action
        training_data.append({
            "run_id": run_id,
            "generation": generation,
            "team_id": root_team.id,
            "action": action,
            "reward": rew,
            "is_finished": term or trunc,
            "time_step": step,
            "time": time.time()
        })

        # Break the loop if the environment is finished
        if term or trunc:
            break

    # Close the plot window after the loop
    plt.close(fig)

    # Return the collected training data
    return training_data


def train(run_id, num_generations):
    eacg = EACG()

    for team in eacg.teamPopulation:
        Database.add_team(run_id, team)

        for program in team.learners:
            Database.add_program(run_id, program, team)

    seeds = [random.randint(0, 2 ** 31 - 1) for _ in range(num_generations)]

    fixed_seed = random.randint(0, 2 ** 31 - 1)
    seeds = [fixed_seed for _ in range(num_generations)]
    for generation, seed in zip(range(1, num_generations + 1), seeds):
        training_data = []
        for i, root_team in enumerate(eacg.get_root_teams()):
            print(f"Generation {generation}. Team {i + 1} of {Parameters.POPULATION_SIZE}")
            data = run_environment(seed, eacg, root_team, generation, run_id)
            training_data.extend(data)

        Database.add_training_data(training_data)

        print("Showing output now")
        print(Database.get_ranked_teams(run_id, generation).sort_values('rank').head(25))

        survivor_ids = Database.get_survivor_ids(run_id, generation)
        root_teams = eacg.get_root_teams()

        removed_teams = list(filter(lambda x: x.id not in survivor_ids, root_teams))
        survivors = list(filter(lambda x: x.id in survivor_ids, root_teams))

        # Apply lucky breaks
        lucky_break_ids = Database.get_ranked_teams(run_id, generation).sort_values('rank').head(Parameters.NUM_LUCKY_BREAKS)['team_id'].to_list()
        for team in filter(lambda x: x.id in lucky_break_ids, root_teams):
            team.lucky_breaks += 1

        for root_team in removed_teams:
            if root_team.lucky_breaks > 0:
                root_team.lucky_breaks -= 1
                continue
            for learner in root_team.learners:
                learner.referenced_by.remove(root_team.id)
            eacg.teamPopulation.remove(root_team)

        # this is done through a list comprehension because
        # removing elements from a list while iterating introduces bugs.
        eacg.learnerPopulation = [learner for learner in eacg.learnerPopulation if len(learner.referenced_by) > 0]

        print("Cloning existing teams and adding new teams to the database now")
        while len(eacg.get_root_teams()) < Parameters.POPULATION_SIZE:
            survivor = random.choice(survivors)
            clone = copy.deepcopy(survivor)
            clone.referenced_by = []

            clone.id = uuid4()
            # again, being careful not to edit lists while iterating through them
            for learner in list(clone.learners):
                learner.id = uuid4()
                learner.referenced_by = [clone.id]

            Mutator.mutateTeam(eacg, clone)
            eacg.teamPopulation.append(clone)


if __name__ == '__main__':
    print("Connecting to the database...")

    Database.connect(
        user=Parameters.DATABASE_USER_ID,
        password=Parameters.DATABASE_PASSWORD,
        host=Parameters.DATABASE_IP,
        port=Parameters.DATABASE_PORT,
        database=Parameters.DATABASE_NAME
    )

    print("Database connected.")

    run_id = uuid4()
    train(run_id=run_id, num_generations=600)

    print(f"Finished run: {run_id}")
