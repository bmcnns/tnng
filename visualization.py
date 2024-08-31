import eacg
from parameters import Parameters
from learner import Learner
from team import Team

from typing import List
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

import matplotlib.pyplot as plt


import math
import uuid


class Debugger:
    """
    The Debugger classes provides functionality to display information
    about the team and program population as well as the ability to visualize
    the policy graphs created by root teams.
    """

    def plotTeam(_team: Team, tnng, ax) -> None:
        """
        Display the policy graph for a root team, showing all learners, their actions,
        and recursively drawing any referenced teams and their learners.
        """
        G = nx.DiGraph()

        def process_team(team):
            if team.id in visited:
                return  # Avoid processing the same team multiple times

            visited.add(team.id)

            # Add the team as a node with a distinct color and larger size
            G.add_node(team.id, label=f"Team {str(team.id)[:5]}", color='lightblue', node_type='team', size=600)

            # Iterate through the learners of the team
            for learner in team.learners:
                # Add a node for the learner with smaller size
                G.add_node(learner.id, label="{}", node_type='learner', color='lightgrey', size=300)
                G.add_edge(team.id, learner.id)  # Connect the team to the learner

                if learner.is_atomic():
                    # If the learner is atomic, add a node for the action and connect it to the learner
                    action_id = str(uuid.uuid4())  # Create a unique ID for the action node
                    G.add_node(action_id, label=learner.action, color='white', node_type='action', size=150)
                    G.add_edge(learner.id, action_id)  # Connect the learner to the action
                else:
                    # For non-atomic learners, find the referenced team and process it recursively
                    for other_team in tnng.teamPopulation:
                        if other_team.id == learner.action.id:
                            process_team(other_team)
                            G.add_edge(learner.id, other_team.id)  # Connect the learner to the referenced team

        visited = set()

        # Start processing the root team
        process_team(_team)

        # Plot the graph using 'neato' layout
        pos = nx.nx_agraph.graphviz_layout(G, prog='neato')
        colors = [G.nodes[n]['color'] for n in G.nodes]
        labels = nx.get_node_attributes(G, 'label')
        sizes = [G.nodes[n]['size'] for n in G.nodes]

        # Draw the graph with different node sizes
        nx.draw(G, pos, ax=ax, with_labels=True, labels=labels, node_color=colors, node_size=sizes,
                # Different sizes for teams and learners
                arrows=True, edge_color='gray', font_size=8, font_weight='bold')

        # Draw edges with longer length to actions
        nx.draw_networkx_edges(G, pos, ax=ax, width=1.5, arrows=True, arrowstyle='-|>', min_source_margin=10,
                               min_target_margin=10)

