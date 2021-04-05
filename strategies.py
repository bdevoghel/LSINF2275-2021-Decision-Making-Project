from testbench import compare_policies
import numpy as np
from snakes_and_ladders import SECURITY, NORMAL, RISKY
from snakes_and_ladders import Board
from typing import Tuple

# -----------------------------------------------------------------------------
# Strategies
# -----------------------------------------------------------------------------
"""
each strategy should follow the same template :
    - parameters layout and circle
    - return name and policy
each strategy should be added to strategies to be executed in the testbench
"""
def strategy_template(layout : list, circle : bool) -> Tuple[str, list] :
    # define policy
    policy = None
    # choose name and return
    return "name", policy

def risky(layout, circle) :
    policy = np.ones(15, dtype=int) * RISKY

    return "risky", policy

def secure(layout, circle) :
    policy = np.ones(15, dtype=int) * SECURITY

    return "secure", policy

def suboptimal(layout, circle):
    board = Board(layout, circle)

    policy = np.zeros(len(board.layout) - 1, dtype=int)
    for state in range(len(board.layout) - 1):
        state_expectation = {}
        for die in board.dice:
            possible_states = board.apply_delta(np.ones(len(die.possible_steps), dtype=int)*state, np.array(die.possible_steps))
            trapped_states, extra_cost = board.apply_traps(possible_states, np.ones(len(possible_states), dtype=bool))

            normal_dists = mean_dist_to_objective(possible_states)
            trapped_dists = mean_dist_to_objective(trapped_states) + np.mean(extra_cost)

            if die.type == SECURITY:
                state_expectation[die] = normal_dists
            elif die.type == NORMAL:
                state_expectation[die] = (normal_dists + trapped_dists) / 2
            elif die.type == RISKY:
                state_expectation[die] = trapped_dists
        min_die = min(state_expectation, key=state_expectation.get)
        policy[state] = min_die.type

    return "suboptimal", policy

# -----------------------------------------------------------------------------
# Other functions
# -----------------------------------------------------------------------------

def mean_dist_to_objective(states):
    dists = np.zeros(len(states))

    idx = np.logical_and(states >= 3, states <= 9)
    dists[idx] = 10 - states[idx]

    idx = states >= 10
    dists[idx] = 14 - states[idx]

    idx = states <= 2
    dists[idx] = 12 - states[idx] # mean of distance to objective considering two lanes

    return np.mean(dists)

# -----------------------------------------------------------------------------
# List of strategies
# -----------------------------------------------------------------------------

# add strategies to the list if you want them to be executed in the testbench
# each strategy should return a name and a np.array of dice (integers)
strategies = [risky, secure, suboptimal]

def get_policies(layout, circle) :
    policies = []
    for strat in strategies :
        policies.append(strat(layout, circle))
    return policies