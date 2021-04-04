import numpy as np
from snakes_and_ladders import SECURITY, NORMAL, RISKY
from snakes_and_ladders import Board


def mean_dist_to_objective(states):
    dists = np.zeros(len(states))

    idx = np.logical_and(states >= 3, states <= 9)
    dists[idx] = 10 - states[idx]

    idx = states >= 10
    dists[idx] = 14 - states[idx]

    idx = states <= 2
    dists[idx] = 12 - states[idx] # mean of distance to objective considering two lanes

    return np.mean(dists)


def get_strategy(layout, circle):
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

    return policy
