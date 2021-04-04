import snakes_and_ladders as SaL
from snakes_and_ladders import SECURITY, NORMAL, RISKY
from snakes_and_ladders import ORDINARY, RESTART, PENALTY, PRISON, GAMBLE
import numpy as np

# -----------------------------------------------------------------------------
# Testing constants
# -----------------------------------------------------------------------------

# print information to standard output
verbose = True
# number of iterations for empirical tests
nb_iterations = 1e5
# filename to write results
filename = "results.txt"

# -----------------------------------------------------------------------------
# Layouts
# -----------------------------------------------------------------------------

# only ORDINARY squares, no traps
layout_ordinary = np.ones(15) * ORDINARY

# only PENALTY (go back 3) traps
layout_penalty = np.ones(15) * PENALTY
layout_penalty[[0, -1]] = ORDINARY  # start and goal squares must be ordinary

# only PRISON (stay one turn) traps
layout_prison = np.ones(15) * PRISON
layout_prison[[0, -1]] = ORDINARY   # start and goal squares must be ordinary

# only GAMBLE (random teleportation)
layout_gamble = np.ones(15) * GAMBLE
layout_gamble[[0, -1]] = ORDINARY   # start and goal squares must be ordinary

# random initialized traps
layout_random = np.random.randint(low=0, high=5, size=15)
layout_random[[0, -1]] = ORDINARY   # start and goal squares must be ordinary

# custom layout
layout_custom1 = np.ones(15) * ORDINARY
layout_custom1[[7, 8, 12]] = GAMBLE, RESTART, PENALTY

# custom layout
layout_custom2 = np.array(
           [ORDINARY,   GAMBLE,     RESTART,    GAMBLE,     PENALTY, 
            ORDINARY,   PRISON,     ORDINARY,   RESTART,    ORDINARY, 
            PRISON,     PENALTY,    RESTART,    GAMBLE,     ORDINARY])

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

def test_markov(layout, circle=False, name="") :
    """runs the markov decision tests, prints in the output file and returns 
       the optimal dice to use"""
    results = SaL.test_markovDecision(layout, circle, name, verbose)
    # TODO print results to file
    return results

def test_empirical() :
    """runs the empirical tests and prints in the output file"""
    # TODO

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # SaL.test_markovDecision(layout_ordinary, False, "ORDINARY")
    # result = SaL.test_markovDecision(layout_prison, False, "PRISON")
    # SaL.test_empirically(layout_prison, False, *result)
    # SaL.test_markovDecision(layout_penalty, False, "PENALTY")
    # SaL.test_markovDecision(layout_random, False, "RANDOM")
    # SaL.test_markovDecision(layout_custom1, False, "CUSTOM1")

    result = SaL.test_markovDecision(layout_custom2, False, "CUSTOM2", verbose=True)
    SaL.test_empirically(layout_custom2, False, *result, nb_iter=1e5, verbose=True)

    result = SaL.test_markovDecision(layout_custom2, True, "CUSTOM2", verbose=True)
    SaL.test_empirically(layout_custom2, True, *result, nb_iter=1e5, verbose=True)

    # SaL.test_empirically(layout_custom2, True, policy=np.array([SECURITY for _ in range(15)]))
    # SaL.test_empirically(layout_custom2, True, policy=np.array([NORMAL for _ in range(15)]))
    # SaL.test_empirically(layout_custom2, True, policy=np.array([RISKY for _ in range(15)]))
    # SaL.test_empirically(layout_custom2, True, policy=np.array([np.random.randint(RISKY) + 1 for _ in range(15)]))
    # SaL.test_empirically(layout_custom2, True, policy=None)
