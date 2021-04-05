from os import getpgid
import snakes_and_ladders as SaL
from snakes_and_ladders import SECURITY, NORMAL, RISKY
from snakes_and_ladders import ORDINARY, RESTART, PENALTY, PRISON, GAMBLE
import strategies
import numpy as np

# -----------------------------------------------------------------------------
# Testing constants
# -----------------------------------------------------------------------------

# print information to standard output 
#   0 for no verbose,
#   2 for max verbose
verbose = 1
# number of iterations for empirical tests
nb_iterations = 1e5
# filename to write results
filename = "results.txt"
# open file in append mode
f = open(filename, 'a')

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

def test_markov(layout, circle=False, name="markov", write_file=True) :
    """runs the markov decision tests, prints in the output file and returns 
       the optimal dice to use"""
    markov_exp, dice = SaL.test_markovDecision(layout, circle, verbose == 2)
    # write
    if write_file :
        print("markov", file=f)
        print(f"    name        : {name}", file=f)
        print(f"    layout      : {layout}", file=f)
        print(f"    circle      : {circle}", file=f)
        print(f"    dice        : {dice}", file=f)
        print(f"    expectation : {markov_exp[0]:>7.4f}", file=f)
        print("", file=f)

    return markov_exp, dice

def test_empirical(layout, circle=False, expectation=None, policy=None, name="empiric", write_file=True) :
    """runs the empirical tests and prints in the output file"""
    empiric_exp, dice = SaL.test_empirically(layout, circle, expectation, policy, nb_iterations, verbose == 2)
    # write
    if write_file :
        print("empiric", file=f)
        print(f"    name        : {name}", file=f)
        print(f"    layout      : {layout}", file=f)
        print(f"    circle      : {circle}", file=f)
        print(f"    iterations  : {int(nb_iterations)}", file=f)
        print(f"    dice        : {dice}", file=f)
        print(f"    expectation : {empiric_exp:>7.4f}", file=f)
        print("", file=f)

    return empiric_exp, dice


def compare_models(layout, circle=False) :
    markov_exp, markov_dice = test_markov(layout, circle)
    _, _ = test_empirical(layout, circle, markov_exp, markov_dice)


def compare_policies(policies, layout, circle=False) :
    """compare different policies with one another and with the optimal policy"""
    _, optimal_policy = test_markov(layout, circle, write_file=False)
    policies.append(("optimal", optimal_policy))

    # test for each policy
    for policy in policies:
        # if a policy is a tuple of name, dice
        if len(policy) == 2 :
            name, dice = policy
        # if a policy is only a list of dice
        elif len(policy) == 15 :
            name, dice = None, policy
        
        empiric_exp, _ = test_empirical(layout, circle, policy=dice, name=name)
        if verbose >= 1 : 
            print(f"\nName : {name}")
            print(f"    expectation : {empiric_exp:>7.4f}")
            print(f"    policy : {list(dice)}")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    #compare_models(layout_custom2, False)
    #compare_models(layout_custom2, True)
    policies = strategies.get_policies(layout_custom2, True)
    compare_policies(policies, layout_custom2, True)

    f.close()


# -----------------------------------------------------------------------------
# OLD STUFF
# -----------------------------------------------------------------------------
# # SaL.test_markovDecision(layout_ordinary, False, "ORDINARY")
# # result = SaL.test_markovDecision(layout_prison, False, "PRISON")
# # SaL.test_empirically(layout_prison, False, *result)
# # SaL.test_markovDecision(layout_penalty, False, "PENALTY")
# # SaL.test_markovDecision(layout_random, False, "RANDOM")
# # SaL.test_markovDecision(layout_custom1, False, "CUSTOM1")

# result = SaL.test_markovDecision(layout_custom2, False, "CUSTOM2", verbose=True)
# SaL.test_empirically(layout_custom2, False, *result, nb_iter=1e5, verbose=True)

# result = SaL.test_markovDecision(layout_custom2, True, "CUSTOM2", verbose=True)
# SaL.test_empirically(layout_custom2, True, *result, nb_iter=1e5, verbose=True)

# # SaL.test_empirically(layout_custom2, True, policy=np.array([SECURITY for _ in range(15)]))
# # SaL.test_empirically(layout_custom2, True, policy=np.array([NORMAL for _ in range(15)]))
# # SaL.test_empirically(layout_custom2, True, policy=np.array([RISKY for _ in range(15)]))
# # SaL.test_empirically(layout_custom2, True, policy=np.array([np.random.randint(RISKY) + 1 for _ in range(15)]))
# # SaL.test_empirically(layout_custom2, True, policy=None)