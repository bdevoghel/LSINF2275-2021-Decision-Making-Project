import snakes_and_ladders as SaL
from snakes_and_ladders import SECURITY, NORMAL, RISKY
from snakes_and_ladders import ORDINARY, RESTART, PENALTY, PRISON, GAMBLE
import strategies
import numpy as np
import threading
from layouts import *

# -----------------------------------------------------------------------------
# Testing constants
# -----------------------------------------------------------------------------

# print information to standard output 
#   0 for no verbose,
#   2 for max verbose
verbose = 1
# number of iterations for empirical tests
nb_iterations = 1e6
# filename to write results
filename = "results_val_layout_5.txt"
# open file in write mode
f = open(filename, 'w')

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

def test_markov(layout, circle=False, name="markov", write_file=True) :
    """runs the markov decision tests, prints in the output file and returns 
       the optimal dice to use"""
    markov_exp, dice = SaL.test_markovDecision(layout, circle, verbose == 2)
    # write
    if write_file :
        print("new markov", file=f)
        print(f"    name        : {name}", file=f)
        print(f"    layout      : {' '.join(map(str, layout))}", file=f)
        print(f"    circle      : {circle}", file=f)
        print(f"    dice        : {' '.join(map(str, dice))}", file=f)
        print(f"    expectation : {' '.join(map(lambda x: format(x, '>7.4f'), markov_exp))}", file=f)
        print("", file=f)

    return markov_exp, dice

def test_empirical(layout, circle=False, expectation=None, policy=None, name="empiric", write_file=True) :
    """runs the empirical tests and prints in the output file"""
    empiric_exp, dice = SaL.test_empirically(layout, circle, expectation, policy, nb_iterations, verbose == 2)
    # write
    if write_file :
        print("new empiric", file=f)
        print(f"    name        : {name}", file=f)
        print(f"    layout      : {' '.join(map(str, layout))}", file=f)
        print(f"    circle      : {circle}", file=f)
        print(f"    iterations  : {int(nb_iterations)}", file=f)
        print(f"    dice        : {' '.join(map(str, dice))}" if dice is not None else "", file=f)
        print(f"    expectation : {' '.join(map(lambda x: format(x, '>7.4f'), empiric_exp))}", file=f)
        print("", file=f)

    if verbose >= 1:
        print(f"\nName : {name}")
        print(f"    expectation : {' '.join(map(lambda x: format(x, '>7.4f'), empiric_exp))}")
        print(f"    policy : {list(dice)}" if dice is not None else f"    policy : pure random")

    return empiric_exp, dice


def compare_models(layout, circle=False) :
    markov_exp, markov_dice = test_markov(layout, circle)
    _, _ = test_empirical(layout, circle, markov_exp, markov_dice)


def compare_policies(policies, layout, circle=False, add_optimal = True, add_pure_random=False) :
    """compare different policies with one another and with the optimal policy"""
    _, optimal_policy = test_markov(layout, circle, write_file=add_optimal)
    if add_optimal : policies.append(("optimal", optimal_policy))
    if add_pure_random : policies.append(("pure random", None))

    threads = []

    # test for each policy
    for policy in policies :
        # if a policy is a tuple of name, dice
        if len(policy) == 2 : name, dice = policy
        # if a policy is only a list of dice
        elif len(policy) == 15 : name, dice = None, policy
        threads.append(threading.Thread(target=test_empirical, args=(layout, circle), kwargs={'policy': dice, 'name':name}))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # for layout in test_layouts[:2] :
    layout = test_layouts[5]
    for circle in [True, False] :
        policies = strategies.get_policies(layout, circle)
        compare_policies(policies, layout, circle, add_optimal=True, add_pure_random=True)

    f.close()