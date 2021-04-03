import numpy as np
import itertools
import time

"""
SQUARES : 
- 0 : ordinary
- 1 : restart trap : go back to square 1
- 2 : penalty trap : go back 3 steps
- 3 : prison trap  : skip next turn
- 4 : gable trap   : random teleportation anywhere with uniform probability

DICE :
- security : 0/1     - invincible
- normal   : 0/1/2   - 50% chance to trigger trap
- risky    : 0/1/2/3 - 100% chance to trigger trap
"""

SECURITY, NORMAL, RISKY = 1, 2, 3
ORDINARY, RESTART, PENALTY, PRISON, GAMBLE = 0, 1, 2, 3, 4

class Die:
    def __init__(self, die_type):
        """Die type must be SECURITY or NORMAL or RISKY"""
        self.type = die_type
        self.possible_steps = list(range(self.type + 1))
        self.steps_proba = 1 / len(self.possible_steps)
        self.possible_trap_trigger = [True] if self.type == RISKY else ([False] if self.type == SECURITY else [True, False])
        self.trap_trigger_proba = 1 if self.type == RISKY else (0 if self.type == SECURITY else 0.5)

    def roll(self, times=1):
        """Returns number of steps to advance and tells if trap triggers or not"""
        nb_steps = np.random.randint(low=0, high=self.type + 1, size=times)
        does_trigger = [False if rand > self.type - 1.5 else True for rand in np.random.rand(times)]
        return (nb_steps, does_trigger) if times > 1 else (nb_steps[0], does_trigger[0])

    def get_all_possible_roll_combinations(self):
        return list(itertools.product(self.possible_steps, self.possible_trap_trigger))

    def __hash__(self):
        return self.type

    def __repr__(self):
        return f"[Die:{self.type}]"


class Board:
    def __init__(self, layout, circle):
        self.dice = [Die(die_type=dt) for dt in [SECURITY, NORMAL, RISKY]]
        self.layout = layout
        self.circle = circle

        # transition matrices
        self.P = {die: np.array(self.compute_transition_matrix(die, traps=True)) for die in self.dice}
        self.P_no_traps = {die: np.array(self.compute_transition_matrix(die, traps=False)) for die in self.dice}

    def __repr__(self):
        return f"[Board:{list(self.layout)}-Circle:{self.circle}]"

    def compute_transition_matrix(self, die: Die, traps=True):
        """Computes transition matrix in canonical form for corresponding die"""
        tm = np.zeros((len(self.layout), len(self.layout)))
        for square in range(15):
            tm[square, :] = self.compute_landing_proba(square, die, traps=traps)
        return tm

    def accessible_squares(self, pos, delta, budget=1.):
        """
        Returns a list of tuples of the accessible squares with the budget distributed
        according to the squares' respective landing probabilities
        """
        if pos < 2:
            return [(pos+delta, budget)]
        elif pos == 2:
            if delta == 0:
                return [(pos+delta, budget)]
            else:
                return [(pos+delta, budget/2), (pos+7+delta, budget/2)]  # lane split
        elif pos < 7:
            return [(pos+delta, budget)]
        elif pos == 7:
            if delta == 3:
                return [(14, budget)]
            else:
                return [(pos + delta, budget)]
        elif pos == 8:
            if delta < 2:
                return [(pos + delta, budget)]
            else:
                if self.circle:
                    if delta == 2:
                        return [(14, budget)]
                    else:  # delta == 3
                        return [(0, budget)]
                else:
                    return [(14, budget)]
        elif pos == 9:
            if delta < 1:
                return [(pos + delta, budget)]
            else:
                if self.circle:
                    if delta == 1:
                        return [(14, budget)]
                    else:  # delta == 2 or 3
                        return [(delta - 2, budget)]
                else:
                    return [(14, budget)]
        elif pos < 12:
            return [(pos+delta, budget)]
        elif pos == 12:
            if delta < 3:
                return [(pos + delta, budget)]
            else:
                if self.circle:
                    return [(delta - 3, budget)]
                else:
                    return [(14, budget)]
        elif pos == 13:
            if delta < 2:
                return [(pos + delta, budget)]
            else:
                if self.circle:
                    return [(delta - 2, budget)]
                else:
                    return [(14, budget)]
        elif pos == 14:
            return [(pos, budget)]
        else:
            print("ERROR - Not possible to compute accessible squares")

    def apply_delta(self, positions, deltas):
        new_positions = -np.ones(len(positions), dtype=int)

        # before intersection
        idx = positions < 2
        new_positions[idx] = positions[idx] + deltas[idx]

        # at intersection but not moving
        idx = np.logical_and(positions == 2, deltas == 0)
        new_positions[idx] = positions[idx]

        # at intersection and moving
        idx = np.logical_and(positions == 2, deltas != 0)
        new_positions[idx] = np.choose(np.random.randint(0, 2, sum(idx)), np.array([positions[idx] + deltas[idx], positions[idx] + deltas[idx] + 7]))

        # --- SLOW LANE ---
        # 'jump' is the jump from 9 to 14, when slow and fast lanes meet back

        # before jump
        idx = np.logical_and(positions > 2, positions < 7)
        new_positions[idx] = positions[idx] + deltas[idx]

        # from 7 with no jump
        idx = np.logical_and(positions == 7, deltas < 3)
        new_positions[idx] = positions[idx] + deltas[idx]
        
        # from 7 with jump
        idx = np.logical_and(positions == 7, deltas == 3)
        new_positions[idx] = 14
        
        # from 8 with no jump
        idx = np.logical_and(positions == 8, deltas < 2)
        new_positions[idx] = positions[idx] + deltas[idx]

        # from 8 with jump
        idx = np.logical_and(positions == 8, deltas == 2)
        new_positions[idx] = 14

        # from 8 with overshoot
        idx = np.logical_and(positions == 8, deltas > 2)
        new_positions[idx] = deltas[idx] - 3 if self.circle else 14
        
        # from 9 without jump
        idx = np.logical_and(positions == 9, deltas < 1)
        new_positions[idx] = positions[idx] + deltas[idx]
        
        # from 9 with jump
        idx = np.logical_and(positions == 9, deltas == 1)
        new_positions[idx] = 14
        
        # from 9 with overshoot
        idx = np.logical_and(positions == 9, deltas > 1)
        new_positions[idx] = deltas[idx] - 2 if self.circle else 14
        
        # --- FAST LANE ---
        # before possible overshoot
        idx = np.logical_and(positions > 9, positions < 12)
        new_positions[idx] = positions[idx] + deltas[idx]
        
        # from 12 without overshoot
        idx = np.logical_and(positions == 12, deltas < 3)
        new_positions[idx] = positions[idx] + deltas[idx]
        
        # from 12 with overshoot
        idx = np.logical_and(positions == 12, deltas >= 3)
        new_positions[idx] = deltas[idx] - 3 if self.circle else 14
        
        # from 13 without overshoot
        idx = np.logical_and(positions == 13, deltas < 2)
        new_positions[idx] = positions[idx] + deltas[idx]
        
        # from 13 with overshoot
        idx = np.logical_and(positions == 13, deltas >= 2)
        new_positions[idx] = deltas[idx] - 2 if self.circle else 14
        
        # if already at the end
        idx = positions == 14
        new_positions[idx] = 14
        
        return new_positions

    def apply_penalty(self, pos):
        if 10 <= pos <= 12:
            return pos - 7 - 3
        else:
            return max(0, pos - 3)

    def apply_n_penalty(self, states):
        new_states = -np.ones(len(states), dtype=int)

        # going back to before intersection
        idx = np.logical_and(states >= 10, states <= 12)
        new_states[idx] = states[idx] - 7 - 3

        # normal
        idx = np.logical_or(states < 10, states > 12)
        new_states[idx] = np.maximum(np.zeros(sum(idx), dtype=int), states[idx] - 3)

        return new_states

    def compute_landing_proba(self, start_position, die: Die, traps=True):
        """
        Returns vector with landing probabilities on each square
        :param start_position: must be in [0, 14]
        :param die: selected die
        :param traps: if False we ignore all traps on the board
        """
        landing_proba = np.zeros(len(self.layout))
        for d in die.possible_steps:
            for state, budget in self.accessible_squares(start_position, d, die.steps_proba):
                if traps:
                    if self.layout[state] == ORDINARY:
                        landing_proba[state] += budget
                    elif self.layout[state] == RESTART:
                        landing_proba[0] += budget * die.trap_trigger_proba
                        landing_proba[state] += budget * (1 - die.trap_trigger_proba)
                    elif self.layout[state] == PENALTY:
                        landing_proba[self.apply_penalty(state)] += budget * die.trap_trigger_proba
                        landing_proba[state] += budget * (1 - die.trap_trigger_proba)
                    elif self.layout[state] == PRISON:
                        landing_proba[state] += budget
                    elif self.layout[state] == GAMBLE:
                        landing_proba += budget * die.trap_trigger_proba / len(self.layout)
                        landing_proba[state] += budget * (1 - die.trap_trigger_proba)
                else:
                    landing_proba[state] += budget

        assert np.abs(np.sum(landing_proba)-1) < 1e-15, "Sum of probabilities must be equal to 1"
        return landing_proba

    def apply_traps(self, states, does_trigger):
        new_states = -np.ones(len(states), dtype=int)
        extra_costs = np.zeros(len(states), dtype=float)

        square_types = self.layout[states]

        idx = np.logical_or(square_types == ORDINARY, np.logical_not(does_trigger))
        new_states[idx] = states[idx]
        
        idx = np.logical_and(square_types == RESTART, does_trigger)
        new_states[idx] = 0

        idx = np.logical_and(square_types == PENALTY, does_trigger)
        new_states[idx] = self.apply_n_penalty(states[idx])

        idx = np.logical_and(square_types == PRISON, does_trigger)
        new_states[idx] = states[idx]
        extra_costs[idx] = 1

        idx = np.logical_and(square_types == GAMBLE, does_trigger)
        new_states[idx] = np.random.choice(range(len(self.layout)), len(states[idx]))

        return new_states, extra_costs

    def roll_n_dice(self, positions, die_types):
        nb_steps = np.zeros(len(positions))

        security_throws = np.where(die_types == SECURITY)[0]
        normal_throws   = np.where(die_types == NORMAL)[0]
        risky_throws    = np.where(die_types == RISKY)[0]

        nb_steps[security_throws] = np.random.choice(self.dice[SECURITY - 1].possible_steps, len(security_throws), True)
        nb_steps[normal_throws]   = np.random.choice(self.dice[NORMAL   - 1].possible_steps, len(normal_throws),   True)
        nb_steps[risky_throws]    = np.random.choice(self.dice[RISKY    - 1].possible_steps, len(risky_throws),    True)

        return nb_steps

    def does_trigger_trap(self, die_types):
        does_trigger = np.zeros(len(die_types), dtype=bool)
        does_trigger[die_types == RISKY] = True
        does_trigger[die_types == NORMAL] = np.random.choice([True, False], sum(die_types == NORMAL), True)
        return does_trigger


def markovDecision(layout: np.ndarray, circle: bool):
    """
    :param layout: vector representing the layout of the game,
                   containing 15 values representing the 15 squares, values in [0, 4]
    :param circle: indicates if the player must land exactly on the final square to win or still wins by overstepping
    """

    board = Board(layout, circle)

    policy = np.zeros(len(board.layout) - 1, dtype=int)
    costs = np.zeros(len(board.layout), dtype=float)
    # costs[-1] = 0  # goal

    # VALUE ITERATION ALGORITHM
    eps = 1e-6
    delta = eps + 1.

    prisons = np.where(board.layout == PRISON)[0]

    # Looping until expected number of turns has converged for every state
    while delta > eps:
        # Keeping track of last iteration's expected number of turns for each state
        prev_costs = costs.copy()
        for state in range(len(policy)):
            cost_per_die = {}
            for die in board.dice:
                # cost to throw the dice is 1 (1 turn) + expected extra cost of falling on a prison with this die
                action_cost = 1. + np.sum(board.P_no_traps[die][state][prisons] * die.trap_trigger_proba)

                # Adding the cost to expectation of next turn costs
                cost_per_die[die] = action_cost + board.P[die][state] @ costs

            # computing the best action for this state
            cheapest_die = min(cost_per_die, key=cost_per_die.get)
            costs[state] = cost_per_die[cheapest_die]
            policy[state] = cheapest_die.type

        # max difference between previous and current estimation of expected number of turns
        delta = np.max(np.abs(costs - prev_costs))

    return [costs[:-1], policy]


def test_markovDecision(layout, circle, name=""):
    assert isinstance(layout, np.ndarray) and len(layout) == 15, f"Input layout is not a ndarray or is not of length 15"
    assert isinstance(circle, bool), f"Input circle is not a bool"

    result = markovDecision(layout, circle)

    assert isinstance(result, list) and len(result) == 2, \
        f"Result is not in correct format (should be a list like [expectation, dice])\n\nRESULT : {result}"
    expectation, dice = result
    assert isinstance(expectation, np.ndarray) and len(expectation) == 14, \
        f"Output expected cost is not a ndarray or is not of length 14\n\nEXPECTATION : {expectation}"
    assert isinstance(dice, np.ndarray) and len(dice) == 14, \
        f"Output dice is not a ndarray or is not of length 14\n\nDICE       : {dice}"

    _format = "{:<7}" * 14
    print(f"\nSuccess {name} - Circle: {circle}"
          f"\n              {_format.format(*list(range(1, 15)))}"
          f"\nEXPECTATION : {_format.format(*np.around(expectation, 2))}"
          f"\nDICE        : {_format.format(*np.around(dice, 2))}"
          f"\n")

    return result


def test_empirically(layout, circle, expectation=None, policy=None, nb_iter=1e4, verbose=True):
    board = Board(layout, circle)
    nb_rolls = np.zeros(int(nb_iter))
    states = np.zeros(int(nb_iter), dtype=int)
    not_done = np.ones(int(nb_iter), dtype=bool)

    policy = policy if policy is not None else np.random.randint(SECURITY, RISKY+1, len(board.layout)-1)

    print(f"Simulating {int(nb_iter)} games with following"
          f"\n   - layout : {layout}, circle={circle}"
          f"\n   - policy : {policy if policy is not None else 'random die selection'}")

    while np.sum(not_done) != 0:

        states_left = states[not_done]

        trap_trigger = board.does_trigger_trap(policy[states_left])

        nb_steps = board.roll_n_dice(states_left, policy[states_left])
        new_states = board.apply_delta(states_left, nb_steps)
        new_states, extra_costs = board.apply_traps(new_states, trap_trigger)

        nb_rolls[not_done] += 1. + extra_costs

        states[not_done] = new_states
        not_done[states == len(board.layout) - 1] = False

    print(f"Expectation results : " +
          (f"\n   - Optimal (MDP) : {expectation[0]:>7.4f}" if expectation is not None else "") +
          f"\n   - Empiric       : {np.mean(nb_rolls):>7.4f} "
          f"| σ = {np.std(nb_rolls):>7.4f} "
          f"| [{np.min(nb_rolls)}, {np.max(nb_rolls)}]"
          f"\n")
