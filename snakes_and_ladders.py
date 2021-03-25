import numpy as np

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

SECURITY = 1
NORMAL = 2
RISKY = 3

ORDINARY = 0
RESTART = 1
PENALTY = 2
PRISON = 3
GAMBLE = 4

# only ORDINARY squares, no traps
layout_ORDINARY = np.array([ORDINARY for _ in range(15)])

# only PENALTY (go back 3) traps
layout_PENALTY = np.ones(15) * PENALTY
layout_PENALTY[[0, -1]] = 0  # start and goal squares must be ordinary

# only PRISON (stay one turn) traps
layout_PRISON = np.ones(15) * PRISON
layout_PRISON[[0, -1]] = 0  # start and goal squares must be ordinary

# random initialized traps
layout_random = np.random.randint(low=0, high=5, size=15)
layout_random[[0, -1]] = 0  # start and goal squares must be ordinary

# custom layout
layout_custom1 = np.array([ORDINARY for _ in range(15)])
layout_custom1[8] = RESTART
layout_custom1[7] = GAMBLE
layout_custom1[12] = PENALTY

# custom layout
layout_custom2 = np.array([ORDINARY for _ in range(15)])
layout_custom2[1] = GAMBLE
layout_custom2[2] = RESTART
layout_custom2[3] = GAMBLE
layout_custom2[4] = PENALTY
layout_custom2[6] = PRISON
layout_custom2[8] = RESTART
layout_custom2[10] = PRISON
layout_custom2[11] = PENALTY
layout_custom2[12] = RESTART
layout_custom2[13] = GAMBLE


class Die:
    def __init__(self, die_type):
        """Die type must be SECURITY or NORMAL or RISKY"""
        self.type = die_type
        self.possible_steps = list(range(self.type + 1))
        self.steps_proba = 1 / len(self.possible_steps)
        self.possible_trap_trigger = [True] if self.type == 3 else ([False] if self.type == 1 else [True, False])
        self.trap_trigger_proba = 1 if self.type == 3 else (0 if self.type == 1 else 0.5)

    def roll(self, times=1):
        """Returns number of steps to advance and tells if trap triggers or not"""
        nb_steps = np.random.randint(low=0, high=self.type + 1, size=times)
        does_trigger = [False if rand > self.type - 1.5 else True for rand in np.random.rand(times)]
        return (nb_steps, does_trigger) if times > 1 else (nb_steps[0], does_trigger[0])

    def get_all_possible_roll_combinations(self):
        return [item for sublist in [[(steps, trigger) for steps in self.possible_steps] for trigger in self.possible_trap_trigger] for item in sublist]

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
        self.P = {die: np.array(self.compute_transition_matrix(die)) for die in self.dice}

        self.prison_extra_cost = {die: self.compute_prison_extra_cost(die) for die in self.dice}

    def __repr__(self):
        return f"[Board:{list(self.layout)}-Circle:{self.circle}]"

    def compute_transition_matrix(self, die: Die):
        """Computes transition matrix in canonical form for corresponding die"""
        tm = []
        for square in range(15):
            tm.append(self.compute_landing_proba(square, die))
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

    def apply_penalty(self, pos):
        if 10 <= pos <= 12:
            return pos - 7 - 3
        else:
            return max(0, pos - 3)

    def compute_landing_proba(self, start_position, die: Die):
        """
        Returns vector with landing probabilities on each square
        :param start_position: must be in [0, 14]
        :param die: selected die
        """
        landing_proba = np.zeros(len(self.layout))
        for step in die.possible_steps:
            for new_pos, budget in self.accessible_squares(start_position, step, die.steps_proba):
                square_type = self.layout[new_pos]
                if square_type == ORDINARY:
                    landing_proba[new_pos] += budget
                elif square_type == RESTART:
                    landing_proba[0] += die.trap_trigger_proba * budget
                    landing_proba[new_pos] += (1 - die.trap_trigger_proba) * budget
                elif square_type == PENALTY:
                    landing_proba[self.apply_penalty(new_pos)] += die.trap_trigger_proba * budget
                    landing_proba[new_pos] += (1 - die.trap_trigger_proba) * budget
                elif square_type == PRISON:
                    landing_proba[new_pos] += budget  # TODO modify ??
                elif square_type == GAMBLE:
                    landing_proba[:] += budget * die.trap_trigger_proba / len(self.layout)
                    landing_proba[new_pos] += budget * (1 - die.trap_trigger_proba)

        return landing_proba

    def compute_prison_extra_cost(self, die):
        extra_cost = np.zeros(len(self.layout))
        if die.type == NORMAL:
            extra_cost[np.where(self.layout == PRISON)] += 0.5
        elif die.type == RISKY:
            extra_cost[np.where(self.layout == PRISON)] += 1
        return extra_cost

    def roll_dice(self, pos, die_type):
        nb_steps, does_trigger = Die(die_type).roll()
        accessible_squares = [int(x[0]) for x in self.accessible_squares(pos, nb_steps)]
        new_pos = np.random.choice(accessible_squares)
        in_prison = False

        square_type = self.layout[new_pos]
        if does_trigger:
            if square_type == RESTART:
                new_pos = 0
            elif square_type == PENALTY:
                new_pos = self.apply_penalty(new_pos)
            elif square_type == PRISON:
                in_prison = True
            elif square_type == GAMBLE:
                new_pos = np.random.choice(range(len(self.layout)))
        return new_pos, in_prison


def markovDecision(layout: np.ndarray, circle: bool):
    """
    :param layout: vector representing the layout of the game,
                   containing 15 values representing the 15 squares, values in [0, 4]
    :param circle: indicates if the player must land exactly on the final square to win or still wins by overstepping
    """

    board = Board(layout, circle)

    policy = np.zeros(len(board.layout) - 1, dtype=int)
    costs = np.ones(len(board.layout))
    costs[-1] = 0  # goal

    # VALUE ITERATION ALGORITHM
    eps = 1e-6
    delta = eps + 1

    # Looping until expected number of turns has converged for every state
    while delta > eps:
        # Keeping track of last iteration's expected number of turns for each state
        prev_cost = costs.copy()
        for state in range(len(policy)):
            cost_per_die = {}
            for die in board.dice:
                action_cost = 1  # cost to throw the dice is 1 (1 turn)

                # Adding the cost to expectation of next turns costs
                # (+ adding the extra cost of falling on a prison trap if the normal or risky die is used)
                cost_per_die[die] = action_cost + board.prison_extra_cost[die][state] + np.dot(board.P[die][state], costs)

            # computing the best action for this state
            cheapest_die = min(cost_per_die, key=cost_per_die.get)
            costs[state] = cost_per_die[cheapest_die]
            policy[state] = cheapest_die.type

        # max difference between previous and current estimation of expected number of turns
        delta = np.max(np.abs(costs - prev_cost))

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
    nb_rolls = []
    print(f"Simulating {int(nb_iter)} games with following"
          f"\n   - layout : {layout}, circle={circle}"
          f"\n   - policy : {policy if policy is not None else 'random die selection'}")
    if verbose:
        print(f" - Progress : {0:>2}%", end='')
    for i in range(int(nb_iter)):
        if verbose:
            if i % (nb_iter / 20) == 0:
                print(f"\b\b\b{int(100 * i / nb_iter):>2}%", end='')
            if i == nb_iter - 1:
                print("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b", end='')

        pos = 0
        nb_rolls.append(0)
        while pos != len(layout) - 1:
            pos, in_prison = board.roll_dice(pos, policy[pos] if policy is not None else np.random.randint(RISKY) + 1)
            nb_rolls[-1] += 1 if not in_prison else 2

    print(f"Expectation results : " +
          (f"\n   - Optimal (MDP) : {expectation[0]:>7.4f}" if expectation is not None else "") +
          f"\n   - Empiric       : {np.mean(nb_rolls):>7.4f} "
          f"| σ = {np.std(nb_rolls):>7.4f} "
          f"| [{np.min(nb_rolls)}, {np.max(nb_rolls)}]"
          f"\n")


if __name__ == '__main__':
    # test_markovDecision(layout_ORDINARY, False, "ORDINARY")
    result = test_markovDecision(layout_PRISON, False, "PRISON")
    test_empirically(layout_PRISON, False, *result)
    # test_markovDecision(layout_PENALTY, False, "PENALTY")
    # test_markovDecision(layout_random, False, "RANDOM")
    # test_markovDecision(layout_custom1, False, "CUSTOM1")

    # result = test_markovDecision(layout_custom2, False, "CUSTOM2")
    # test_empirically(layout_custom2, False, *result)

    # result = test_markovDecision(layout_custom2, True, "CUSTOM2")
    # test_empirically(layout_custom2, True, *result)

    # test_empirically(layout_custom2, True, policy=np.array([SECURITY for _ in range(15)]))
    # test_empirically(layout_custom2, True, policy=np.array([NORMAL for _ in range(15)]))
    # test_empirically(layout_custom2, True, policy=np.array([RISKY for _ in range(15)]))
    # test_empirically(layout_custom2, True, policy=np.array([np.random.randint(RISKY) + 1 for _ in range(15)]))
    # test_empirically(layout_custom2, True, policy=None)
