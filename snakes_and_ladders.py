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

# layout 1 : only ordinary squares, no traps
test_layout1 = np.array([0 for _ in range(15)])

# layout 2 : random initialized traps
test_layout2 = np.random.randint(low=0, high=5, size=15)
test_layout2[[0, -1]] = 0  # start and goal squares must be ordinary

# layout 3 : perso
test_layout3 = np.array([ORDINARY for _ in range(15)])
test_layout3[8] = RESTART
test_layout3[7] = GAMBLE
test_layout3[12] = PENALTY


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
        return nb_steps, does_trigger

    def get_all_possible_roll_combinations(self):
        return [item for sublist in [[(steps, trigger) for steps in self.possible_steps] for trigger in self.possible_trap_trigger] for item in sublist]

    def __hash__(self):
        return self.type


class Board:
    def __init__(self, layout, circle):
        self.dice = [Die(die_type=dt) for dt in [SECURITY, NORMAL, RISKY]]
        self.layout = layout
        self.circle = circle

        # transition matrices
        self.P = {die:np.array(self.compute_transition_matrix(die)) for die in self.dice}

    def compute_transition_matrix(self, die: Die):
        """Computes transition matrix in canonical form for corresponding die"""
        tm = []
        for square in range(15):
            tm.append(self.compute_landing_proba(square, die))
        return tm

    def accessible_square(self, pos, delta, budget):
        if pos < 2:
            return [(pos+delta, budget)]
        elif pos == 2:
            if delta == 0:
                return [(pos+delta, budget)]
            else:
                return [(pos+delta, budget/2), (pos+7+delta, budget/2)] # lane split
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
            return pos-7 - 3
        else:
            return max(0, pos - 3)

    def compute_landing_proba(self, start_position, die: Die):
        """
        Returns vector with landing probabilities on each square
        :param start_position: must be in [0, 14]
        """
        landing_proba = np.zeros(len(self.layout))
        for step in die.possible_steps:
            for new_pos, budget in self.accessible_square(start_position, step, die.steps_proba):
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
                    pass  # TODO
                elif square_type == GAMBLE:
                    landing_proba[range(len(self.layout))] += 1 / len(self.layout)

        return landing_proba


def markovDecision(layout: np.ndarray, circle: bool):
    """
    :param layout: vector representing the layout of the game,
                   containing 15 values representing the 15 squares, values in [0, 4]
    :param circle: indicates if the player must land exactly on the final square to win or still wins by overstepping
    """
    board = Board(layout, circle)
    actions = np.zeros(len(board.layout) - 1)
    costs = - np.ones(len(board.layout))
    costs[-1] = 0

    # Value-iteration algo
    def rec(pos, previous_die, budget):
        if costs[pos] != -1:
            return budget * costs[pos]
        elif budget < 1e-12:
            return 0
        cost = 2 if pos == PRISON else 1
        action_costs = {}
        for die in board.dice:
            total_cost = 0
            for state, prob in enumerate(board.P[die][pos]):
                if prob > 0 and state != pos:
                    total_cost += rec(state, die, budget*prob)
            action_costs[die] = cost + total_cost
            action_costs[die] /= 1 - board.P[die][pos, pos]
        min_action = min(action_costs, key=action_costs.get)
        costs[pos] = action_costs[min_action]
        actions[pos] = min_action.type
        return costs[pos] * budget

    for i in range(len(actions) - 1):
        costs[i] = rec(i, None, 1)

    return [costs[:-1], actions]


def test_markovDecision():
    layout = test_layout3
    circle = True

    assert isinstance(layout, np.ndarray) and len(layout) == 15, f"Input layout is not a ndarray or is not of length 15"
    assert isinstance(circle, bool), f"Input circle is not a bool"

    result = markovDecision(layout, circle)

    assert isinstance(result, list) and len(result) == 2, \
        f"Result is not in correct format (should be a list like [expec, dice])\n\nRESULT : {result}"
    expec, dice = result
    assert isinstance(expec, np.ndarray) and len(expec) == 14, \
        f"Output expected cost is not a ndarray or is not of length 14\n\nEXPEC : {expec}"
    assert isinstance(dice, np.ndarray) and len(dice) == 14, \
        f"Output dice is not a ndarray or is not of length 14\n\nDICE : {dice}"

    print(f"Success\n\nEXPEC : {expec}\nDICE  : {dice}")


if __name__ == '__main__':
    test_markovDecision()

    # Bonus : implement empirical tests to show convergence towards obtained results
