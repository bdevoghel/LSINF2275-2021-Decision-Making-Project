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

# layout 1 : only ordinary squares, no traps
test_layout1 = np.array([0 for _ in range(15)])

# layout 2 : random initialized traps
test_layout2 = np.random.randint(low=0, high=5, size=15)
test_layout2[[0, -1]] = 0  # start and goal squares must be ordinary


class Die:
    def __init__(self, die_type):
        """Die type must be SECURITY or NORMAL or RISKY"""
        self.type = die_type

    def roll(self, times=1):
        """Returns number of steps to advance and tells if trap triggers or not"""
        nb_steps = np.random.randint(low=0, high=self.type + 1, size=times)
        does_trigger = [False if rand > self.type - 1.5 else True for rand in np.random.rand(times)]
        return nb_steps, does_trigger


class Board:
    def __init__(self, layout, circle):
        self.layout = layout
        self.circle = circle

    def compute_landing_proba(self, start_position, selected_die: Die):
        """Starting position must be [0, 14]"""
        landing_proba = np.random.uniform(size=len(self.layout))  # TODO
        return landing_proba


def markovDecision(layout, circle):
    board = Board(layout, circle)

    # TODO
    expec = np.random.randint(low=0, high=5, size=14)
    dice = np.random.randint(low=0, high=5, size=14)

    return [expec, dice]


def test_markovDecision():
    layout = test_layout1
    circle = False

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
