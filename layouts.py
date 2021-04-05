from snakes_and_ladders import ORDINARY, RESTART, PENALTY, PRISON, GAMBLE
import numpy as np

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

# only PRISON (return to starting square)
layout_restart = np.ones(15) * RESTART
layout_restart[[0, -1]] = ORDINARY   # start and goal squares must be ordinary

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

test_layouts = [layout_ordinary, layout_penalty, layout_prison, layout_gamble, layout_restart, layout_custom1, layout_custom2]
test_layouts_val = [layout_ordinary, layout_penalty]