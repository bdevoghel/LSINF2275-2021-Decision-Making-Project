from matplotlib import pyplot as plt
from layouts import *

filename = "results_gus.txt"
layout_match = [
    ('All ordinary', layout_ordinary),
    ('All penalty', layout_penalty),
    ('All prison', layout_prison),
    ('All gamble', layout_gamble),
    ('All restart', layout_restart),
    ('Custom 1', layout_custom1),
    ('Custom 2', layout_custom2)
]

# class to store the result
class Result :
    def __init__(self, type):
        self.type = type
        self.policy_name = None
        self.policy_dice = None
        self.layout      = None
        self.circle      = None
        self.iterations  = None
        self.expectation = None

# read file
results = []
with open(filename, 'r') as f :
    for line in f.readlines() :
        line = line.split()
        if len(line) == 0 :
            continue

        elif line[0] == "new" :
            results.append(Result(line[-1]))
        
        elif line[0] == "name" :
            results[-1].policy_name = line[-1]
        
        elif line[0] == "layout" :
            results[-1].layout = list(map(int, map(float, line[2:])))
        
        elif line[0] == "circle" :
            results[-1].circle = bool(line[-1])

        elif line[0] == "iterations" :
            results[-1].iterations = int(line[-1])

        elif line[0] == "dice" :
            results[-1].policy_dice = list(map(int, map(float, line[2:])))

        elif line[0] == "expectation" :
            results[-1].expectation = list(map(float, line[2:]))
        
# group by layout
grouped = []
for result in results :
    # keep track if a result needs a new group
    new_group = True
    # check all existing groups
    for i, group in enumerate(grouped) :
        if result.layout == group[0].layout :
            grouped[i].append(result)
            new_group = False
            break
    if new_group :
        grouped.append([result])
results = grouped

# plotting
# for each layout
for group in grouped :
    # create one plot
    plt.figure(figsize=(5,5))
    # get layout name
    title = "No name found for this layout"
    for layout_name, layout_tiles in layout_match :
        if list(group[0].layout) == list(layout_tiles) :
            title = layout_name
            break
    # get names and expectation
    names = []
    expec = []
    for r in group :
        names.append(r.policy_name)
        expec.append(r.expectation[0])
    # create bars
    plt.bar(names, expec)
    # labels and titles
    plt.title(f"{title}")
    plt.xlabel("policy")
    plt.ylabel("expectation")
    # save figure
    plt.savefig(f"plots/{title.replace(' ','')}", bbox_inches='tight')

    # plotting empiric/theoric graph
    # get names and expectation
    plt.figure(figsize=(7, 5))
    names = []
    expec = []

    X_axis = np.arange(1, len(r.layout))

    for r in group:
        if r.policy_name == 'markov':
            plt.bar(X_axis - 0.2, r.expectation, 0.4, label='Theoretical')
        elif r.policy_name == 'optimal':
            plt.bar(X_axis + 0.2, r.expectation, 0.4, label='Empirical')

    # labels and titles
    title = f"{title} Empirical versus Theoretical"
    plt.title(f"{title}")
    plt.xlabel("State")
    plt.ylabel("expectation")
    plt.legend()
    # save figure
    plt.savefig(f"plots/{title.replace(' ', '')}", bbox_inches='tight')
