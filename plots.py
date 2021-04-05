from matplotlib import pyplot as plt
from layouts import *

filename = "results/final_results.txt"
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
class Result:
    def __init__(self, type):
        self.type = type
        self.policy_name = None
        self.policy_dice = None
        self.layout = None
        self.circle = None
        self.iterations = None
        self.expectation = None


# read file
results = []
with open(filename, 'r') as f:
    for line in f.readlines():
        line = line.split()
        if len(line) == 0 or line[0] == "#":
            continue

        elif line[0] == "new":
            results.append(Result(line[-1]))

        elif line[0] == "name":
            results[-1].policy_name = line[-1]

        elif line[0] == "layout":
            results[-1].layout = list(map(int, map(float, line[2:])))

        elif line[0] == "circle":
            results[-1].circle = line[-1] == "True"

        elif line[0] == "iterations":
            results[-1].iterations = int(line[-1])

        elif line[0] == "dice":
            results[-1].policy_dice = list(map(int, map(float, line[2:])))

        elif line[0] == "expectation":
            results[-1].expectation = list(map(float, line[2:]))

# group by layout
grouped = []
for result in results:
    # keep track if a result needs a new group
    new_group = True
    # check all existing groups
    for i, group in enumerate(grouped):
        if result.layout == group[0].layout:
            grouped[i].append(result)
            new_group = False
            break
    if new_group:
        grouped.append([result])
results = grouped


# plot functions
def graph_theoretical_vs_empirical():
    # for each layout
    for group in grouped:
        # create one plot
        plt.figure(figsize=(5, 5))
        # get layout name
        title = "No name found for this layout"
        for layout_name, layout_tiles in layout_match:
            if list(group[0].layout) == list(layout_tiles):
                title = layout_name
                break
        # get names and expectation
        names = []
        expec = []
        for r in group:
            names.append(r.policy_name)
            expec.append(r.expectation[0])
        # create bars
        plt.bar(names, expec)
        # labels and titles
        plt.title(f"{title}")
        plt.xlabel("Policy")
        plt.ylabel("Expectation")
        # save figure
        plt.savefig(f"plots/{title.replace(' ', '')}", bbox_inches='tight')

        # plotting empiric/theoric graph
        # get names and expectation
        plt.figure(figsize=(7, 5))
        names = []
        expec = []

        X_axis = np.arange(1, len(r.layout))

        for r in group:
            # if only plot the theory vs empiric
            if r.circle:
                if r.policy_name == 'markov':
                    plt.bar(X_axis - 0.3, r.expectation, 0.2, label='MDP with circle', color="#4E79A7")
                elif r.policy_name == 'optimal':
                    plt.bar(X_axis - 0.1, r.expectation, 0.2, label='Empirical with circle', color="#A0CBE8")
            else:
                if r.policy_name == 'markov':
                    plt.bar(X_axis + 0.1, r.expectation, 0.2, label='MDP without circle', color="#F28E2B")
                elif r.policy_name == 'optimal':
                    plt.bar(X_axis + 0.3, r.expectation, 0.2, label='Empirical without circle', color="#FFBE7D")

        # labels and titles
        title = f"{title} - Empirical vs MDP"
        plt.title(f"{title}")
        plt.xlabel("State")
        plt.ylabel("Expectation")
        plt.legend()
        # save figure
        plt.savefig(f"plots/{title.replace(' ', '')}", bbox_inches='tight')


def graph_different_policies():
    # for each layout
    for group in grouped:
        # create one plot
        plt.figure(figsize=(7, 5))
        # get layout name
        title = "No name found for this layout"
        for layout_name, layout_tiles in layout_match:
            if list(group[0].layout) == list(layout_tiles):
                title = layout_name
                break
        # get names and expectation
        names_circle = []
        expec_circle = []
        names_no_circle = []
        expec_no_circle = []

        for r in group:
            if r.circle:
                names_circle.append(r.policy_name)
                expec_circle.append(r.expectation[0])
            else:
                names_no_circle.append(r.policy_name)
                expec_no_circle.append(r.expectation[0])

        # sort arrays
        names_circle = ['markov', 'optimal', 'suboptimal', 'secure', 'normal', 'risky', 'random']
        idx_circle = np.argsort(names_circle)
        names_circle = np.array(names_circle)[idx_circle]
        expec_circle = np.array(expec_circle)[idx_circle]

        names_no_circle = ['markov', 'optimal', 'suboptimal', 'secure', 'normal', 'risky', 'random']
        idx_no_circle = np.argsort(names_no_circle)
        names_no_circle = np.array(names_no_circle)[idx_no_circle]
        expec_no_circle = np.array(expec_no_circle)[idx_no_circle]

        # create bars
        X_axis = np.arange(len(names_circle))
        plt.xticks(X_axis, names_circle)
        plt.bar(X_axis - 0.2, expec_no_circle, 0.4, label='without circle')
        plt.bar(X_axis + 0.2, expec_circle, 0.4, label='with circle')

        # labels and titles
        plt.title(f"{title}")
        plt.xlabel("Policy")
        plt.ylabel("Expectation")
        plt.legend()
        # save figure
        plt.savefig(f"plots/{title.replace(' ', '')}", bbox_inches='tight')


if __name__ == '__main__':
    graph_different_policies()
    graph_theoretical_vs_empirical()
