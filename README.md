# LSINF2275 - Data Mining and Decision Making - Project 1 : Snakes and Ladders

Group Members: 
- Augustin d'Oultremont - 2239 1700 - INFO
- Brieuc de Voghel - 5910 1700 - INFO
- Valentin Lemaire - 1634 1700 - INFO

## Project structure:
This repository contains the following architecture :
```
/
|
+-- snakes_and_ladders.py 
|
+-- testbench.py
|
+-- strategies.py
|
+-- layouts.py
|
+-- plots.py
|
+-- README.txt
|
+-- plots/
|   |
|   +-- [...]
|
+-- results/
	|
	+-- final_results.txt
```

## File descriptions:

- [snakes_and_ladders.py](/snakes_and_ladders.py) : this file contains the function `markovDecision(layout, circle)` required by the project and all of its auxiliary functions and classes as well as the `test_empirically` function that allows us to test empirically any strategy on a given layout

- [testbench.py](/testbench.py) : this file launches empirical tests on many different layouts and writes results to the `/results/final_results.txt` file

- [strategies.py](/strategies.py) : this file exports different game strategies (greedy, only one type of die, ...)

- [layouts.py](/layouts.py) : this file exports the different test layouts of the game

- [plots.py](/plots.py) : this file reads from the `/results/final_results.txt` file and plots the different graphs we need and exports them in the `/plots` repository
