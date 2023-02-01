# Off-Policy Evaluation for Episodic Partially Observable Markov Decision Processes under Non-Parametric Models

[Supp_proximal_reinforcement_learning.pdf](Supp_proximal_reinforcement_learning.pdf): Appendix of paper

## Requirements
```
python >= 3.7.10
numpy
scipy
pandas
sklearn >=0.24.1
torch >= 1.10.2
matplotlib
seaborn
gym
```

## Instructions
- We attached the `gym` package cloned from openAI, we only use the classes defined in `gym`.
- Run `simulation.sh` to get `ResultA.csv` and `ResultB.csv`, we use a cuda device (RTX3090) in the experiment
- Run `plots.py` to generate figures 5 (a) and (b) in simulation study.
