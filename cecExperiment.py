from evolution.evolutionStrategy import evolution
from gradientDescent.gradient_descent import gradientDescent
import matplotlib.pyplot as plt
from evolution.cec2017.functions import f1, f9
import numpy as np
import os

testFunctions = [("f1", f1), ("f9", f9)]
mutationForce = 2
populationSize = 100
reproductionSize = 200
days = 500
dimensions = 10
initialCoords = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 10.0, 20.0, 30.0, 40.0, 50.0])
lengthFactor = 1

if not os.path.isdir("cecGraphs"):
    os.makedirs("cecGraphs")

for fTuple in testFunctions:
    name, f = fTuple
    fig, ax = plt.subplots(figsize=(12,6), dpi=300)
    bestSpecimens = evolution(f, mutationForce, populationSize, reproductionSize, days, dimensions)
    minimum, iterationValues = gradientDescent(f, initialCoords, lengthFactor, days)
    ax.set_yscale("log")
    ax.set_ylabel("f(x)")
    ax.set_xlabel("iterations")
    ax.plot(range(len(bestSpecimens)), bestSpecimens, label="evolution")
    ax.plot(range(len(iterationValues)), iterationValues, label="gradient")
    ax.legend()
    ax.set_title(f"Minimal value of {name}")
    fig.savefig(f"cecGraphs/{name}")
