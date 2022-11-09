from evolution.evolutionStrategy import evolution
from gradientDescent.gradient_descent import gradientDescent
import matplotlib.pyplot as plt
from evolution.cec2017.functions import f1, f9
import numpy as np
import os

testFunctions = [("f1", f1), ("f9", f9)]
mutationForces = [1, 5, 10]
populationAndReproductionSizes = [(10, 20), (50, 100), (100, 200)]
days = 300
dimensions = 10
initialCoords = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 10.0, 20.0, 30.0, 40.0, 50.0])
lengthFactor = 2.0
evolutionTrials = 5

if not os.path.isdir("cecGraphs"):
    os.makedirs("cecGraphs")

for mutationForce in mutationForces:
    for fTuple in testFunctions:
        name, f = fTuple
        fig, ax = plt.subplots(figsize=(12,6), dpi=300)
        ax.set_yscale("log")
        ax.set_ylabel("f(x)")
        ax.set_xlabel("iterations")
        
        ax.set_title(f"Minimal value of {name} with starting σ = {mutationForce}")
        minimum, iterationValues = gradientDescent(f, initialCoords, lengthFactor, days)
        ax.plot(range(len(iterationValues)), iterationValues, label="gradient")

        for populationAndReproductionSize in populationAndReproductionSizes:
            populationSize, reproductionSize = populationAndReproductionSize
            results = []
            for _ in range(evolutionTrials):
                results.append(evolution(f, mutationForce, populationSize, reproductionSize, days, dimensions))
            bestSpecimens = []
            for i in range(len(results[0])):
                averageResult = 0
                for j in range(evolutionTrials):
                    averageResult += results[j][i]
                bestSpecimens.append(averageResult/evolutionTrials)
            ax.plot(range(len(bestSpecimens)), bestSpecimens, label=f"μ, λ = {populationSize, reproductionSize}")
        ax.legend()
        fig.savefig(f"cecGraphs/{name}, σ = {mutationForce}")
