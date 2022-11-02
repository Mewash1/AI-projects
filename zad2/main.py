from cec2017.functions import f1, f9
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.optimize import fmin


def generatePopulation(f, dimensions, populationSize):
    population = []
    for _ in range(populationSize):
        x = np.random.uniform(-100, 100, size=dimensions)
        valueTup = (x, f(x))
        population.append(valueTup)
    return population

def reprodution(population):
    newPopulation = []
    for _ in range(int(len(population) / 2)):
        specimen1, specimen2 = random.choice(population), random.choice(population)
        newSpecimen = specimen1 if specimen1[1] <= specimen2[1] else specimen2
        newPopulation.append(newSpecimen)
        newPopulation.append(newSpecimen)
    return newPopulation

def mutation(population, mutationForce, f):
    mutatedPopulation = []
    for specimen in population:
        x = np.random.normal(3, 2.5, size=(len(specimen[0])))
        newX = specimen[0] + mutationForce * x
        mutatedPopulation.append((newX, f(newX)))
    return mutatedPopulation


def evolution(mutationForce, f, mutantsSize, populationSize, days, dimensions):
    population = generatePopulation(f, dimensions, populationSize)
    bestSpecimens = []
    for _ in range(days):
        reproducedPopulation = reprodution(population)
        mutatedPopulation = mutation(reproducedPopulation, mutationForce, f)

        sortedPopulation = sorted(reproducedPopulation, key=lambda x : x[1])
        sortedMutants = sorted(mutatedPopulation, key=lambda x : x[1])

        population = sorted(sortedPopulation[:(populationSize - mutantsSize)] + sortedMutants[:mutantsSize], key=lambda x : x[1])
        populationValues = [specimen[1] for specimen in population]
        bestSpecimens.append(sum(populationValues)/len(populationValues))
    return bestSpecimens

if __name__ == "__main__":
    populationSize = 100
    mutantsSize = 10
    mutationForce = 2
    days = 500
    dimensions = 10
    bestSpecimens = evolution(mutantsSize, f1, mutantsSize, populationSize, days, dimensions)

    # specimenValues = [specimen[1] for specimen in bestSpecimens]

    for value in bestSpecimens:
        print(value)

    plt.yscale("log")
    plt.plot(range(len(bestSpecimens)), bestSpecimens)
    plt.show()

    # fmin(f9,np.array([0,0,0,0,0]))