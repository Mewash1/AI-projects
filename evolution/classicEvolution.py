from cec2017.functions import f1, f9
import numpy as np
import random
import matplotlib.pyplot as plt


def generatePopulation(f, dimensions, populationSize):
    population = []
    for _ in range(populationSize):
        x = np.random.uniform(-100, 100, size=dimensions)
        population.append((x, f(x)))
    return population

def reprodution(population):
    newPopulation = []
    for _ in range(int(len(population) / 2)):
        specimen1, specimen2 = random.choice(population), random.choice(population)
        newSpecimen = specimen1 if specimen1[1] <= specimen2[1] else specimen2
        newPopulation.extend([newSpecimen, newSpecimen])
    return newPopulation

def mutation(population, mutationForce, f):
    mutatedPopulation = []
    for specimen in population:
        x = np.random.normal(0.0, 1.0, size=(len(specimen[0])))
        newX = specimen[0] + mutationForce * x
        mutatedPopulation.append((newX, f(newX)))
    return mutatedPopulation


def evolution(f, mutationForce, numberOfMutants, populationSize, days, dimensions):
    population = generatePopulation(f, dimensions, populationSize)
    bestSpecimens = []
    for _ in range(days):
        reproducedPopulation = reprodution(population)
        mutatedPopulation = mutation(reproducedPopulation, mutationForce, f)

        # the specimen with the lowest function value is the best one, so it sits at the top of the list
        sortedPopulation = sorted(reproducedPopulation, key=lambda x : x[1])
        sortedMutants = sorted(mutatedPopulation, key=lambda x : x[1])

        population = sorted(sortedPopulation[:(populationSize - numberOfMutants)] + sortedMutants[:numberOfMutants], key=lambda x : x[1])
        populationValues = [specimen[1] for specimen in population]
        
        # average specimen value
        bestSpecimens.append(sum(populationValues)/len(populationValues))
    return bestSpecimens

if __name__ == "__main__":
    f = f1
    mutationForce = 2
    numberOfMutants = 10
    populationSize = 100
    days = 5000
    dimensions = 10
    bestSpecimens = evolution(f, mutationForce, numberOfMutants, populationSize, days, dimensions)

    for value in bestSpecimens:
        print(value)

    plt.yscale("log")
    plt.plot(range(len(bestSpecimens)), bestSpecimens)
    plt.show()
