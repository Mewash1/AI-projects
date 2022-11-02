from cec2017.functions import f1, f9
import numpy as np
import random
import copy


def generatePopulation(f):
    population = []
    for _ in range(100):
        x = np.random.uniform(-100, 100, size=10)
        valueTup = (x, f(x))
        population.append(valueTup)
    return population

def reprodution(population):
    populationCopy = copy.deepcopy(population)
    newPopulation = []
    for _ in range(int(len(population) / 2)):
        specimen1, specimen2 = random.choice(populationCopy), random.choice(populationCopy)
        if specimen1[1] <= specimen2[1]:
            newPopulation.append(specimen1)
            newPopulation.append(copy.deepcopy(specimen1))
        else:
            newPopulation.append(specimen2)
            newPopulation.append(copy.deepcopy(specimen2))

    return newPopulation

def mutation(population, mutationForce, f):
    mutatedPopulation = []
    for specimen in population:
        x = np.random.normal(3, 2.5, size=(len(specimen[0])))
        newX = specimen[0] + mutationForce * x
        mutatedPopulation.append((newX, f(newX)))
    return mutatedPopulation


def evolution(startingPopulation, mutationForce, f, mutantsSize, populationSize, days):
    population = startingPopulation
    bestSpecimens = []
    for _ in range(days):
        reproducedPopulation = reprodution(population)
        mutatedPopulation = mutation(reproducedPopulation, mutationForce, f)

        sortedPopulation = sorted(reproducedPopulation, key=lambda x : x[1])
        sortedMutants = sorted(mutatedPopulation, key=lambda x : x[1])

        population = sorted(sortedPopulation[:(populationSize - mutantsSize)] + sortedMutants[:mutantsSize], key=lambda x : x[1])
        bestSpecimens.append(population[0])
    return bestSpecimens

if __name__ == "__main__":
    populationSize = 100
    mutantsSize = 80
    population = generatePopulation(f1)
    mutationForce = 3
    days = 100
    bestSpecimens = evolution(population, mutantsSize, f1, mutantsSize, populationSize, days)
    for specimen in bestSpecimens:
        print(specimen[1])
    
