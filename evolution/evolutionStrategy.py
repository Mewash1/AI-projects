import numpy as np
import random
import math


def reproduction(population, reproductionSize):
    newPopulation = []
    for _ in range(reproductionSize):
        newPopulation.append(random.choice(population))
    return newPopulation


def generatePopulation(f, dimensions, populationSize, mutationForce):
    population = []
    for _ in range(populationSize):
        mainChromosome = np.random.uniform(-100, 100, size=dimensions)
        mutationChromosome = np.array([mutationForce] * dimensions)
        population.append((mainChromosome, mutationChromosome, f(mainChromosome)))
    return population


def criscross(population, f):
    newPopulation = []
    for _ in range(len(population)):
        mother, father = random.choices(population, k=2)
        mainChromosome = []
        for i in range(len(mother[0])):
            weight = np.random.uniform()
            mainChromosome.append(mother[0][i] * weight + father[0][i] * (1 - weight))
        newPopulation.append((np.array(mainChromosome), mother[1], f(np.array(mainChromosome))))
    return newPopulation


def mutation(population, f):
    newPopulation = []
    for specimen in population:
        a = np.random.normal()
        mainChromosome, mutationChromosome = [], []
        for i in range(len(specimen[0])):
            b = np.random.normal()
            r1 = 1/math.sqrt(2 * len(specimen[0]))
            r2 = 1/math.sqrt(2 * math.sqrt(len(specimen[0])))
            mutationVar = specimen[1][i] * math.exp(r2 * a + r1 * b)
            mainChromosome.append(specimen[0][i] + mutationVar * np.random.normal())
            mutationChromosome.append(mutationVar)
        newPopulation.append((np.array(mainChromosome), np.array(mutationChromosome), f(np.array(mainChromosome))))
    return newPopulation


def evolution(f, mutationForce, populationSize, reproductionSize, days, dimensions):
    population = generatePopulation(f, dimensions, populationSize, mutationForce)
    bestSpecimens = []
    for _ in range(days + 1):
        reproducedPopulation = reproduction(population, reproductionSize)
        criscrossedPopulation = criscross(reproducedPopulation, f)
        mutatedPopulation = mutation(criscrossedPopulation, f)

        wholePopulation = population + mutatedPopulation
        sortedPopulation = sorted(wholePopulation, key=lambda x : x[2])
        population = sortedPopulation[:populationSize]
        populationValues = [specimen[2] for specimen in population]
        bestSpecimens.append(sum(populationValues)/len(populationValues))
    return bestSpecimens
