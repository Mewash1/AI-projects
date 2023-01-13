import random
import json
import os
import numpy as np
from collections import namedtuple

class Q_learning:
    def __init__(self, initial_value, learning_rate, env, discount_factor, from_jason=False, epsilon=1, tau=1):
        self.initial_value = initial_value
        self.learning_rate = learning_rate
        if from_jason and os.path.isfile("qtable.json"):
            self.table = self.loadTable("qtable.json")
        else:
            self.table = self.generateTable()
        self.env = env
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.tau = tau
        self.movesCount = np.zeros(6)
    
    def resetTable(self):
        self.table = self.generateTable()

    def generateTable(self):
        Q_dict = []
        for _ in range(500):
            Q_dict.append(np.array([random.uniform(-1, 1) for _ in range(6)]))
        return np.array(Q_dict)
    
    def greedyChoice(self, state):
        """
        Choose action that has the higest Q-value.
        """
        return np.argmax(self.table[state])
    
    def epsilonGreedyChoice(self, state):
        """
        Choose random action with probability epsilon, or use greedy choice with probability 1 - epsilon.
        """
        choice = random.uniform(0, 1)
        if choice < self.epsilon:
            return random.choice(range(6))
        else:
            return self.greedyChoice(state)
    
    def boltzmannChoice(self, state):
        """
        Choose action with probability from Boltzmann distribution.
        """
        moves = self.table[state]
        denominator = np.sum(np.exp(moves)/self.tau)
        moveProb = np.exp(moves)/denominator
        return random.choices(range(6), moveProb)[0]

    def countChoice(self, state):
        """
        Choose action that has been chosen the least amount of times 20% of the time.
        Else, choose greedily.
        """
        choice = random.uniform(0,1)
        if choice >= 0.2:
            move = self.greedyChoice(state)
        else:
            move = np.argmin(self.movesCount)
        self.movesCount[move] += 1
        return move
    
    def train(self, episodes, turns, strategy):
        pickMove = None
        if strategy == "greedy":
            pickMove = self.greedyChoice
        elif strategy == "epsilonGreedy":
            pickMove = self.epsilonGreedyChoice
        elif strategy == "boltzmann":
            pickMove = self.boltzmannChoice
        elif strategy == "count":
            pickMove = self.countChoice
        else:
            raise ValueError("this strategy does not exist")
        for _ in range(episodes):
            self.episode(turns, pickMove)
    
    def episode(self, turns, pickMove):
        observation_old, info = self.env.reset()
        for _ in range(turns):
            move = pickMove(observation_old)
            observation_new, reward, terminated, truncated, info = self.env.step(move)
            self.table[observation_old][move] = self.table[observation_old][move] + self.learning_rate * (reward + self.discount_factor * max(self.table[observation_new]) - self.table[observation_old][move])
            observation_old = observation_new
            if terminated:
                break
    
    def test(self, episodes, turns, useGreedy=False, useEpsilonGreedy=False, useBoltzmann=False, useCount=False):
        """
        Runs the algorithm without modyfing the Qtable. Depending on the function parameters,
        it is possible to use every strategy for testing the environment.
        Results may vary.
        """
        results = dict()
        Strategy = namedtuple('Strategy', ['use', 'f'])
        strategies = {
            "greedy" : Strategy(useGreedy, self.greedyChoice),
            "epsilonGreedy" : Strategy(useEpsilonGreedy, self.epsilonGreedyChoice),
            "boltzmann" : Strategy(useBoltzmann, self.boltzmannChoice),
            "count" : Strategy(useCount, self.countChoice)
        }
        for name, strategy in strategies.items():
            if strategy.use is True:
                miniResults = {"positive":0, "negative":0, "average":0}
                for _ in range(episodes):
                    wasAchieved, numOfTurns = self.episodeTest(turns, strategy.f)
                    if wasAchieved:
                        miniResults["positive"] += 1
                        miniResults["average"] += numOfTurns
                    else:
                        miniResults["negative"] += 1
                if miniResults["positive"] != 0:
                    miniResults["average"] = miniResults["average"] / miniResults["positive"]
                else:
                    miniResults["average"] = 0
                results[f"{name}"] = miniResults
        return results
                
    def episodeTest(self, turns, pickMove):
        """
        Episode without changing the Qtable. Returns True if the goal has been reached
        and False if it hasn't been reached. It also returns the number of turns which it took
        to achieve the goal.
        """
        observation_old, info = self.env.reset()
        for i in range(turns):
            move = pickMove(observation_old)
            observation_new, reward, terminated, truncated, info = self.env.step(move)
            observation_old = observation_new
            if terminated:
                return (True, i+1)
        return (False, turns)

    def saveTable(self):
        jstring = json.dumps(self.table.tolist())
        with open("qtable.json", 'w') as file:
            file.write(jstring)
    
    def loadTable(self, path):
        with open(path, 'r') as file:
            return np.array(json.load(file))
