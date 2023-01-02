import random
import json
import os
import math

class Q_learning:
    def __init__(self, initial_value, learning_rate, env, discount_factor, from_jason=False, epsilon=1, tau=1):
        self.initial_value = initial_value
        self.learning_rate = learning_rate
        if from_jason and os.path.isfile("qtable.json"):
            self.table = self.getTable()
        else:
            self.table = self.generateTable()
        self.env = env
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.tau = tau
        self.movesCount = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}
    
    def resetTable(self):
        self.table = self.generateTable()

    def generateTable(self):
        Q_dict = dict()
        for i in range(500):
            Q_dict[i] = [0 for _ in range(6)]
        return Q_dict
    
    def greedyChoice(self, state):
        """
        Choose action that has the higest Q-value.
        """
        moves = self.table[state]
        maxMove = max(moves)
        goodMoves = []
        for i, move in enumerate(moves):
            if move == maxMove:
                goodMoves.append(i)
        if len(goodMoves) == 0:
            return 0
        else:
            return random.choice(goodMoves)
    
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
        moveProb = []
        denominator = 0
        for qvalue in moves:
            denominator += math.exp((qvalue)/self.tau)
        for qvalue in moves:
            moveProb.append(math.exp((qvalue)/self.tau)/denominator)
        values = [0,1,2,3,4,5]
        return random.choices(values, moveProb)[0]

    def countChoice(self, state):
        """
        Choose action that has been chosen the least amount of times.
        """
        minMove = min(self.movesCount.values())
        goodMoves = []
        for key, value in self.movesCount.items():
            if value == minMove:
                goodMoves.append(key)
        choice = random.choice(goodMoves)
        self.movesCount[choice] += 1
        print(choice)
        return choice
    
    def trainGreedy(self, episodes, turns):
        """
        Train the model using greedy algorithim.
        """
        for _ in range(episodes):
            self.episode(turns, self.greedyChoice)
    
    def trainEpsilonGreedy(self, episodes, turns, epsilon=None):
        """
        Train the model using epsilon greedy algorithim.
        """
        if epsilon is not None and epsilon <= 1 and epsilon >= 0:
            self.epsilon = epsilon
        for _ in range(episodes):
            self.episode(turns, self.epsilonGreedyChoice)
    
    def trainBoltzmann(self, episodes, turns, tau=None):
        """
        Train the model using Boltzmann distribution.
        """
        if tau is not None:
            self.tau = tau
        for _ in range(episodes):
            self.episode(turns, self.boltzmannChoice)
    
    def trainCount(self, episodes, turns):
        """
        Train the model using counting algorithim.
        """
        for _ in range(episodes):
            self.episode(turns, self.countChoice)
    
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
        results = dict()
        strategies = [self.greedyChoice, self.epsilonGreedyChoice, self.boltzmannChoice, self.countChoice]
        useStrategy = [useGreedy, useEpsilonGreedy, useBoltzmann, useCount]
        names = ["greedy", "epsilonGreedy", "boltzmann", "count"]
        for i in range(len(strategies)):
            if useStrategy[i] is True:
                miniResults = {"positive":0, "negative":0, "average":0}
                for _ in range(episodes):
                    wasAchieved, numOfTurns = self.episodeTest(turns, strategies[i])
                    if wasAchieved:
                        miniResults["positive"] += 1
                        miniResults["average"] += numOfTurns
                    else:
                        miniResults["negative"] += 1
                if miniResults["positive"] != 0:
                    miniResults["average"] = miniResults["average"] / miniResults["positive"]
                else:
                    miniResults["average"] = 0
                results[f"{names[i]}"] = miniResults
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
        jstring = json.dumps(self.table)
        with open("qtable.json", 'w') as file:
            file.write(jstring)
    
    def getTable(self):
        with open("qtable.json", 'r') as file:
            table = json.load(file)
            newTable = dict()
            for key, value in table.items():
                newTable[int(key)] = value
        return newTable
