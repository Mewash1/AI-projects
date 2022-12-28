import gym
import random
import json
import os

class Q_learning:
    def __init__(self, initial_value, learning_rate, env, discount_factor, from_jason=False):
        self.initial_value = initial_value
        self.learning_rate = learning_rate
        if from_jason and os.path.isfile("qtable.json"):
            self.table = self.getTable()
        else:
            self.table = self.generateTable()
        self.env = env
        self.discount_factor = discount_factor
    
    def generateTable(self):
        Q_dict = dict()
        for i in range(500):
            Q_dict[i] = [self.initial_value] * 6
        return Q_dict
    
    def pickMove(self, state):
        try:
            moves = self.table[state]
        except KeyError:
            moves = [self.initial_value] * 6
        maxMove = max(moves)
        goodMoves = []
        for i, move in enumerate(moves):
            if move == maxMove:
                goodMoves.append(i)
        return random.choice(goodMoves)
    
    def getBestValue(self, state):
        return max(self.table[state])
    
    def train(self, episodes, turns):
        for _ in range(episodes):
            self.episode(turns)
    
    def episode(self, turns):
        observation_old, info = self.env.reset()
        for _ in range(turns):
            move = self.pickMove(observation_old)
            observation_new, reward, terminated, truncated, info = self.env.step(move)
            self.table[observation_old][move] = self.table[observation_old][move] + self.learning_rate * (reward + (self.discount_factor * self.getBestValue(observation_new)) - self.table[observation_old][move])
            observation_old = observation_new
            if terminated:
                break
    
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

if __name__ == "__main__":
    envi = gym.make('Taxi-v3')
    
    episodes = 5000
    moves = 50
    test_moves = 50

    Q_learn = Q_learning(initial_value=0, learning_rate=0.5, env=envi, discount_factor=0.1, from_jason=True)
    #Q_learn.train(episodes=episodes, turns=moves)
    envi.close()

    envi = gym.make('Taxi-v3', render_mode="human")
    Q_learn.env = envi
    Q_learn.episode(test_moves)

    #Q_learn.saveTable()

    envi.close()
