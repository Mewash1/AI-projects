from scipy.optimize import minimize
import numpy as np

class SVM:
    def __init__(self, lambdaVar, trainingSet, testSet, startingWeight, delta):
        self.lambdaVar = lambdaVar
        self.trainingSet = trainingSet
        self.testSet = testSet
        self.weight = startingWeight
        self.delta = delta

    def fun(self, weight):
        output = 0
        b = weight[-1]
        for point in self.trainingSet:
            attributes, objectClass = point
            output += max(0, 1 - objectClass*(np.dot(weight[:-1], attributes) - b))
        output = 1/len(self.trainingSet) * output
        output += self.lambdaVar * pow(np.linalg.norm(weight), 2)
        return output

    def funKernel(self, weight):
        output = 0
        b = weight[-1]
        for point in self.trainingSet:
            attributes, objectClass = point
            output += max(0, 1 - objectClass*(np.dot(weight[:-1], attributes) - b))
        output = 1/len(self.trainingSet) * output
        output += self.lambdaVar * pow(np.linalg.norm(weight), 2)
        return output

    def learnLin(self):
        self.weight = minimize(self.fun, self.weight).x
    
    def learnKernel(self):
        self.weight = minimize(self.funKernel, self.weight).x
    
    def RBMkernel(self):
        pass
    
    def test(self):
        positiveResults = 0
        for wine in self.testSet:
            wineClass = self.getPointClass(wine[0])
            if wineClass == wine[1]:
                positiveResults += 1
        print(f"Size of test set = {len(self.testSet)}\nGood classifications = {positiveResults}")

    def getPointClass(self, point):
        b = self.weight[-1]
        output = np.dot(self.weight[:-1], point) - b
        return 1 if output >= 1 else -1
