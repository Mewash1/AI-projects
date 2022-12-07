from scipy.optimize import minimize
import numpy as np
from math import exp

class SVM:
    def __init__(self, lambdaVar, trainingSet, testSet, startingAlpha, delta):
        self.lambdaVar = lambdaVar
        self.trainingSet = trainingSet
        self.testSet = testSet
        self.alpha = startingAlpha
        self.weight = np.zeros(len(self.trainingSet[0][0]))
        self.delta = delta
        self.b = 0

    def fun(self, alpha):
        output = 0
        b = alpha[-1]
        for point in self.trainingSet:
            attributes, objectClass = point
            output += max(0, 1 - objectClass*(np.dot(alpha[:-1], attributes) - b))
        output = 1/len(self.trainingSet) * output
        output += self.lambdaVar * pow(np.linalg.norm(alpha), 2)
        return output

    def funDual(self, alpha, kernel):
        firstSum = sum(alpha)
        secondSum = 0
        for i in range(len(alpha)):
            for j in range(len(alpha)):
                i_attributes, i_objectClass = self.trainingSet[i]
                if firstSum * i_objectClass == 0 and alpha[i] >= 0 and alpha[i] <= 1/(len(alpha)*self.lambdaVar*2):
                    j_attributes, j_objectClass = self.trainingSet[j]
                    secondSum += i_objectClass * alpha[i] * j_objectClass * alpha[j] * (kernel(i_attributes, j_attributes))
        return (0.5 * secondSum) - firstSum # -f(x) - maximize
    
    def learnPolynomialKernel(self):
        self.alpha = minimize(self.funDual, self.alpha, (self.polynomialKernel)).x
        self.setWeightAndB()

    def learnRBFkernel(self):
        self.alpha = minimize(self.funDual, self.alpha, (self.RBFkernel)).x
        self.setWeightAndB()

    def setWeightAndB(self):
        bPoint = None
        for i in range(len(self.alpha)):
            attributes, objectClass = self.trainingSet[i]
            if self.alpha[i] > 0 and self.alpha[i] < 1/(len(self.alpha)*self.lambdaVar*2):
                bPoint = self.trainingSet[i]
            self.weight += objectClass * self.alpha[i] * np.array(attributes)
        if bPoint is not None:
            self.b = np.dot(self.weight, bPoint[0]) - bPoint[1]

    def polynomialKernel(self, x1, x2):
        return pow(np.dot(x1,x2), self.delta)
    
    def RBFkernel(self, x1, x2):
        return exp((-(np.linalg.norm(np.array(x1, dtype=object) - np.array(x2, dtype=object)))**2)/(2*(self.delta**2)))
    
    def test(self):
        positiveResults = 0
        for wine in self.testSet:
            output = np.dot(self.weight, wine[0]) - self.b
            wineClass = 1 if output >= 1 else -1
            if wineClass == wine[1]:
                positiveResults += 1
        return (positiveResults/len(self.testSet)* 100)
