from scipy.optimize import minimize, LinearConstraint
import numpy as np
from math import exp

class SVM:
    def __init__(self, lambdaVar, trainingSet, testSet, startingAlpha, delta):
        self.lambdaVar = lambdaVar
        self.trainingSet = trainingSet
        self.testSet = testSet
        self.alpha = startingAlpha
        self.delta = delta
        self.b = 0

    def funDual(self, alpha, kernel):
        firstSum = sum(alpha)
        secondSum = 0
        for i in range(len(alpha)):
            for j in range(len(alpha)):
                i_attributes, i_objectClass = self.trainingSet[i]
                j_attributes, j_objectClass = self.trainingSet[j]
                secondSum += i_objectClass * alpha[i] * j_objectClass * alpha[j] * (kernel(i_attributes, j_attributes))
        return (0.5 * secondSum) - firstSum # -f(x) - maximize
    
    def constraint(self, alpha):
        output = 0
        for i in range(len(alpha)):
            output += alpha[i] * self.trainingSet[i][1]
        return output

    def learnPolynomialKernel(self):
        cons = {'type':'eq', 'fun': self.constraint}
        bounds = [(0.0, 1/len(self.alpha)*self.lambdaVar*2) for _ in range(len(self.alpha))]
        self.alpha = minimize(self.funDual, self.alpha, (self.polynomialKernel), bounds=(bounds), constraints=cons).x
        self.setB(self.polynomialKernel)

    def learnRBFkernel(self):
        cons = {'type':'eq', 'fun': self.constraint}
        bounds = [(0.0, 1/len(self.alpha)*self.lambdaVar*2) for _ in range(len(self.alpha))]
        self.alpha = minimize(self.funDual, self.alpha, (self.RBFkernel), bounds=(bounds), constraints=cons).x
        self.setB(self.RBFkernel)

    def setB(self, kernel):
        bPoint = None
        for i in range(len(self.alpha)):
            if self.alpha[i] > 0 and self.alpha[i] < 1/(len(self.alpha)*self.lambdaVar*2):
                bPoint = self.trainingSet[i]
        if bPoint is not None:
            bSum = 0
            for j in range(len(self.alpha)):
                bSum += self.alpha[j] * self.trainingSet[j][1] * kernel(self.trainingSet[j][0], bPoint[0])
            self.b = bSum - bPoint[1]

    def polynomialKernel(self, x1, x2):
        return pow(np.dot(x1,x2) + 1, self.delta) 
    
    def RBFkernel(self, x1, x2):
        return exp((-(np.linalg.norm(np.array(x1, dtype=object) - np.array(x2, dtype=object)))**2)/(2*(self.delta**2)))

    def test(self, rbf=False):
        if not rbf:
            kernel = self.polynomialKernel
        else:
            kernel = self.RBFkernel
        positiveResults = 0
        for wine in self.testSet:
            output = 0
            for i in range(len(self.alpha)):
                output += self.alpha[i] * self.trainingSet[i][1] * kernel(self.trainingSet[i][0],wine[0])
                output -= self.b
            wineClass = 1 if output >= 1 else -1
            if wineClass == wine[1]:
                positiveResults += 1
        return (positiveResults/len(self.testSet)* 100)
