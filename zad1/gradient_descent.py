import math
import copy
from autograd import grad
import numpy as np

class LabFunc:
    def __init__(self, alpha, n):
        self.alpha = alpha
        self.n = n
    def returnValue(self, x):
        y = 0
        for i in range(1, self.n):
            y += pow(self.alpha, i - 1/self.n - 1) * pow(x[i - 1], 2)
        return y

class QuadraticFunc:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
    def returnValue(self, x):
        y = 0
        for value in x:
            y += self.a * pow(value, 2) + self.b * value + self.c
        return y

def gradientDescent(f, x0, lengthFactor, maxSteps):
    # krok początkowy - x0
    old_x = x0
    gradFunc = grad(f.returnValue)
    # main loop
    for i in range(maxSteps):
        gradValue = gradFunc(old_x)
        # new_x = [value - (lengthFactor * gradValue) for value in old_x]
        new_x = []
        for i, value in enumerate(old_x):
            new_x.append(value - (lengthFactor * gradValue[i]))
        if np.linalg.norm(gradValue) <= lengthFactor:
            return f.returnValue(new_x)
        elif f.returnValue(new_x) >= f.returnValue(old_x):
            lengthFactor = lengthFactor / 2
        else:
            old_x = new_x
    return f.returnValue(new_x)    

if __name__ == "__main__":
    # lengthFactor should be very small
    labFunc = LabFunc(10, 10)
    testFunc = QuadraticFunc(1, 0, 0)
    # print(gradientDescent(testFunc, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 0.05, 100))
    print(gradientDescent(labFunc, np.array([0.0, 1.0, -1.0, 2.0, 2.1, 2.2, -2.1, -2.2, -3.0, 3.0]), 0.03, 5000))