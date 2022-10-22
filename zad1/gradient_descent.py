import time
import random
import matplotlib.pyplot as plt
from autograd import grad
import numpy as np

class LabFunc:
    def __init__(self, alpha, n):
        self.alpha = alpha
        self.n = n
    def returnValue(self, x):
        y = 0
        for i in range(self.n):
            y += pow(self.alpha, i/self.n - 1) * pow(x[i], 2)
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
    localLengthFactor = lengthFactor
    old_x = x0
    gradFunc = grad(f.returnValue)
    old_value = f.returnValue(old_x)
    iterationValues = [old_value]
    # main loop
    for i in range(maxSteps):
        gradValue = gradFunc(old_x)
        new_x = old_x - (gradValue * localLengthFactor)
        new_value = f.returnValue(new_x)
        # print("old value= ", old_value, "\n",  "new value = ", new_value, "---",  i, "\n", "lengthFactor = ", localLengthFactor)
        if np.linalg.norm(gradValue) <= lengthFactor * 0.1:
            return f.returnValue(new_x), iterationValues 
        if new_value >= old_value:
            localLengthFactor = localLengthFactor * 0.1
        else:
            old_x = new_x
            old_value = f.returnValue(old_x)
            localLengthFactor = lengthFactor
        iterationValues.append(old_value)
        # print(f.returnValue(new_x))
    return f.returnValue(new_x), iterationValues    

if __name__ == "__main__":
    # lengthFactor should be very small
    maxSteps = 100
    lengthFactor = 0.2
    iterations = range(maxSteps + 1)

    for i in [1, 10, 1000]:
        labFunc = LabFunc(i, 10)
        minimum, iterationTimes = gradientDescent(labFunc, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0]), float(lengthFactor), maxSteps)
        #plt.plot(iterations[0:len(iterationTimes)], iterationTimes, label=f"alpha = {i}, length factor = {lengthFactor}")
        plt.plot(range(len(iterationTimes)), iterationTimes, label=f"alpha = {i}, length factor = {lengthFactor}")
    
    print("Len of iterationTimes = ",  len(iterationTimes))
    plt.yscale('log')
    plt.xlabel("t")
    plt.ylabel("q(x)")
    plt.title("Values of q(x) over iterations")
    plt.legend(loc="upper right")
    plt.show()

