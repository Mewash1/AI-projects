import time
import matplotlib.pyplot as plt
from autograd import grad
from matplotlib.scale import LogScale
import numpy as np

class LabFunc:
    def __init__(self, alpha, n):
        self.alpha = alpha
        self.n = n
    def returnValue(self, x):
        y = 0
        for i in range(1, self.n + 1):
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
    old_x = x0
    gradFunc = grad(f.returnValue)
    iterationTimes = []
    # main loop
    for i in range(maxSteps):
        # start = time.time()
        gradValue = gradFunc(old_x)
        new_x = []
        for j, value in enumerate(old_x):
            new_x.append(value - (lengthFactor * gradValue[j]))
        #if np.linalg.norm(gradValue) <= lengthFactor:
        #    return f.returnValue(new_x)
        if f.returnValue(new_x) >= f.returnValue(old_x):
            lengthFactor = lengthFactor / 2
        else:
            old_x = new_x
        # end = time.time()
        iterationTimes.append(f.returnValue(new_x))
        # print(f.returnValue(new_x))
    return f.returnValue(new_x), iterationTimes    

if __name__ == "__main__":
    # lengthFactor should be very small
    maxSteps = 1000
    iterations = [i for i in range(1, maxSteps + 1)]

    for i in [1, 10, 100]:
        labFunc = LabFunc(i, 10)
        for lengthFactor in [1, 2, 3]:
            minimum, iterationTimes = gradientDescent(labFunc, np.array([100.0, 30.0, 30.0, 40.0, 50.0, 60.0, -50.0, -70.0, -39.0, -10.0]), float(lengthFactor), maxSteps)
            plt.plot(iterations, iterationTimes, label=f"alpha = {i}, length factor = {lengthFactor}")
    
    plt.yscale('log')
    plt.xlabel("t")
    plt.ylabel("q(x)")
    plt.title("Values of q(x) over iterations")
    plt.legend(loc="upper right")
    plt.show()