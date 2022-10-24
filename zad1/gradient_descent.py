import matplotlib.pyplot as plt
from autograd import grad
from autograd import numpy as np
import os

class LabFunc:
    def __init__(self, alpha, n):
        self.alpha = alpha
        self.n = n
    def returnValue(self, x):
        y = 0
        for i in range(self.n):
            y += pow(self.alpha, i/self.n - 1) * pow(x[i], 2)
        return y

def gradientDescent(f, x0, lengthFactor, maxSteps):
    localLengthFactor = lengthFactor
    old_x = x0
    gradFunc = grad(f.returnValue)
    old_value = f.returnValue(old_x)
    iterationValues = [old_value]
    for i in range(maxSteps):
        gradValue = gradFunc(old_x)
        new_x = old_x - (gradValue * localLengthFactor)
        new_value = f.returnValue(new_x)
        if np.linalg.norm(gradValue) <= lengthFactor * 0.1:
            return f.returnValue(new_x), iterationValues 
        if new_value >= old_value:
            localLengthFactor = localLengthFactor * 0.1
        else:
            old_x = new_x
            old_value = f.returnValue(old_x)
            localLengthFactor = lengthFactor
        iterationValues.append(old_value)
    return f.returnValue(new_x), iterationValues    

if __name__ == "__main__":
    maxSteps = 100
    iterations = range(maxSteps + 1)
    if not os.path.isdir("graphs"):
        os.makedirs("graphs")
    for lengthFactor in [0.01, 0.1, 1, 10]:
        fig, ax = plt.subplots(dpi=300)
        for alpha in [1, 5, 10, 50, 100]:
            labFunc = LabFunc(alpha, 10)
            minimum, iterationTimes = gradientDescent(labFunc, np.array([10.0, 20.0, 30.0, 40.0, 50.0, 10.0, 20.0, 30.0, 40.0, 50.0]), float(lengthFactor), maxSteps)
            ax.plot(range(len(iterationTimes)), iterationTimes, label=f"alpha = {alpha}")
        plt.yscale('log')
        plt.xlabel("t")
        plt.ylabel("q(x)")
        plt.title("Values of q(x) over iterations")
        plt.legend()
        fig.savefig(f"graphs/lengthFactor={lengthFactor}.jpg")
    #plt.show()
