import time
import matplotlib.pyplot as plt
from autograd import grad
from autograd import numpy as np
import os

class LabFunc:
    def __init__(self, alpha, n):
        self.alpha = alpha
        self.n = n
    def __call__(self, x):
        y = 0
        for i in range(self.n):
            y += pow(self.alpha, i/self.n - 1) * pow(x[i], 2)
        return y

def gradientDescent(f, x0, lengthFactor, maxSteps):
    localLengthFactor = lengthFactor
    old_x = x0
    gradFunc = grad(f)
    old_value = f(old_x)
    iterationValues = [old_value]
    for i in range(maxSteps):
        gradValue = gradFunc(old_x)
        new_x = old_x - (gradValue * localLengthFactor)
        new_value = f(new_x)
        if np.linalg.norm(gradValue) <= lengthFactor * 0.1:
            return f(new_x), iterationValues 
        if new_value >= old_value:
            localLengthFactor = localLengthFactor * 0.1
        else:
            old_x = new_x
            old_value = f(old_x)
        iterationValues.append(old_value)
    return f(new_x), iterationValues    

if __name__ == "__main__":

    if not os.path.isdir("graphs"):
        os.makedirs("graphs")

    maxSteps = 10000
    alphaValues = [1, 5, 10, 50, 100]
    lengthFactors = [0.01, 0.1, 1.0, 10.0]
    
    for lengthFactor in lengthFactors:
        fig1, (ax1, ax2) = plt.subplots(2, figsize=(12,6), dpi=300)
        fig1.tight_layout(pad=3.0)
        alphaTimes = []
        for alpha in alphaValues:
            labFunc = LabFunc(alpha, 10)
            start = time.time()
            minimum, iterationValues = gradientDescent(labFunc, np.array([10.0, 20.0, 30.0, 40.0, 50.0, 10.0, 20.0, 30.0, 40.0, 50.0]), lengthFactor, maxSteps)
            end = time.time()
            ax1.plot(range(len(iterationValues)), iterationValues, label=f"alpha = {alpha}")
            alphaTimes.append((alpha, end - start))
        ax2.plot([value[0] for value in alphaTimes], [value[1] for value in alphaTimes])
        
        ax1.set_yscale('log')
        ax1.set_xlabel("t")
        ax1.set_ylabel("q(x)")
        ax1.set_title("Values of q(x) over iterations")
        ax1.legend()

        ax2.set_xlabel("alpha")
        ax2.set_ylabel("time (seconds)")
        ax2.set_xscale("log")
        ax2.set_title("Alpha values over time of execution")
        fig1.savefig(f"graphs/lengthFactor={lengthFactor}.jpg")
