from typing import List
from autograd import grad

class LabFunc:
    def __init__(self, alpha, n):
        self.alpha = alpha
        self.n = n
    def returnValue(self, x:List):
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

def solver(f, x0: List, alpha):
    # krok początkowy - x0
    old_x = x0
    grad_f = grad(f.returnValue)
    alpha_k = alpha
    # main loop
    while True:
        grad_k = grad_f(old_x)
        new_x = [value - (alpha_k * grad_k) for value in old_x]
        if grad_k <= alpha_k:
            return f.returnValue(new_x)
        elif f.returnValue(new_x) >= f.returnValue(old_x):
            alpha_k / 2
        else:
            old_x = copy.deepcopy(new_x)

def test(x):
    return x


if __name__ == "__main__":
    # alpha should be very small
    labFunc = LabFunc(1, 10)
    testFunc = QuadraticFunc(1, 0, 0)
    print(solver(testFunc, [50.0], 0.05))