import numpy as np
from wineProcessing import *
from svm import SVM

def testForWineType(trainingSize, testSize, lambdaVar, delta):
    redWines = processWineType("winequality-red.csv", isRed=True)
    whiteWines = processWineType("winequality-white.csv", isRed=False)
    allWines = np.concatenate((redWines, whiteWines))
    np.random.shuffle(allWines)
    trainingSet = allWines[:trainingSize]
    allWines = allWines[trainingSize:]
    testSet = allWines[:testSize]

    svm = SVM(lambdaVar=lambdaVar, trainingSet=trainingSet, testSet=testSet, delta=delta, startingAlpha=np.array([1]*(len(trainingSet))))
    svm.learnPolynomialKernel()
    return svm.test()

def testForQuality(trainingSize, testSize, lambdaVar, delta, RBF=True):
    redWines = processWineQuality("winequality-red.csv")
    whiteWines = processWineQuality("winequality-white.csv")
    allWines = np.concatenate((redWines, whiteWines))
    np.random.shuffle(allWines)
    trainingSet = allWines[:trainingSize]
    allWines = allWines[trainingSize:]
    testSet = allWines[:testSize]

    svm = SVM(lambdaVar=lambdaVar, trainingSet=trainingSet, testSet=testSet, delta=delta, startingAlpha=np.array([1]*(len(trainingSet))))
    if RBF:
        svm.learnRBFkernel()
    else:
        svm.learnPolynomialKernel()
    return svm.test()

if __name__ == "__main__":
    n = 10
    outputString = "RBF Kernel: \n\n"
    for lambdaVar in [0.01]:
        for delta in [1]:
            percents = []
            for i in range(n):
                percents.append(testForQuality(trainingSize=10, testSize=6000, lambdaVar=0.01, delta=delta))
            outputString += f"lambda={lambdaVar}, delta={delta}, effiency={round(sum(percents)/n, 2)}%\n"

    outputString += "Polynomial kernel: \n\n"
    for lambdaVar in [0.01]:
        for delta in [1]:
            percents = []
            for i in range(n):
                percents.append(testForQuality(trainingSize=10, testSize=6000, lambdaVar=0.01, delta=delta, RBF=False))
            outputString += f"lambda={lambdaVar}, delta={delta}, effiency={round(sum(percents)/n, 2)}%\n"

    with open("results1.txt", 'w') as file:
        file.write(outputString)