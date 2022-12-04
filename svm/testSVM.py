import numpy as np
from wineProcessing import *
from svm import SVM

def testForWineType(trainingSize, testSize, dimensions, lambdaVar):
    redWines = processWineType("winequality-red.csv", isRed=True)
    whiteWines = processWineType("winequality-white.csv", isRed=False)
    allWines = np.concatenate((redWines, whiteWines))
    np.random.shuffle(allWines)
    trainingSet = allWines[:trainingSize]
    allWines = allWines[trainingSize:]
    testSet = allWines[:testSize]

    svm = SVM(lambdaVar=lambdaVar, trainingSet=trainingSet, testSet=testSet, delta=0, startingWeight=np.array([1]*(dimensions+1)))
    svm.learnLin()
    svm.test()

def testForQuality(trainingSize, testSize, dimensions, lambdaVar):
    redWines = processWineQuality("winequality-red.csv")
    whiteWines = processWineQuality("winequality-white.csv")
    allWines = np.concatenate((redWines, whiteWines))
    np.random.shuffle(allWines)
    trainingSet = allWines[:trainingSize]
    allWines = allWines[trainingSize:]
    testSet = allWines[:testSize]

    svm = SVM(lambdaVar=lambdaVar, trainingSet=trainingSet, testSet=testSet, delta=0, startingWeight=np.array([1]*(dimensions+1)))
    svm.learnLin()
    svm.test()

if __name__ == "__main__":
    #testForWineType(trainingSize=100, testSize=5000, dimensions=12, lambdaVar=0.01)
    testForQuality(trainingSize=100, testSize=2000, dimensions=11, lambdaVar=0.01)