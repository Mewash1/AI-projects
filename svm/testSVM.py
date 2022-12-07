import numpy as np
import csv
from wineProcessing import *
from svm import SVM

def testForQuality(trainingSize, testSize, lambdaVar, delta, delimiter, RBF=True):
    redWines = processWineQuality("winequality-red.csv", delimiter=delimiter)
    whiteWines = processWineQuality("winequality-white.csv", delimiter=delimiter)
    allWines = np.concatenate((redWines, whiteWines))
    np.random.shuffle(allWines)
    trainingSet = allWines[:trainingSize]
    allWines = allWines[trainingSize:]
    testSet = allWines[:testSize]

    svm = SVM(lambdaVar=lambdaVar, trainingSet=trainingSet, testSet=testSet, delta=delta, startingAlpha=np.array([1]*(len(trainingSet))))
    if RBF:
        svm.learnRBFkernel()
        return svm.test(rbf=True)
    else:
        svm.learnPolynomialKernel()
        return svm.test(rbf=False)
    
def generateResultsFromCsv(filename):
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        for field in fieldnames[:-1]:
            effectivenessField = {}
            for row in reader:
                if row[field] not in effectivenessField:
                    effectivenessField[row[field]] = list()
                effectivenessField[row[field]].append(float(row['effectiveness']))
            print("\n", field, "\n")
            for key, value in effectivenessField.items():
                print(f"{key} - effectiveness {round(sum(value)/len(value),2)}%")
            file.seek(0)
            file.readline()
        effectivenessList = []
        for row in reader:
            effectivenessList.append(float(row['effectiveness']))
        print(f"\nGeneral effectiveness: {round(sum(effectivenessList)/len(effectivenessList),2)}%")


if __name__ == "__main__":

    testSize = 6000
    delimiters = [3,5,7]
    lambdas = [0.01,0.1,1]
    deltas = [1,10,100]
    trainingSizes= [100]
    generateRBF = False
    generatePolynomial = False
    generateLinear = True

    if generateRBF:
        with open("rbf.csv", 'w') as file:

            writer = csv.DictWriter(file, fieldnames=["delimiter", "lambda", "delta", "training set size", "effectiveness"])
            writer.writeheader()

            for lambdaVar in lambdas:
                for delta in deltas:
                    for trainingSize in trainingSizes:
                        for delimiter in delimiters:
                            percent = testForQuality(trainingSize=trainingSize, testSize=6000, lambdaVar=0.01, delta=delta, delimiter=delimiter, RBF=True)
                            writer.writerow({"delimiter":delimiter, "lambda":lambdaVar, "delta":delta, "training set size":trainingSize, "effectiveness":round(percent, 2)})
    
    if generatePolynomial:
        with open("polynomial.csv", 'w') as file:

            writer = csv.DictWriter(file, fieldnames=["delimiter", "lambda", "delta", "training set size", "effectiveness"])
            writer.writeheader()

            for lambdaVar in lambdas:
                for delta in deltas:
                    for trainingSize in trainingSizes:
                        for delimiter in delimiters:
                            percent = testForQuality(trainingSize=trainingSize, testSize=6000, lambdaVar=0.01, delta=delta, delimiter=delimiter, RBF=False)
                            writer.writerow({"delimiter":delimiter, "lambda":lambdaVar, "delta":delta, "training set size":trainingSize, "effectiveness":round(percent, 2)})
    
    if generateLinear:
        with open("linear.csv", 'w') as file:

            writer = csv.DictWriter(file, fieldnames=["delimiter", "lambda", "delta", "training set size", "effectiveness"])
            writer.writeheader()

            for lambdaVar in lambdas:
                for trainingSize in trainingSizes:
                    for delimiter in delimiters:
                        percent = testForQuality(trainingSize=trainingSize, testSize=6000, lambdaVar=0.01, delta=1, delimiter=delimiter, RBF=False)
                        writer.writerow({"delimiter":delimiter, "lambda":lambdaVar, "delta":1, "training set size":trainingSize, "effectiveness":round(percent, 2)})
