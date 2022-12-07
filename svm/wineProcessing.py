import csv
import numpy as np

def processWine(file):
    with open(file, 'r') as rfile:
        reader = csv.DictReader(rfile, delimiter=';')
        points = []
        for row in reader:
            attributes = []
            for point in row.values():
                attributes.append(float(point))
            points.append(attributes)
        return points

def processWineType(file, isRed=True):
        points = processWine(file)
        wineClass = 1 if isRed else -1
        points = [(entry, wineClass) for entry in points]
        return np.array(points, dtype=object)

def processWineQuality(file, delimiter):
    points = processWine(file)
    pointsWithQuality = []
    for point in points:
        quality = -1 if point[-1] <= delimiter else 1
        pointsWithQuality.append((point[:-1], quality))
    return np.array(pointsWithQuality, dtype=object)