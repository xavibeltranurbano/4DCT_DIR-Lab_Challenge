# # -----------------------------------------------------------------------------
# # Registration Evaluation File
# # Author: Xavier Beltran Urbano and Frederik Hartmann
# # Date Created: 12-12-2023
# # -----------------------------------------------------------------------------

import itk
from utils import Utils
import os
import numpy as np

class Evaluation:
    util = Utils()

    def __init__(self):
        pass

    def extractOutputPoints(self, pointFilePath, outputFilePath):
        # reads an elastix point file from pointFilePath and stores OutputIndexFixed in outputFilePath     
        with open(pointFilePath, 'r') as inputFile, open(outputFilePath, 'w') as outputFile:
            for line in inputFile:
                segments = line.split(';')
                for segment in segments:
                    if 'OutputIndexFixed =' in segment:
                        start = segment.find('[') + 1
                        end = segment.find(']')
                        outputPointData = segment[start:end].split()
                        outputPoint = ' '.join(outputPointData)
                        outputFile.write(outputPoint + '\n')
                        break  
    
    @staticmethod
    def readPointsFromFile(pointFilePath):
        # reads points in pointFilePath and returns a numpy array
        with open(pointFilePath, 'r') as file:
            lines = file.readlines()
            if "point" in lines[0].lower() or "index" in lines[0].lower():
                lines = lines[2:]
            points = [list(map(float, line.split())) for line in lines]
        return points

    @staticmethod
    def normalizePoints(points, spacing):
        # normalizes a list of points. Spacing is expected to have same shape as a single point. e.g. spacing=[0.625, 0.625, 2.5] for point [x, y, z]
        return [np.array(point) * np.array(spacing) for point in points]

    @staticmethod
    def targetRegistrationError(pointSet1, pointSet2):
        # returns the average target registration error between pointSet1 and pointSet2 using the euclidean norm
        treValues = []
        for p1, p2 in zip(pointSet1, pointSet2):
            tre = np.linalg.norm(np.array(p1) - np.array(p2))
            treValues.append(tre)
        return np.mean(treValues)

    

if __name__ == "__main__":
    # INPUTS
    pointFilePath = "/PATH/TO/ELASTIX/POINT/FILE.txt"
    outputFilePath = "/PATH/TO/SAVE/EXTRACTED/POINTS.txt"
    pointSetPath1 =  "/PATH/TO/LOAD/EXTRACTED/POINTS.txt"
    pointSetPath2 =  "/PATH/TO/GROUND/TRUTH.txt"
    spacing = [0.625, 0.625, 2.5]

    # COMPUTATION
    evaluation = Evaluation()
    evaluation.extractOutputPoints(pointFilePath, outputFilePath)
    
    pointSet1 = evaluation.readPointsFromFile(pointSetPath1)
    pointSet2 = evaluation.readPointsFromFile(pointSetPath2)

    normalizedPointSet1 = evaluation.normalizePoints(pointSet1, spacing)
    normalizedPointSet2 = evaluation.normalizePoints(pointSet2, spacing)

    tre = evaluation.targetRegistrationError(normalizedPointSet1, normalizedPointSet2)
    print(tre)

