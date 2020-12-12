import numpy as np


def SortTrainingPointDistances(value):
    return value[0]


class Knn:
    def __init__(self, foldCount, classCount):
        self._trainData = np.split(
            np.load("hw2_data/knn/train_data.npy"), foldCount)
        self._trainLabels = np.split(
            np.load("hw2_data/knn/train_labels.npy"), foldCount)

        self._testData = np.load("hw2_data/knn/test_data.npy")
        self._testLabels = np.load("hw2_data/knn/test_labels.npy")

        self._classCount = classCount

    def GetDistance(self, example, trainPoint):
        a = (example[0] - trainPoint[0]) * (example[0] - trainPoint[0])
        b = (example[1] - trainPoint[1]) * (example[1] - trainPoint[1])
        c = (example[2] - trainPoint[2]) * (example[2] - trainPoint[2])
        d = (example[3] - trainPoint[3]) * (example[3] - trainPoint[3])

        return a + b + c + d

    def MakePrediction(self, closestTrainPoints):
        majorityCount = [0] * self._classCount

        for trainPoint in closestTrainPoints:
            majorityCount[trainPoint[1]] += 1

        winnerLabel = 0
        winnerLabelCount = majorityCount[0]
        for i in range(len(majorityCount)):
            if (majorityCount[i] > winnerLabelCount):
                winnerLabelCount = majorityCount[i]
                winnerLabel = i

        return winnerLabel

    def GetAccuracyOfExample(self, foldIndex, k, example):
        trainPointDistances = []

        for i in range(len(self._trainData)):
            if (i == foldIndex):
                continue

            for j in range(len(self._trainData[i])):
                distance = self.GetDistance(example[0], self._trainData[i][j])
                trainPointDistances.append((distance, self._trainLabels[i][j]))

        trainPointDistances.sort(key=SortTrainingPointDistances)
        trainPointDistances = trainPointDistances[:k]

        exampleLabel = example[1]
        prediction = self.MakePrediction(trainPointDistances)
        if (exampleLabel == prediction):
            return 1

        return 0

    def GetAccuracyOfFold(self, foldIndex, k):
        accuracy = 0
        exampleCountInFold = len(self._trainData[foldIndex])
        for i in range(exampleCountInFold):
            example = (self._trainData[foldIndex][i],
                       self._trainLabels[foldIndex][i])

            accuracy += self.GetAccuracyOfExample(foldIndex, k, example)

        return accuracy / exampleCountInFold

    def Train(self, k):
        accuracy = 0
        foldCount = len(self._trainData)
        for i in range(foldCount):
            accuracy += self.GetAccuracyOfFold(i, k)

        return accuracy / foldCount

    def Debug(self):
        print("debugging...")


def main():
    knn = Knn(10, 3)

    for i in range(200):
        print(knn.Train(i))


if __name__ == '__main__':
    main()
