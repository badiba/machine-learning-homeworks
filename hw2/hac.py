import matplotlib.pyplot as plt
import numpy as np
import copy
import os


class Hac:
    def __init__(self):
        dirname = os.path.dirname(os.path.abspath(__file__))
        self._dataset1 = np.load(os.path.join(
            dirname, "hw2_data", "hac", "data1.npy"))
        self._dataset2 = np.load(os.path.join(
            dirname, "hw2_data", "hac", "data2.npy"))
        self._dataset3 = np.load(os.path.join(
            dirname, "hw2_data", "hac", "data3.npy"))
        self._dataset4 = np.load(os.path.join(
            dirname, "hw2_data", "hac", "data4.npy"))

    def GetDistanceBetween(self, left, right):
        xSquared = (left[0] - right[0]) * (left[0] - right[0])
        ySquared = (left[1] - right[1]) * (left[1] - right[1])
        return xSquared + ySquared

    def SingleLinkage(self, dataset, left, right):
        shortestDistance = float('inf')

        for i in left:
            for j in right:
                distance = self.GetDistanceBetween(dataset[i], dataset[j])
                if (distance < shortestDistance):
                    shortestDistance = distance

        return distance

    def GetDistanceBetweenClusters(self, dataset, left, right, criterion):
        if (criterion == "SingleLinkage"):
            return self.SingleLinkage(dataset, left, right)

        return self.SingleLinkage(dataset, left, right)

    def GetFinalClusters(self, dataset, criterion, minK):
        clusters = []
        for i in range(len(dataset)):
            clusters.append([i])

        while len(clusters) > minK:
            shortestDistance = float('inf')
            pair = (-1, -1)

            for i in range(len(clusters)):
                j = i + 1

                while j < len(clusters):
                    distance = self.GetDistanceBetweenClusters(
                        dataset, clusters[i], clusters[j], criterion)

                    if (distance < shortestDistance):
                        shortestDistance = distance
                        pair = (i, j)

                    j += 1

            leftSize = len(clusters[pair[0]])
            rightSize = len(clusters[pair[1]])
            if (leftSize > rightSize):
                (clusters[pair[0]]).extend(clusters[pair[1]])
                del clusters[pair[1]]
            else:
                (clusters[pair[1]]).extend(clusters[pair[0]])
                del clusters[pair[0]]

        return clusters


def main():
    hac = Hac()
    dede = hac.GetFinalClusters(hac._dataset1, "SingleLinkage", 2)
    print(dede)


if __name__ == '__main__':
    main()
