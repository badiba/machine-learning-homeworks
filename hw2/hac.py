import matplotlib.pyplot as plt
import numpy as np
import copy
import os


class Criterion:
    SingleLinkage = 0
    CompleteLinkage = 1
    AverageLinkage = 2
    Centroid = 3


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

        return shortestDistance

    def CompleteLinkage(self, dataset, left, right):
        longestDistance = float('-inf')

        for i in left:
            for j in right:
                distance = self.GetDistanceBetween(dataset[i], dataset[j])
                if (distance > longestDistance):
                    longestDistance = distance

        return longestDistance

    def AverageLinkage(self, dataset, left, right):
        totalDistance = 0

        for i in left:
            for j in right:
                totalDistance += self.GetDistanceBetween(
                    dataset[i], dataset[j])

        return totalDistance / (len(left) * len(right))

    def Centroid(self, dataset, left, right):
        leftTotal = [0, 0]
        rightTotal = [0, 0]

        for i in left:
            leftTotal[0] = leftTotal[0] + dataset[i][0]
            leftTotal[1] = leftTotal[1] + dataset[i][1]

        for i in right:
            rightTotal[0] = rightTotal[0] + dataset[i][0]
            rightTotal[1] = rightTotal[1] + dataset[i][1]

        lc = [leftTotal[0] / len(left), leftTotal[1] / len(left)]
        rc = [rightTotal[0] / len(right), rightTotal[1] / len(right)]

        return self.GetDistanceBetween(lc, rc)

    def GetDistanceBetweenClusters(self, dataset, left, right, criterion):
        if (criterion == Criterion.SingleLinkage):
            return self.SingleLinkage(dataset, left, right)
        elif (criterion == Criterion.CompleteLinkage):
            return self.CompleteLinkage(dataset, left, right)
        elif (criterion == Criterion.AverageLinkage):
            return self.AverageLinkage(dataset, left, right)

        return self.Centroid(dataset, left, right)

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

    def PlotFinalCluster(self, dataset, clusters):
        xAxisClustered = []
        yAxisClustered = []
        colors = ['green', 'blue', 'black', 'red']

        for i in range(len(clusters)):
            xAxisClustered.append([])
            yAxisClustered.append([])

            for j in range(len(clusters[i])):
                index = clusters[i][j]
                xAxisClustered[i].append(dataset[index][0])
                yAxisClustered[i].append(dataset[index][1])

        for i in range(len(clusters)):
            plt.scatter(xAxisClustered[i],
                        yAxisClustered[i], s=2, color=colors[i])

        # naming the x axis
        plt.xlabel('x - axis')
        # naming the y axis
        plt.ylabel('y - axis')
        # giving a title to my graph
        plt.title('Two lines on same graph!')

        # show a legend on the plot
        plt.legend()

        # function to show the plot
        plt.show()


def main():
    hac = Hac()
    dede = hac.GetFinalClusters(hac._dataset2, Criterion.SingleLinkage, 2)
    hac.PlotFinalCluster(hac._dataset1, dede)


if __name__ == '__main__':
    main()
