import matplotlib.pyplot as plt
import numpy as np
import random
import copy
import os


class Kmeans:
    def __init__(self):
        dirname = os.path.dirname(os.path.abspath(__file__))
        self._clustering1 = np.load(os.path.join(dirname,
                                                 "hw2_data", "kmeans", "clustering1.npy"))
        self._clustering2 = np.load(os.path.join(dirname,
                                                 "hw2_data", "kmeans", "clustering2.npy"))
        self._clustering3 = np.load(os.path.join(dirname,
                                                 "hw2_data", "kmeans", "clustering3.npy"))
        self._clustering4 = np.load(os.path.join(dirname,
                                                 "hw2_data", "kmeans", "clustering4.npy"))

        self._boundaries = [[float('inf'), float('-inf')],
                            [float('inf'), float('-inf')]]

    def SetBoundaries(self, dataset):
        self._boundaries = [[float('inf'), float('-inf')],
                            [float('inf'), float('-inf')]]

        for example in dataset:
            if (example[0] < self._boundaries[0][0]):
                self._boundaries[0][0] = example[0]

            if (example[0] > self._boundaries[0][1]):
                self._boundaries[0][1] = example[0]

            if (example[1] < self._boundaries[1][0]):
                self._boundaries[1][0] = example[1]

            if (example[1] > self._boundaries[1][1]):
                self._boundaries[1][1] = example[1]

    def InitializeClusterMeans(self, clusterCount):
        means = []

        for i in range(clusterCount):
            randomX = random.uniform(
                self._boundaries[0][0], self._boundaries[0][1])

            randomY = random.uniform(
                self._boundaries[1][0], self._boundaries[1][1])

            means.append([randomX, randomY])

        return means

    def GetDistanceBetween(self, instance, mean):
        xSquared = (instance[0] - mean[0]) * (instance[0] - mean[0])
        ySquared = (instance[1] - mean[1]) * (instance[1] - mean[1])
        return xSquared + ySquared

    def GetClosestMeanForInstance(self, instance, means):
        closestMeanIndex = -1
        closestDistance = float('inf')
        for i in range(len(means)):
            distance = self.GetDistanceBetween(instance, means[i])
            if (distance < closestDistance):
                closestDistance = distance
                closestMeanIndex = i

        return closestMeanIndex

    def GetOptimizedMeans(self, dataset, means):
        instanceAssignments = [-1] * len(dataset)
        currentMeans = copy.deepcopy(means)

        assignmentsKeepChanging = True
        while assignmentsKeepChanging:
            assignmentsKeepChanging = False

            for i in range(len(dataset)):
                assignment = self.GetClosestMeanForInstance(
                    dataset[i], currentMeans)

                if (assignment != instanceAssignments[i]):
                    assignmentsKeepChanging = True

                instanceAssignments[i] = assignment

            updatedMeans = []
            for i in range(len(means)):
                updatedMeans.append([0, 0])

            assignmentCounts = [0] * len(means)
            for i in range(len(dataset)):
                meanIndex = instanceAssignments[i]
                updatedMeans[meanIndex][0] += dataset[i][0]
                updatedMeans[meanIndex][1] += dataset[i][1]
                assignmentCounts[meanIndex] += 1

            for i in range(len(updatedMeans)):
                if (assignmentCounts[i] == 0):
                    continue

                updatedMeans[i][0] /= assignmentCounts[i]
                updatedMeans[i][1] /= assignmentCounts[i]

            currentMeans = copy.deepcopy(updatedMeans)

        return instanceAssignments, currentMeans

    def CalculateObjectiveFunction(self, dataset, instanceAssignments, optimizedMeans):
        totalDistance = 0
        for i in range(len(dataset)):
            assignedMean = optimizedMeans[instanceAssignments[i]]
            totalDistance += self.GetDistanceBetween(dataset[i], assignedMean)

        return totalDistance

    def Cluster(self, dataset, clusterCount):
        i = 0
        tof = 0
        restartCount = 10
        while i < restartCount:
            means = self.InitializeClusterMeans(clusterCount)
            instanceAssignments, optimizedMeans = self.GetOptimizedMeans(
                dataset, means)

            tof += self.CalculateObjectiveFunction(
                dataset, instanceAssignments, optimizedMeans)

            i += 1

        return tof / restartCount

    def Train(self, dataset, maxK):
        self.SetBoundaries(dataset)

        i = 1
        ofs = []
        while i <= maxK:
            ofs.append(self.Cluster(dataset, i))

            i += 1
            print(i)

        x = list(range(1, maxK + 1))
        plt.plot(x, ofs)

        plt.xlabel('K (Number of clusters)')
        plt.ylabel('Objective function')
        plt.title('K versus Objective function')
        plt.show()

    def PlotClusteringWithChoosenK(self, dataset, clusterCount):
        self.SetBoundaries(dataset)

        i = 0
        mof = float('inf')
        mofInstanceAssignments = []
        mofOptimizedMeans = []
        restartCount = 10
        while i < restartCount:
            means = self.InitializeClusterMeans(clusterCount)
            instanceAssignments = []
            instanceAssignments, optimizedMeans = self.GetOptimizedMeans(
                dataset, means)

            of = self.CalculateObjectiveFunction(
                dataset, instanceAssignments, optimizedMeans)

            if (of < mof):
                mof = of
                mofInstanceAssignments = instanceAssignments
                mofOptimizedMeans = optimizedMeans

            i += 1

        xAxisClustered = []
        yAxisClustered = []
        colors = ['green', 'blue', 'cyan', 'black', 'yellow', 'red']
        for j in range(clusterCount):
            xAxisClustered.append([])
            yAxisClustered.append([])

        for j in range(len(dataset)):
            clusterIndex = mofInstanceAssignments[j]
            xAxisClustered[clusterIndex].append(dataset[j][0])
            yAxisClustered[clusterIndex].append(dataset[j][1])

        plt.axis([-1.5, 1.5, -1, 1])
        for j in range(clusterCount):
            plt.scatter(xAxisClustered[j],
                        yAxisClustered[j], s=0.8, color=colors[j])

        for mean in mofOptimizedMeans:
            plt.scatter([mean[0]], [mean[1]], s=30, color=colors[-1])

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

    def Debug(self, param=[]):
        gede = [2, 2, 2]
        param = gede


def main():
    kmeans = Kmeans()
    kmeans.PlotClusteringWithChoosenK(kmeans._clustering4, 5)
    #kmeans.Train(kmeans._clustering1, 10)


if __name__ == '__main__':
    main()
