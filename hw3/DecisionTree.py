import pickle
import os
import sys
from dt import divide, entropy, info_gain, gain_ratio, gini, avg_gini_index, chi_squared_test
from graphviz import Digraph


chiSquareTable = [-1, 2.71, 4.61, 6.25, 7.78,
                  9.24, 10.6, 12.0, 13.4, 14.7, 16.0]


def minIndex(lst):
    minimum = float("inf")
    minimumIndex = 0

    for i in range(len(lst)):
        if (lst[i] < minimum):
            minimum = lst[i]
            minimumIndex = i

    return minimumIndex


def maxIndex(lst):
    maximum = float("-inf")
    maximumIndex = 0

    for i in range(len(lst)):
        if (lst[i] > maximum):
            maximum = lst[i]
            maximumIndex = i

    return maximumIndex


class Node:
    def __init__(self):
        self._attribute = -1
        self._isLeaf = False
        self._value = ""
        self._children = []

    def SetNode(self, attribute):
        self._attribute = attribute
        self._isLeaf = False
        self._value = ""
        self._children = []

    def SetLeafNode(self, leafValue):
        self._attribute = -1
        self._isLeaf = True
        self._value = leafValue
        self._children = []


class DecisionTree:
    def __init__(self):
        dirname = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(dirname, "hw3_data", "dt", "data.pkl")

        with open(path, 'rb') as f:
            self._trainData, self._testData, self._attrValsList, self._attrNames = pickle.load(
                f)

        self._root = Node()

    def GetMostCommonLabel(self, data):
        labels = self._attrValsList[-1]
        labelCounts = [0] * len(labels)

        for example in data:
            exampleLabel = example[-1]
            labelIndex = labels.index(exampleLabel)
            labelCounts[labelIndex] += 1

        index = maxIndex(labelCounts)
        return labels[index]

    def IsAllSame(self, data):
        checkLabel = data[0][-1]
        for example in data:
            if (example[-1] != checkLabel):
                return False, -1

        return True, checkLabel

    def DecideOnAttribute(self, data, method, compare):
        attributePerformances = []
        attributeDivisions = []

        for i in range(len(self._attrNames)):
            performance, division = method(data, i, self._attrValsList)
            attributePerformances.append(performance)
            attributeDivisions.append(division)

        attrIndex = compare(attributePerformances)
        return attrIndex, attributeDivisions[attrIndex]

    def CreateTree(self, node, data, method, compare):
        isAllSame = self.IsAllSame(data)
        if (isAllSame[0]):
            node.SetLeafNode(isAllSame[1])
            return

        attrIndex, division = self.DecideOnAttribute(data, method, compare)
        node.SetNode(attrIndex)

        for subset in division:
            if (len(subset) == 0):
                childNode = Node()
                childNode.SetLeafNode(self.GetMostCommonLabel(data))
                node._children.append(childNode)

            else:
                childNode = Node()
                self.CreateTree(childNode, subset, method, compare)
                node._children.append(childNode)

    def CreatePrePrunedTree(self, node, data, method, compare):
        isAllSame = self.IsAllSame(data)
        if (isAllSame[0]):
            node.SetLeafNode(isAllSame[1])
            return

        attrIndex, division = self.DecideOnAttribute(data, method, compare)
        chiSquareValue, df = chi_squared_test(
            data, attrIndex, self._attrValsList)

        if (chiSquareValue < chiSquareTable[df]):
            node.SetLeafNode(self.GetMostCommonLabel(data))
            return

        node.SetNode(attrIndex)

        for subset in division:
            if (len(subset) == 0):
                childNode = Node()
                childNode.SetLeafNode(self.GetMostCommonLabel(data))
                node._children.append(childNode)

            else:
                childNode = Node()
                self.CreatePrePrunedTree(childNode, subset, method, compare)
                node._children.append(childNode)

    def Test(self, node, dataset):
        correctCount = 0
        for example in dataset:
            treeTraverser = node
            while (not treeTraverser._isLeaf):
                attrValue = example[treeTraverser._attribute]
                branchIndex = self._attrValsList[treeTraverser._attribute].index(
                    attrValue)

                treeTraverser = treeTraverser._children[branchIndex]

            if (treeTraverser._value == example[-1]):
                correctCount += 1

        return correctCount / len(dataset)

    def PostPruneTree(self, node, data, validationDataset):
        division = divide(data, node._attribute, self._attrValsList)

        for i in range(len(node._children)):
            if (not node._children[i]._isLeaf):
                self.PostPruneTree(
                    node._children[i], division[i], validationDataset)

        # change most common label
        for i in range(len(node._children)):
            if (not node._children[i]._isLeaf):
                currentChild = node._children[i]

                alternativeChild = Node()
                alternativeChild.SetLeafNode(
                    self.GetMostCommonLabel(division[i]))

                oldAcc = self.Test(self._root, validationDataset)

                node._children[i] = alternativeChild
                newAcc = self.Test(self._root, validationDataset)

                if (oldAcc > newAcc):
                    node._children[i] = currentChild

    def GetTreeSummary(self, node, summary):
        summary.append(node._attribute)

        for child in node._children:
            self.GetTreeSummary(child, summary)

    def PrintTreeHelper(self, node, dot, indices, currentIndex):
        if (node._isLeaf):
            return

        for child in node._children:
            childIndex = indices[-1] + 1
            indices.append(childIndex)
            dot.node(str(childIndex), self._attrNames[child._attribute])
            dot.edge(str(currentIndex), str(childIndex), "dede")
            self.PrintTreeHelper(child, dot, indices, childIndex)

    def PrintTree(self, node):
        if (node._isLeaf):
            print(node._value)
            return

        print(self._attrNames[node._attribute])
        for child in node._children:
            self.PrintTree(child)


def main():
    methodOne = info_gain, maxIndex
    methodTwo = gain_ratio, maxIndex
    methodThree = avg_gini_index, minIndex

    if (len(sys.argv) == 2):
        argument = sys.argv[1]

        decisionTree = DecisionTree()
        if (argument == "2-1"):
            decisionTree.CreateTree(
                decisionTree._root, decisionTree._trainData, methodOne[0], methodOne[1])

        elif (argument == "2-2"):
            decisionTree.CreateTree(
                decisionTree._root, decisionTree._trainData, methodTwo[0], methodTwo[1])

        elif (argument == "2-3"):
            decisionTree.CreateTree(
                decisionTree._root, decisionTree._trainData, methodThree[0], methodThree[1])

        elif (argument == "2-4"):
            decisionTree.CreatePrePrunedTree(
                decisionTree._root, decisionTree._trainData, methodTwo[0], methodTwo[1])

        elif (argument == "2-5"):
            decisionTree.CreateTree(
                decisionTree._root, decisionTree._trainData, methodTwo[0], methodTwo[1])

            trainSplitIndex = int(len(decisionTree._trainData) * 0.8)
            trainSet = decisionTree._trainData[:trainSplitIndex]
            validationSet = decisionTree._trainData[trainSplitIndex:]

            decisionTree.PostPruneTree(
                decisionTree._root, trainSet, validationSet)

        else:
            print("Wrong argument. See README file. Terminating...")
            exit()

        accuracy = decisionTree.Test(
            decisionTree._root, decisionTree._testData)
        print(accuracy)

    else:
        print("Wrong argument. See README file. Terminating...")
        exit()


if __name__ == '__main__':
    main()
