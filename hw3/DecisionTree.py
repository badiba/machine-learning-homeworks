import pickle
import os
from dt import divide, entropy, info_gain, gain_ratio, gini, avg_gini_index, chi_squared_test
from graphviz import Digraph


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

    def Test(self, node):
        correctCount = 0
        for example in self._testData:
            treeTraverser = node
            while (not treeTraverser._isLeaf):
                attrValue = example[treeTraverser._attribute]
                branchIndex = self._attrValsList[treeTraverser._attribute].index(
                    attrValue)

                treeTraverser = treeTraverser._children[branchIndex]

            if (treeTraverser._value == example[-1]):
                correctCount += 1

        return correctCount / len(self._testData)

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

    decisionTree = DecisionTree()
    decisionTree.CreateTree(
        decisionTree._root, decisionTree._trainData, methodThree[0], methodThree[1])

    #summary = []
    #decisionTree.GetTreeSummary(decisionTree._root, summary)
    # print(summary)

    accuracy = decisionTree.Test(decisionTree._root)
    print(accuracy)

    # decisionTree.PrintTree(decisionTree._root)


if __name__ == '__main__':
    main()
