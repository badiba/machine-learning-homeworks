import math
import os
import string
from nb import vocabulary, train, test


def main():
    dirname = os.path.dirname(os.path.abspath(__file__))
    trainDataPath = os.path.join(dirname, "hw4_data", "news", "train_data.txt")
    trainLabelsPath = os.path.join(
        dirname, "hw4_data", "news", "train_labels.txt")

    testDataPath = os.path.join(dirname, "hw4_data", "news", "test_data.txt")
    testLabelsPath = os.path.join(
        dirname, "hw4_data", "news", "test_labels.txt")

    trainData = []
    with open(trainDataPath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.translate(str.maketrans('', '', string.punctuation))
            words = line.split(" ")
            words[-1] = words[-1].rstrip()
            trainData.append(words)

        f.close()

    trainLabels = []
    with open(trainLabelsPath, 'r') as f:
        trainLabels = f.read().splitlines()

        f.close()

    testData = []
    with open(testDataPath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.translate(str.maketrans('', '', string.punctuation))
            words = line.split(" ")
            words[-1] = words[-1].rstrip()
            testData.append(words)

        f.close()

    testLabels = []
    with open(testLabelsPath, 'r') as f:
        testLabels = f.read().splitlines()

        f.close()

    vocab = vocabulary(trainData)
    theta, pi = train(trainData, trainLabels, vocab)
    scores = test(theta, pi, vocab, testData)

    resultLabels = []
    for predictions in scores:
        winnerLabel = (float('-inf'), "None")
        for score in predictions:
            if (score[0] > winnerLabel[0]):
                winnerLabel = score

        resultLabels.append(winnerLabel[1])

    correctPredictions = 0
    for i in range(len(testLabels)):
        if (testLabels[i] == resultLabels[i]):
            correctPredictions += 1

    print("Accuracy: " + str(correctPredictions / len(testLabels)))


if __name__ == '__main__':
    main()
