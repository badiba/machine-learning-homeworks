import math
import os
import string


def vocabulary(data):
    """
    Creates the vocabulary from the data.
    :param data: List of lists, every list inside it contains words in that sentence.
                 len(data) is the number of examples in the data.
    :return: Set of words in the data
    """

    # add text preprocessing

    vocab = set()

    for example in data:
        for word in example:
            if (not word in vocab):
                vocab.add(word)

    return vocab


def train(train_data, train_labels, vocab):
    """
    Estimates the probability of a specific word given class label using additive smoothing with smoothing constant 1.
    :param train_data: List of lists, every list inside it contains words in that sentence.
                       len(train_data) is the number of examples in the training data.
    :param train_labels: List of class names. len(train_labels) is the number of examples in the training data.
    :param vocab: Set of words in the training set.
    :return: theta, pi. theta is a dictionary of dictionaries. At the first level, the keys are the class names. At the
             second level, the keys are all of the words in vocab and the values are their estimated probabilities.
             pi is a dictionary. Its keys are class names and values are their probabilities.
    """

    # add text preprocessing

    pi = {}
    theta = {}
    classTotalWordCount = {}
    for label in train_labels:
        theta[label] = {}
        pi[label] = 0
        classTotalWordCount[label] = 0

        for word in vocab:
            theta[label][word] = 1

    for i in range(len(train_data)):
        label = train_labels[i]
        pi[label] += 1

        for word in train_data[i]:
            theta[label][word] += 1
            classTotalWordCount[label] += 1

    for label in theta:
        for wordProbability in theta[label]:
            division = classTotalWordCount[label] + len(vocab)
            theta[label][wordProbability] /= division

    for label in pi:
        pi[label] /= len(train_data)

    return theta, pi


def test(theta, pi, vocab, test_data):
    """
    Calculates the scores of a test data given a class for each class. Skips the words that are not occurring in the
    vocabulary.
    :param theta: A dictionary of dictionaries. At the first level, the keys are the class names. At the second level,
                  the keys are all of the words in vocab and the values are their estimated probabilities.
    :param pi: A dictionary. Its keys are class names and values are their probabilities.
    :param vocab: Set of words in the training set.
    :param test_data: List of lists, every list inside it contains words in that sentence.
                      len(test_data) is the number of examples in the test data.
    :return: scores, list of lists. len(scores) is the number of examples in the test set. Every inner list contains
             tuples where the first element is the score and the second element is the class name.
    """

    # add text preprocessing

    scores = []

    for example in test_data:
        scores.append([])
        si = len(scores) - 1

        wordCounts = {}
        for word in vocab:
            wordCounts[word] = 0

        for word in example:
            if (word in vocab):
                wordCounts[word] += 1

        for label in theta:
            score = math.log(pi[label])

            for word in theta[label]:
                score += wordCounts[word] * math.log(theta[label][word])

            scores[si].append((score, label))

    return scores


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
