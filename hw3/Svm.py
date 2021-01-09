import numpy as np
import os
import sys
from draw import draw_svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pandas as pd
import random


class SVM:
    def __init__(self, extraLoad=None):
        dirname = os.path.dirname(os.path.abspath(__file__))
        self._trainDataSep = np.load(os.path.join(
            dirname, "hw3_data", "linsep", "train_data.npy"))
        self._trainLabelsSep = np.load(os.path.join(
            dirname, "hw3_data", "linsep", "train_labels.npy"))
        self._trainDataNonSep = np.load(os.path.join(
            dirname, "hw3_data", "nonlinsep", "train_data.npy"))
        self._trainLabelsNonSep = np.load(os.path.join(
            dirname, "hw3_data", "nonlinsep", "train_labels.npy"))

        if (extraLoad != None):
            self._trainData = np.load(os.path.join(
                dirname, "hw3_data", extraLoad, "train_data.npy"))
            self._trainLabels = np.load(os.path.join(
                dirname, "hw3_data", extraLoad, "train_labels.npy"))
            self._testData = np.load(os.path.join(
                dirname, "hw3_data", extraLoad, "test_data.npy"))
            self._testLabels = np.load(os.path.join(
                dirname, "hw3_data", extraLoad, "test_labels.npy"))

            normalizationFactor = (2.0 / 255.0)

            self._trainData = self._trainData.astype(np.float)
            for example in self._trainData:
                for i in range(len(example)):
                    example[i] = example[i] * normalizationFactor - 1

            self._testData = self._testData.astype(np.float)
            for example in self._testData:
                for i in range(len(example)):
                    example[i] = example[i] * normalizationFactor - 1

        self._kernels = ["linear", "rbf", "poly", "sigmoid"]
        self._cValues = [0.01, 0.1, 1, 10, 100]
        self._gamma = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]

    def TrainLinearlySeparable(self):
        dirname = os.path.dirname(os.path.abspath(__file__))

        cValues = [0.01, 0.1, 1, 10, 100]
        for i in range(len(cValues)):
            svc = SVC(kernel='linear', C=cValues[i])
            svc.fit(self._trainDataSep, self._trainLabelsSep)

            savePath = os.path.join(dirname, "3-1-" + str(i))
            draw_svm(svc, self._trainDataSep,
                     self._trainLabelsSep, -3, 3, -3, 3, savePath)

    def TrainLinearlyNonSeparable(self):
        dirname = os.path.dirname(os.path.abspath(__file__))

        for kernel in self._kernels:
            svc = SVC(kernel=kernel)
            svc.fit(self._trainDataSep, self._trainLabelsSep)

            savePath = os.path.join(dirname, "3-2-" + kernel)
            draw_svm(svc, self._trainDataSep,
                     self._trainLabelsSep, -3, 3, -3, 3, savePath)

    def TrainCatDog(self):
        parameters = {'kernel': self._kernels,
                      'C': self._cValues, 'gamma': self._gamma}
        svc = SVC()
        clf = GridSearchCV(svc, parameters)
        clf.fit(self._trainData, self._trainLabels)

        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        print(pd.concat([pd.DataFrame(clf.cv_results_["params"]), pd.DataFrame(
            clf.cv_results_["mean_test_score"], columns=["Accuracy"])], axis=1))

    def TestCatDog(self):
        svc = SVC(C=0.01, gamma=0.01000, kernel="poly")
        svc.fit(self._trainData, self._trainLabels)
        predictions = svc.predict(self._testData)

        correctPredictions = 0
        for i in range(len(predictions)):
            if (predictions[i] == self._testLabels[i]):
                correctPredictions += 1

        print(correctPredictions / len(predictions))

    def PrintConfusionMatrix(self, tz, to, fz, fo):
        print("True Zeroes: " + str(tz))
        print("True Ones: " + str(to))
        print("False Zeroes: " + str(fz))
        print("False Ones: " + str(fo))

    def GetTestAccuracy(self, predictions):
        correctPredictions = 0
        trueZero = 0
        trueOne = 0
        falseZero = 0
        falseOne = 0

        for i in range(len(predictions)):
            if (predictions[i] == self._testLabels[i]):
                if (predictions[i] == 0):
                    trueZero += 1
                else:
                    trueOne += 1

                correctPredictions += 1

            else:
                if (predictions[i] == 0):
                    falseZero += 1
                else:
                    falseOne += 1

        accuracy = correctPredictions / len(predictions)
        return accuracy, trueZero, trueOne, falseZero, falseOne

    def PlainTraining(self):
        svc = SVC(C=1, kernel="rbf")
        svc.fit(self._trainData, self._trainLabels)
        predictions = svc.predict(self._testData)

        acc, tz, to, fz, fo = self.GetTestAccuracy(predictions)

        print("Test accuracy after training without handling the imbalance problem: " + str(acc))
        self.PrintConfusionMatrix(tz, to, fz, fo)

    def OverSampleTrain(self, zeroIndicies, oneIndicies):
        zeroCount = len(zeroIndicies)
        oneCount = len(oneIndicies)

        overSampledTrainSet = []
        overSampledTrainLabels = []

        for example in self._trainData:
            overSampledTrainSet.append(example)

        for label in self._trainLabels:
            overSampledTrainLabels.append(label)

        for i in range(oneCount - zeroCount):
            randomIndex = random.randint(0, zeroCount - 1)
            addIndex = zeroIndicies[randomIndex]
            overSampledTrainSet.insert(addIndex, self._trainData[addIndex])
            overSampledTrainLabels.insert(addIndex, 0)

        svc = SVC(C=1, kernel="rbf")
        svc.fit(overSampledTrainSet, overSampledTrainLabels)
        predictions = svc.predict(self._testData)

        acc, tz, to, fz, fo = self.GetTestAccuracy(predictions)
        print("Test accuracy after training with oversampling minority class: " + str(acc))
        self.PrintConfusionMatrix(tz, to, fz, fo)

    def UnderSampleTrain(self, zeroIndicies, oneIndicies):
        underSampledTrainSet = []
        underSampledTrainLabels = []

        threshold = len(zeroIndicies) / len(oneIndicies)
        for i in range(len(self._trainLabels)):
            if (self._trainLabels[i] == 0):
                underSampledTrainSet.append(self._trainData[i])
                underSampledTrainLabels.append(self._trainLabels[i])
            else:
                p = random.uniform(0, 1)
                if (p < threshold):
                    underSampledTrainSet.append(self._trainData[i])
                    underSampledTrainLabels.append(self._trainLabels[i])

        svc = SVC(C=1, kernel="rbf")
        svc.fit(underSampledTrainSet, underSampledTrainLabels)
        predictions = svc.predict(self._testData)

        acc, tz, to, fz, fo = self.GetTestAccuracy(predictions)
        print("Test accuracy after training with undersampling majority class: " + str(acc))
        self.PrintConfusionMatrix(tz, to, fz, fo)

    def BalancedTrain(self):
        svc = SVC(C=1, kernel="rbf", class_weight="balanced")
        svc.fit(self._trainData, self._trainLabels)
        predictions = svc.predict(self._testData)

        acc, tz, to, fz, fo = self.GetTestAccuracy(predictions)

        print("Test accuracy after training with class_weight=balanced: " + str(acc))
        self.PrintConfusionMatrix(tz, to, fz, fo)

    def TrainCatDogImba(self):
        zeroIndicies = []
        oneIndicies = []
        for i in range(len(self._trainLabels)):
            if (self._trainLabels[i] == 0):
                zeroIndicies.append(i)
            elif (self._trainLabels[i] == 1):
                oneIndicies.append(i)

        self.PlainTraining()
        self.OverSampleTrain(zeroIndicies, oneIndicies)
        self.UnderSampleTrain(zeroIndicies, oneIndicies)
        self.BalancedTrain()


def main():
    if (len(sys.argv) == 2):
        argument = sys.argv[1]

        if (argument == "3-1"):
            supportVectorMachine = SVM()
            supportVectorMachine.TrainLinearlySeparable()
            print("Figures are saved to the execution path.")

        elif (argument == "3-2"):
            supportVectorMachine = SVM()
            supportVectorMachine.TrainLinearlyNonSeparable()
            print("Figures are saved to the execution path.")

        elif (argument == "3-4"):
            supportVectorMachine = SVM("catdogimba")
            supportVectorMachine.TrainCatDogImba()

        else:
            print("Wrong argument. See README file. Terminating...")
            exit()

    elif (len(sys.argv) == 3):
        argOne = sys.argv[1]
        argTwo = sys.argv[2]

        if (argOne == "3-3" and argTwo == "train"):
            supportVectorMachine = SVM("catdog")
            supportVectorMachine.TrainCatDog()
        elif (argOne == "3-3" and argTwo == "test"):
            supportVectorMachine = SVM("catdog")
            supportVectorMachine.TestCatDog()

        else:
            print("Wrong argument. See README file. Terminating...")
            exit()

    else:
        print("Wrong argument. See README file. Terminating...")
        exit()


if __name__ == '__main__':
    main()
