import numpy as np
import os
import sys
from draw import draw_svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pandas as pd


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

    def Train(self):
        parameters = {'kernel': self._kernels, 'C': self._cValues, 'gamma': self._gamma}
        svc = SVC()
        clf = GridSearchCV(svc, parameters)
        clf.fit(self._trainData, self._trainLabels)
        
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        print(pd.concat([pd.DataFrame(clf.cv_results_["params"]), pd.DataFrame(
            clf.cv_results_["mean_test_score"], columns=["Accuracy"])], axis=1))


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

        elif (argument == "catdog" or argument == "catdogimba"):
            supportVectorMachine = SVM(argument)
            supportVectorMachine.Train()
            
        else:
            print("Wrong argument. See README file. Terminating...")
            exit()
    else:
        print("Wrong argument. See README file. Terminating...")
        exit()



if __name__ == '__main__':
    main()
