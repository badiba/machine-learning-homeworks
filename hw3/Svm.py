import numpy as np
import os
from draw import draw_svm
from sklearn.svm import SVC


class SVM:
    def __init__(self):
        dirname = os.path.dirname(os.path.abspath(__file__))
        self._trainDataSep = np.load(os.path.join(
            dirname, "hw3_data", "linsep", "train_data.npy"))
        self._trainLabelsSep = np.load(os.path.join(
            dirname, "hw3_data", "linsep", "train_labels.npy"))
        self._trainDataNonSep = np.load(os.path.join(
            dirname, "hw3_data", "nonlinsep", "train_data.npy"))
        self._trainLabelsNonSep = np.load(os.path.join(
            dirname, "hw3_data", "nonlinsep", "train_labels.npy"))

        self._kernels = ["linear", "rbf", "poly", "sigmoid"]

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


def main():
    supportVectorMachine = SVM()
    supportVectorMachine.TrainLinearlySeparable()


if __name__ == '__main__':
    main()
