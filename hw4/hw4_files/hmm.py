import numpy as np


def forward(A, B, pi, O):
    """
    Calculates the probability of an observation sequence O given the model(A, B, pi).
    :param A: state transition probabilities (NxN)
    :param B: observation probabilites (NxM)
    :param pi: initial state probabilities (N)
    :param O: sequence of observations(T) where observations are just indices for the columns of B (0-indexed)
        N is the number of states,
        M is the number of possible observations, and
        T is the sequence length.
    :return: The probability of the observation sequence and the calculated alphas in the Trellis diagram with shape
             (N, T) which should be a numpy array.
    """

    N = len(pi)
    T = len(O)

    alpha = np.zeros((N, T))

    for state in range(N):
        alpha[state][0] = pi[state] * B[state][O[0]]

    i = 1
    while (i < T):
        for j in range(N):
            nodeValue = 0

            for k in range(N):
                nodeValue += alpha[k][i - 1] * A[k][j] * B[j][O[i]]

            alpha[j][i] = nodeValue

        i += 1

    forwardResult = 0
    for i in range(N):
        forwardResult += alpha[i][T - 1]

    return forwardResult, alpha


def viterbi(A, B, pi, O):
    """
    Calculates the most likely state sequence given model(A, B, pi) and observation sequence.
    :param A: state transition probabilities (NxN)
    :param B: observation probabilites (NxM)
    :param pi: initial state probabilities(N)
    :param O: sequence of observations(T) where observations are just indices for the columns of B (0-indexed)
        N is the number of states,
        M is the number of possible observations, and
        T is the sequence length.
    :return: The most likely state sequence with shape (T,) and the calculated deltas in the Trellis diagram with shape
             (N, T). They should be numpy arrays.
    """
