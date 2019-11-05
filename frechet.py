import matplotlib.pyplot as plt
import numpy as np


def n_max(*args):
    m = sorted(range(len(args)), key=lambda x: args[x][0], reverse=True)[0]
    return args[m]


def n_min(*args):
    m = sorted(range(len(args)), key=lambda x: args[x][0])[0]
    return args[m]


def distance_frechet(P, Q):
    """
    Function for finding Frechet distance between two lines.
    :param P: ndarray
        one of the lines
    :param Q: ndarray
        one of the lines
    :return distance: float
        value of the Frechet distance
    :return i: int
        index of the point of P at which the distance is reached
    :return j: int
        index of the point of Q at which the distance is reached
    """

    def c(i, j):
        n_i = i
        n_j = j
        if ca[i][j] > -1:
            return ca[i][j], n_i, n_j
        elif i == 0 and j == 0:
            ca[i][j] = np.linalg.norm(P[0] - Q[0])
        elif i > 0 and j == 0:
            ca[i][j], n_i, n_j = n_max(c(i - 1, 0), (np.linalg.norm(P[i] - Q[0]), i, 0))
        elif i == 0 and j > 0:
            ca[i][j], n_i, n_j = n_max(c(0, j - 1), (np.linalg.norm(P[0] - Q[j]), 0, j))
        elif i > 0 and j > 0:
            ca[i][j], n_i, n_j = n_max(n_min(c(i - 1, j), c(i - 1, j - 1), c(i, j - 1)),
                                       (np.linalg.norm(P[i] - Q[j]), i, j))
        else:
            ca[i][j] = float('inf')
        return ca[i][j], n_i, n_j

    p = len(P)
    q = len(Q)
    ca = np.full((p, q), -1.0)
    return c(p - 1, q - 1)


def draw_plot(P, Q, points):
    """
    Function for drawing two lines with Frechet distance between it.
    :param P: ndarray
        one of the lines
    :param Q: ndarray
        one of the lines
    :param points: array_like
        two digits - indexes of the points at which the distance is reached
    """

    plt.figure()
    plt.plot(P[:, 0], P[:, 1], color='blue')
    plt.plot(Q[:, 0], Q[:, 1], color='orange')
    plt.plot([P[points[0], 0], Q[points[1], 0]], [P[points[0], 1], Q[points[1], 1]], color='red')
    plt.legend(['P', 'Q', 'Frechet distance'])
    plt.grid(True)
    plt.show()
