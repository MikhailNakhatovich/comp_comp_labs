import numpy as np


def n_max(*args):
    m = sorted(range(len(args)), key=lambda x: args[x][0], reverse=True)[0]
    return args[m]


def n_min(*args):
    m = sorted(range(len(args)), key=lambda x: args[x][0])[0]
    return args[m]


def distance_frechet(P, Q):
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
