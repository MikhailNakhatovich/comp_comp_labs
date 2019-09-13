import matplotlib.pyplot as plt
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


def draw_plot(P, Q, points):
    plt.figure()
    plt.plot(P[:, 0], P[:, 1], color='blue')
    plt.plot(Q[:, 0], Q[:, 1], color='orange')
    plt.plot([P[points[0]][0], Q[points[1]][0]], [P[points[0]][1], Q[points[1]][1]], color='red')
    plt.legend(['P', 'Q', 'Frechet distance'])
    plt.grid(True)
    plt.show()


def run_test():
    P = np.array([(0, 0), (4, 2), (6, 5), (12, 6), (15, 7), (15, 10), (18, 13)])
    Q = np.array([(1, 1), (2, 5), (7, 7), (8, 12), (13, 14), (15, 16)])

    P_star = np.array([(2, 2), (3, 4), (2, 7), (5, 6), (9, 8), (8, 5), (10, 1), (6, 3), (2, 2)])
    Q_star = np.array([(12, 1), (10, 3), (6, 6), (9, 7), (10, 9), (12, 6), (15, 5), (13, 3), (12, 1)])

    dist, i, j = distance_frechet(P, Q)
    dist_star, i_star, j_star = distance_frechet(P_star, Q_star)

    print("P:\n%s\nQ:\n%s\ndF(P, Q) = %f" % (str(P), str(Q), dist))
    draw_plot(P, Q, (i, j))
    print("\n")
    print("P:\n%s\nQ:\n%s\ndF(P, Q) = %f" % (str(P_star), str(Q_star), dist_star))
    draw_plot(P_star, Q_star, (i_star, j_star))
