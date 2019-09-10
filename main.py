import matplotlib.pyplot as plt
import numpy as np

from frechet import distance_frechet


def draw_plot(P, Q, points):
    plt.figure()
    plt.plot(P[:, 0], P[:, 1], color='blue')
    plt.plot(Q[:, 0], Q[:, 1], color='orange')
    plt.plot([P[points[0]][0], Q[points[1]][0]], [P[points[0]][1], Q[points[1]][1]], color='red')
    plt.legend(['A', 'B', 'Frechet distance'])
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    A = np.array([(0, 0), (4, 2), (6, 5), (12, 6), (15, 7), (15, 10), (18, 13)])
    B = np.array([(1, 1), (2, 5), (7, 7), (8, 12), (13, 14), (15, 16)])

    A_star = np.array([(2, 2), (3, 4), (2, 7), (5, 6), (9, 8), (8, 5), (10, 1), (6, 3)])
    B_star = np.array([(12, 1), (10, 3), (6, 6), (9, 7), (10, 9), (12, 6), (15, 5), (13, 3)])

    dist, i, j = distance_frechet(A, B)
    dist_star, i_star, j_star = distance_frechet(A_star, B_star)

    print("A:\n%s\nB:\n%s\ndF(A, B) = %f" % (str(A), str(B), dist))
    draw_plot(A, B, (i, j))
    print("\n")
    print("A:\n%s\nB:\n%s\ndF(A, B) = %f" % (str(A_star), str(B_star), dist_star))
    draw_plot(np.concatenate((A_star, [A_star[0]])), np.concatenate((B_star, [B_star[0]])), (i_star, j_star))
