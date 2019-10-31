import numpy as np

from frechet import distance_frechet, draw_plot


def run_example_frechet():
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
