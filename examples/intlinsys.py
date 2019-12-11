import numpy as np
from scipy.io import loadmat

from chord_matrix import generate_chord_matrix
from examples.gfileextractor import run_example_extract
from intlinsys import solve, heur_min_cond


def analyze_cond(a):
    for r in np.arange(0.1, 0.55, 0.05):
        min_cond = heur_min_cond(a, r=r, n=100)
        print('r = %4.2f  n = %4d  min_cond = %f' % (r, 100, min_cond))

    for n in np.arange(10, 101, 10):
        min_cond = heur_min_cond(a, n=n)
        print('r = %4.2f  n = %4d  min_cond = %f' % (0.1, n, min_cond))
    min_cond = heur_min_cond(a, n=1000)
    print('r = %4.2f  n = %4d  min_cond = %f' % (0.1, 1000, min_cond))


def run_example_intlinsys1():
    border, center = run_example_extract('data/g035685.00150')
    matrix = generate_chord_matrix(border, center, 4, 6, verbose=False)

    analyze_cond(matrix)

    mat = loadmat('data/35685_SPD16x16.mat')
    t = 150
    sign_bb = mat['sign_bb']
    tp = mat['Data'][0][1][0][0] * 1e-3
    tz = mat['Data'][1][1][0][0]
    ind = int((t - tz) / tp)
    ind_inf = ind - 1 if ind > 0 else ind
    ind_sup = ind + 1 if ind < sign_bb.shape[2] - 1 else ind
    b_inf = np.min(sign_bb[:, :, ind_inf:ind_sup + 1], axis=2)
    b_sup = np.max(sign_bb[:, :, ind_inf:ind_sup + 1], axis=2)
    b_inf = np.rot90(b_inf, 2).T.reshape(256)
    b_sup = np.rot90(b_sup, 2).T.reshape(256)

    tolmax, argmax = solve(matrix, b_inf, b_sup, verbose=True)

    b_inf -= abs(tolmax)
    b_sup += abs(tolmax)

    tolmax, argmax = solve(matrix, b_inf, b_sup, verbose=True)
