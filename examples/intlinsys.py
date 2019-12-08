import numpy as np
from scipy.io import loadmat

from chord_matrix import generate_chord_matrix
from examples.gfileextractor import run_example_extract
from intlinsys import solve


def run_example_intlinsys1():
    border, center = run_example_extract('data/g035685.00150')
    matrix = generate_chord_matrix(border, center, 4, 6, verbose=False)

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
