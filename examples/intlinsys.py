import numpy as np
from scipy.io import loadmat

from chord_matrix import generate_chord_matrix
from examples.gfileextractor import run_example_extract
from tolsolvty.tolsolvty import tolsolvty


def run_example_intlinsys1():
    border, center = run_example_extract('data/g035685.00150')
    matrix = generate_chord_matrix(border, center, 4, 6, verbose=False)

    mat = loadmat('data/35685_SPD16x16.mat')
    t = 150
    sign_bb = mat['sign_bb']
    tp = mat['Data'][0][1][0][0] * 1e-3
    tz = mat['Data'][1][1][0][0]
    ind = int((t - tz) / tp)
    if ind > 0:
        b_inf = np.min(sign_bb[:, :, ind - 1:ind + 1], axis=2)
    else:
        b_inf = sign_bb[:, :, ind]
    if ind < sign_bb.shape[2] - 1:
        b_sup = np.max(sign_bb[:, :, ind:ind + 2], axis=2)
    else:
        b_sup = sign_bb[:, :, ind]
    b_inf = np.rot90(b_inf, 2).T.reshape(256)
    b_sup = np.rot90(b_sup, 2).T.reshape(256)

    print('Condition number of the matrix: %f' % np.linalg.cond(matrix))
    tolmax, argmax, envs, ccode = tolsolvty(matrix, matrix, b_inf, b_sup)
    print(tolmax, argmax, envs, ccode, sep='\n')
