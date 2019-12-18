import numpy as np

from chord_matrix import generate_chord_matrix
from examples.fileextractor import run_example_extract, run_example_loadmat
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

    b_inf, b_sup = run_example_loadmat('data/35685_SPD16x16.mat', 150)

    tolmax, argmax = solve(matrix, b_inf, b_sup, verbose=True)

    b_inf -= abs(tolmax)
    b_sup += abs(tolmax)

    tolmax, argmax = solve(matrix, b_inf, b_sup, verbose=True)
