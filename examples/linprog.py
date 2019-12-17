from chord_matrix import generate_chord_matrix
from examples.fileextractor import run_example_extract, run_example_loadmat
from linprog import solve


def run_example_linprog1():
    border, center = run_example_extract('data/g035685.00150')
    matrix = generate_chord_matrix(border, center, 4, 6, verbose=False)

    b_inf, b_sup = run_example_loadmat('data/35685_SPD16x16.mat', 150)

    x, success = solve(matrix, b_inf, b_sup, verbose=True)

    print("success: %r" % success)
