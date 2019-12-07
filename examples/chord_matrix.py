import matplotlib.pyplot as plt

from chord_matrix import generate_chord_matrix
from examples.gfileextractor import run_example_extract


def run_example_chord1():
    border, center = run_example_extract('g035685.00150')
    matrix = generate_chord_matrix(border, center, 4, 6, verbose=True)
    plt.figure(figsize=(10, 10))
    plt.spy(matrix, aspect='auto', markersize=2, c='k')
    plt.show()
