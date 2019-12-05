from chord_matrix import generate_chord_matrix
from examples.gfileextractor import run_example_extract


def run_example_chord1():
    border, center = run_example_extract()
    matrix = generate_chord_matrix(border, center, 4, 6, verbose=True)
    print(matrix)
