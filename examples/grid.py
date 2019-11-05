import numpy as np

from gfileextractor import extract
from grid import generate_grid, draw_grid


def run_example_grid1():
    mag_mesh = 65
    flux, RBDRY, ZBDRY, NBDRY, R, Z, rdim, zdim = extract('g035685.00150', mag_mesh)
    min = np.argmin(flux)
    rmin = min // mag_mesh
    zmin = min - rmin * mag_mesh
    center = (R[rmin], Z[zmin])
    lines, radial_lines = generate_grid(np.column_stack((RBDRY, ZBDRY)), center, count=3, radials=6)
    draw_grid(lines, center, radial_lines)
