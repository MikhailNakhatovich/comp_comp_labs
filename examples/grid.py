import numpy as np

from gfileextractor import extract
from grid import generate_grid, draw_grid, generate_domain_grid, draw_domains


def run_example_grid1():
    mag_mesh = 65
    flux, RBDRY, ZBDRY, NBDRY, R, Z, rdim, zdim = extract('g035685.00150', mag_mesh)
    min = np.argmin(flux)
    rmin = min // mag_mesh
    zmin = min - rmin * mag_mesh
    center = (R[rmin], Z[zmin])
    border = np.column_stack((RBDRY, ZBDRY))
    lines, radial_lines = generate_grid(border, center, count=3, radials=6)
    draw_grid(lines, center, radial_lines)
    domains = generate_domain_grid(border, center, count=3, radials=6)
    draw_domains(domains, center)
