import numpy as np

from gfileextractor import extract


def run_example_extract():
    mag_mesh = 65
    flux, RBDRY, ZBDRY, NBDRY, R, Z, rdim, zdim = extract('g035685.00150', mag_mesh)
    min = np.argmin(flux)
    rmin = min // mag_mesh
    zmin = min - rmin * mag_mesh
    center = (R[rmin], Z[zmin])
    border = np.column_stack((RBDRY, ZBDRY))
    return border, center
