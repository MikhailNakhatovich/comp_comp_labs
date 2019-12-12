import numpy as np

from gfileextractor import extract


def run_example_extract(filename):
    mag_mesh = 65
    flux, RBDRY, ZBDRY, NBDRY, R, Z, rdim, zdim = extract(filename, mag_mesh)
    min = np.argmin(flux)
    zmin = min // mag_mesh
    rmin = min - zmin * mag_mesh
    center = (R[rmin], Z[zmin])
    border = np.column_stack((RBDRY, ZBDRY))
    return border, center
