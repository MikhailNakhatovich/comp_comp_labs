import numpy as np
from scipy.io import loadmat

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


def run_example_loadmat(filename, t):
    mat = loadmat(filename)
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
    return b_inf, b_sup
