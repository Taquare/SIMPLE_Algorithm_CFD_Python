import numpy as np

from faces import *

def build_continuity_coeffs(u_star, v_star, p, aP_u, aP_v, Nx, Ny, dx, dy, rho):
    """
    """

    # Initialize coefficients arrays:
    aP = np.zeros((Ny, Nx))
    aE = np.zeros((Ny, Nx))
    aW = np.zeros((Ny, Nx))
    aN = np.zeros((Ny, Nx))
    aS = np.zeros((Ny, Nx))
    b  = np.zeros((Ny, Nx))

    aP_u_ew = 0.5 * (aP_u[:, :-1] + aP_u[:, 1:])
    aP_v_ns = 0.5 * (aP_v[:-1, :] + aP_v[1:, :])

    aE[:, :-1] = - (rho*dy*dy)/(aP_u_ew)
    aW[:, 1:]  = - (rho*dy*dy)/(aP_u_ew)
    aN[:-1, :] = - (rho*dx*dx)/(aP_v_ns)
    aS[1:, :]  = - (rho*dx*dx)/(aP_v_ns)

    aP = - (aE + aW + aN + aS)

    # Corrected face velocities:
    ue, uw, vn, vs = rhie_chow_face_velocites(u_star, v_star, p, aP_u, aP_v, dx, dy)

    # Mass fluxes through faces:
    Fe, Fw, Fn, Fs = rho*ue*dy, rho*uw*dy, rho*vn*dx, rho*vs*dx

    # RHS:
    b = Fe - Fw + Fn - Fs

    return aP, aE, aW, aN, aS, b