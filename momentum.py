import numpy as np

from faces import *

def build_momentum_coeffs(u, v, p, axis, Nx, Ny, dx, dy, rho, mu, dt):
    """
    Assemble the coefficients for the x/y-momentum (u/v-velocity) equations.

    Returns aP, aE, aW, aN, aS, b
    """

    # Initialize coefficients arrays:
    aP = np.zeros((Ny, Nx))
    aE = np.zeros((Ny, Nx))
    aW = np.zeros((Ny, Nx))
    aN = np.zeros((Ny, Nx))
    aS = np.zeros((Ny, Nx))
    b  = np.zeros((Ny, Nx))

    # Face velocities:
    ue, uw, vn, vs = linear_face_velocities(u, v)

    # Mass fluxes through faces:
    Fe, Fw, Fn, Fs = rho*ue*dy, rho*uw*dy, rho*vn*dx, rho*vs*dx

    # Coefficient arrays – Diffusion + Convection (Upwind):
    aE = mu*dy/dx + np.maximum(-Fe, 0)
    aW = mu*dy/dx + np.maximum(Fw, 0)
    aN = mu*dx/dy + np.maximum(-Fn, 0)
    aS = mu*dx/dy + np.maximum(Fs, 0)

    # Implicit time term:
    Qt = rho*dx*dy/dt

    # RHS:
    if axis == 'x':
        Qp = -dy * (p[1:-1, 2:] - p[1:-1, :-2]) / 2    # Pressure gradient term
        b = Qt*u[1:-1, 1:-1] + Qp
    
    elif axis == 'y':
        Qp = -dx * (p[2:, 1:-1] - p[:-2, 1:-1]) / 2    # Pressure gradient term
        b = Qt*v[1:-1, 1:-1] + Qp

    # Add boundary contributions to RHS:
    field = {"x": u, "y": v}[axis]

    b[:, -1] += aE[:, -1] * field[1:-1, -1]
    b[:, 0]  += aW[:, 0] * field[1:-1, 0]
    b[-1, :] += aN[-1, :] * field[-1, 1:-1]
    b[0, :]  += aS[0, :] * field[0, 1:-1]

    # Zero boundary coefficients:
    aE[:, -1] = 0
    aW[:, 0]  = 0
    aN[-1, :] = 0
    aS[0, :]  = 0

    aP = Qt + aE + aW + aN + aS

    return aP, aE, aW, aN, aS, b