import numpy as np
import scipy.sparse.linalg
import pyamg

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

    # Diffusion terms:
    aE_d, aW_d, aN_d, aS_d = mu*dy/dx, mu*dy/dx, mu*dx/dy, mu*dx/dy

    # Implicit time term:
    Qt = rho*dx*dy/dt

    # Face velocities:
    ue, uw, vn, vs = linear_face_velocities(u, v)

    # Mass fluxes through faces:
    Fe, Fw, Fn, Fs = rho*ue*dy, rho*uw*dy, rho*vn*dx, rho*vs*dx

    # Convection coefficients (Upwind):
    aE_c = np.maximum(-Fe, 0)
    aW_c = np.maximum(Fw, 0)
    aN_c = np.maximum(-Fn, 0)
    aS_c = np.maximum(Fs, 0)

    # Full coefficient arrays:
    aE = aE_c + aE_d
    aW = aW_c + aW_d
    aN = aN_c + aN_d
    aS = aS_c + aS_d

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


def assemble_sparse_matrix(aP, aE, aW, aN, aS, Nx, Ny):
    """
    Assemble sparse 2D matrix from coefficient arrays.
    """

    d0 = aP.ravel()
    de = aE.ravel()[:-1]
    dw = aW.ravel()[1:]
    dn = aN.ravel()[:-Nx]
    ds = aS.ravel()[Nx:]

    A = scipy.sparse.diags([d0, de, dw, dn, ds], [0, 1, -1, Nx, -Nx], format='csr')

    return A


def solve_momentum(aP, aE, aW, aN, aS, b, Nx, Ny):
    """
    Solve x/y-momentum equation for u* or v*.
    """
    A = assemble_sparse_matrix(aP, aE, aW, aN, aS, Nx, Ny)

    amg_solver = pyamg.ruge_stuben_solver(A)
    res = amg_solver.solve(b.reshape(Nx*Ny), tol=1e-6, cycle='V')

    return res.reshape([Ny, Nx])