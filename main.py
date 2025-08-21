# ==================================================================================================== #
# Nom : Pipe_Flow_2D_SIMPLE                                                                            #
# Projet : CFD_PYTHON_2025                                                                             #
# Version : 1.0                                                                                        #
#                                                                                                      #
# Auteur : Gabin HOUZET (gab1.h@orange.fr)                                                             #
# Date de création : 17/08/2025                                                                        #
# Dernière modification le : 20/08/2025                                                                #
# ---------------------------------------------------------------------------------------------------- #
# Description : Unsteady SIMPLE algorithm on a collocated grid.                                        #
#                                                                                                      #
# Commentaire : -                                                                                      #
# ==================================================================================================== #


# 1. Governing Equations (Incompressible Navier-Stokes):
#       ρ∂u/∂t + ρ(u⋅∇)u = -∇p + μ∇²u + ρf
#                     ∇u = 0

# 2. Temporal Discretization (Implicit Euler):
#    1st Order :
#       ∂/∂t ∫ρu dV ≈ ρΔV/Δt ⋅ ​(un+1​ − un​)
#       → AtP = ρΔxΔy/Δt
#       → Qt = ρΔxΔy/Δt ⋅ un
#
#    3rd Order :
#       ∂/∂t ∫ρu dV ≈ ρΔV/2Δt ⋅ ​(3un+1​ − 4un + un-1​)
#       → AtP = 3ρΔxΔy/2Δt
#       → Qt = ρΔxΔy/2Δt ⋅ (4un - un-1)

# 3. Spacial Discretization:
#    3.1 Diffusion (CDS):
#        ∂²u/∂x² ≈ (ui+1,j - 2ui,j + ui-1,j)/Δx²
#        ∂²u/∂y² ≈ (ui,j+1 - 2ui,j + ui,j-1)/Δy²
#        → AdE = AdW = μΔy/Δx
#        → AdN = AdS = μΔx/Δy
#        → AdP = AdE + AdW + AdN + AdS
#
#    3.2 Convection (Upwind):
#        a. Mass fluxes through faces:
#           Ff = ρ(uf⋅nf)Af
#        b. Upwind Scheme:
#           Ff ⩾ 0 → uP, Ff < 0 → uK, K = E,W,N,S 
#        → AcE = max(-Fe, 0), AcW = max(Fw, 0), AcN = max(-Fn, 0), AcS = max(Fs, 0)
#        → AcP = AcE + AcW + AcN + AcS

# 4. Pressure gradient:
#    x-momentum :
#        - ∫∇p dV ≈ -ΔV/Δx ⋅ (pe - pw)
#        → Qp_u = -Δy⋅ (pe - pw)
#    y-momentum :
#        - ∫∇p dV ≈ -ΔV/Δy ⋅ (pn - ps)
#        → Qp_v = -Δx⋅ (pn - ps)


# ==================== < IMPORTS > ===================

import numpy as np
import math
import scipy.sparse.linalg
import pyamg
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

from boundary_conditions import *
from momentum import *


# ==================== < PARAMETERS > ===================

# Grid & domain properties:
Nx, Ny = 64, 64    # Number of CVs [-]
Lx, Ly = 1, 1      # Dimensions of the domain [m]

dx, dy = Lx/Nx, Ly/Ny    # [m]

# Fluid properties :
rho = 1.0      # Density (ρ) [kg/m^3]
mu = 0.01      # Dynamic viscosity (μ) [kg/m/s]
nu = rho/mu    # Kinematic viscosity (ν) [m^2/s]

# Simulation properties :
Ut = 1.0    # Top wall x-velocity [m/s]

dt = 0.01    # Timestep size [s]

t = 0.0        # Start time [s]
t_end = 5.0    # End time [s]

n_steps = math.ceil((t_end - t)/dt)    # Number of timesteps


# ==================== < FIELD INITIALIZATION > ===================

u = np.zeros((Nx+2, Ny+2))    # x-velocity
v = np.zeros((Nx+2, Ny+2))    # y-velocity
p = np.zeros((Nx+2, Ny+2))    # Pressure

apply_u_bcs(u, Ut)
apply_v_bcs(v)
apply_p_bcs(p)


# ==================== < FUNCTIONS > ===================

def build_mom_coeffs(u, v, p, dir):
    """
    Assemble the coefficients for the x-momentum (u-velocity) equation.

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

    # Initialize face velocities arrays:
    ue = np.zeros_like(u)
    uw = np.zeros_like(u)
    vn = np.zeros_like(v)
    vs = np.zeros_like(v)

    # Face velocities:
    ue = 0.5*(u[1:-1, 1:-1] + u[1:-1, 2:])
    uw = 0.5*(u[1:-1, :-2] + u[1:-1, 1:-1])
    vn = 0.5*(v[1:-1, 1:-1] + v[2:, 1:-1])
    vs = 0.5*(v[:-2, 1:-1] + v[1:-1, 1:-1])

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
    aP = Qt + aE + aW + aN + aS

    if dir == 'x':
        Qp = -dy * (p[1:-1, 2:] - p[1:-1, :-2]) / 2    # Pressure gradient term
        b = Qt*u[1:-1, 1:-1] + Qp
    elif dir == 'y':
        Qp = -dx * (p[2:, 1:-1] - p[:-2, 1:-1]) / 2    # Pressure gradient term
        b = Qt*v[1:-1, 1:-1] + Qp

    return aP, aE, aW, aN, aS, b


def assemble_sparse_matrix(aP, aE, aW, aN, aS):
    d0 = aP.reshape(Nx*Ny)
    de = aE.reshape(Nx*Ny)[:-1]
    dw = aW.reshape(Nx*Ny)[1:]
    dn = aN.reshape(Nx*Ny)[:-Nx]
    ds = aS.reshape(Nx*Ny)[Nx:]

    A = scipy.sparse.diags([d0, de, dw, dn, ds], [0, 1, -1, Nx, -Nx], format='csr')

    # plt.matshow(A.toarray())
    # plt.show()

    return A


# ==================== < MAIN TIME LOOP > ===================

aP_u, aE_u, aW_u, aN_u, aS_u, b_u = build_mom_coeffs(u, v, p, dir='x')
aP_v, aE_v, aW_v, aN_v, aS_v, b_v = build_mom_coeffs(u, v, p, dir='y')

A_u = assemble_sparse_matrix(aP_u, aE_u, aW_u, aN_u, aS_u)
A_v = assemble_sparse_matrix(aP_v, aE_v, aW_v, aN_v, aS_v)