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
from continuity import *
from sparse_solver import *


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
u_star = np.zeros_like(u)     # Tentative x-velocity
v_star = np.zeros_like(v)     # Tentative y-velocity
p_prime = np.zeros_like(p)     # Pressure correction

apply_u_bcs(u, Ut)
apply_v_bcs(v)
apply_p_bcs(p)


# ==================== < MAIN TIME LOOP > ===================

aP_u, aE_u, aW_u, aN_u, aS_u, b_u = build_momentum_coeffs(u, v, p, 'x', Nx, Ny, dx, dy, rho, mu, dt)
u_star[1:-1, 1:-1] = solve(aP_u, aE_u, aW_u, aN_u, aS_u, b_u, Nx, Ny)
apply_u_bcs(u_star, Ut)

aP_v, aE_v, aW_v, aN_v, aS_v, b_v = build_momentum_coeffs(u, v, p, 'y', Nx, Ny, dx, dy, rho, mu, dt)
v_star[1:-1, 1:-1] = solve(aP_v, aE_v, aW_v, aN_v, aS_v, b_v, Nx, Ny)
apply_v_bcs(v_star)

aP_p, aE_p, aW_p, aN_p, aS_p, b_p = build_continuity_coeffs(u_star, v_star, p, aP_u, aP_v, Nx, Ny, dx, dy, rho)
p_prime[1:-1, 1:-1] = solve(aP_p, aE_p, aW_p, aN_p, aS_p, b_p, Nx, Ny)

plt.imshow(p_prime, origin='lower')
plt.colorbar()
plt.show()