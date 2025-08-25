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
from faces import *


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
u_star = np.zeros_like(u)
v_star = np.zeros_like(v)

apply_u_bcs(u, Ut)
apply_v_bcs(v)
apply_p_bcs(p)


# ==================== < MAIN TIME LOOP > ===================

u_star[1:-1, 1:-1], aP_u = solve_momentum(u, v, p, 'x', Nx, Ny, dx, dy, rho, mu, dt)
apply_u_bcs(u_star, Ut)

v_star[1:-1, 1:-1], aP_v = solve_momentum(u, v, p, 'y', Nx, Ny, dx, dy, rho, mu, dt)
apply_v_bcs(v_star)

ue_corr, uw_corr, vn_corr, vs_corr = rhie_chow_face_velocites(u_star, v_star, p, aP_u, aP_v, dx, dy)