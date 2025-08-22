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
u_star = np.zeros_like(u)
v_star = np.zeros_like(v)

apply_u_bcs(u, Ut)
apply_v_bcs(v)
apply_p_bcs(p)


# ==================== < MAIN TIME LOOP > ===================

u_star[1:-1, 1:-1] = solve_momentum(u, v, p, 'x', Nx, Ny, dx, dy, rho, mu, dt)
apply_u_bcs(u_star, Ut)

plt.imshow(u_star, origin='lower')
plt.colorbar()
plt.show()

v_star[1:-1, 1:-1] = solve_momentum(u, v, p, 'y', Nx, Ny, dx, dy, rho, mu, dt)
apply_v_bcs(v_star)

plt.imshow(v_star, origin='lower')
plt.colorbar()
plt.show()