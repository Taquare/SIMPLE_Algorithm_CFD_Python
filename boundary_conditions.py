import numpy as np

def apply_u_bcs(u, Ut):
    '''Apply Boundary Conditions (BCs) on the x-velocity (u)'''
    u[:, -1] = -u[:, -2]          # Right wall: No-slip → v = 0
    u[:, 0] = -u[:, 1]            # Left wall: No-sip → v = 0
    u[-1, :] = 2*Ut - u[-2, :]    # Top wall: no-slip → v = 0
    u[0, :] = -u[1, :]            # Bottom wall: no-slip → v = 0

def apply_v_bcs(v):
    '''Apply Boundary Conditions (BCs) on the y-velocity (v)'''
    v[:, -1] = -v[:, -2]    # Right wall: No-slip → v = 0
    v[:, 0] = -v[:, 1]      # Left wall: No-sip → v = 0
    v[-1, :] = -v[-2, :]    # Top wall: no-slip → v = 0
    v[0, :] = -v[1, :]      # Bottom wall: no-slip → v = 0

def apply_p_bcs(p):
    '''Apply Boundary Conditions (BCs) on the pressure (p)'''
    p[:, -1] = p[:, -2]    # Right wall: No-slip → v = 0
    p[:, 0] = p[:, 1]      # Left wall: No-sip → v = 0
    p[-1, :] = p[-2, :]    # Top wall: no-slip → v = 0
    p[0, :] = p[1, :]      # Bottom wall: no-slip → v = 0