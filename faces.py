import numpy as np

def linear_face_velocities(u, v):
    """
    """

    ue = 0.5*(u[1:-1, 1:-1] + u[1:-1, 2:])
    uw = 0.5*(u[1:-1, :-2] + u[1:-1, 1:-1])
    vn = 0.5*(v[1:-1, 1:-1] + v[2:, 1:-1])
    vs = 0.5*(v[:-2, 1:-1] + v[1:-1, 1:-1])

    return ue, uw, vn, vs

def rhie_chow_face_velocites(u, v, p, aP_u, aP_v, dx, dy):
    """
    """

    ue, uw, vn, vs = linear_face_velocities(u, v)

    aP_ew = 0.5 * (aP_u[:, :-1] + aP_u[:, 1:])
    aP_ns = 0.5 * (aP_v[:-1, :] + aP_v[1:, :])

    dpdx = (p[1:-1, 2:-1] - p[1:-1, 1:-2])/dx
    dpdy = (p[2:-1, 1:-1] - p[1:-2, 1:-1])/dy

    dpdx_ = 0.5/dx * ((p[1:-1, 2:-1] - p[1:-1, :-3]) + (p[1:-1, 3:] - p[1:-1, 1:-2]))
    dpdy_ = 0.5/dy * ((p[2:-1, 1:-1] - p[:-3, 1:-1]) + (p[3:, 1:-1] - p[1:-2, 1:-1]))

    # Update ue, uw, vn, vs
    ue[:, :-1] -= (1.0/aP_ew) * (dpdx - dpdx_)
    uw[:, 1:]  -= (1.0/aP_ew) * (dpdx - dpdx_)
    vn[:-1, :] -= (1.0/aP_ns) * (dpdy - dpdy_)
    vs[1:, :]  -= (1.0/aP_ns) * (dpdy - dpdy_)

    return ue, uw, vn, vs