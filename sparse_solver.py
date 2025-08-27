import scipy.sparse.linalg
import pyamg

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


def solve(aP, aE, aW, aN, aS, b, Nx, Ny):
    """
    Solve system of equations from coefficicent arrays.
    """
    A = assemble_sparse_matrix(aP, aE, aW, aN, aS, Nx, Ny)

    amg_solver = pyamg.ruge_stuben_solver(A)
    res = amg_solver.solve(b.reshape(Nx*Ny), tol=1e-6, cycle='V')

    return res.reshape([Ny, Nx])