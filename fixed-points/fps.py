import numpy as np
import numpy.typing as npt
from numba import njit

Matrix = npt.NDArray[np.float_]
Vector = npt.NDArray[np.float_]

# find the fixed_point of the MSPBE objective
@njit(cache=True)
def lstd(X: Matrix, P: Matrix, R: Matrix, d: Vector):
    I = np.eye(X.shape[0])
    D = np.diag(d)

    A = X.T.dot(D).dot(I - P).dot(X)
    b = X.T.dot(D).dot(R)

    lsq = np.linalg.lstsq(A, b, rcond=None)
    return lsq[0]

# find the fixed_point of the MSBE objective
@njit(cache=True)
def brm(X: Matrix, P: Matrix, R: Matrix, d: Vector):
    D = np.diag(d)
    dX = X - P.dot(X)

    F = dX.T.dot(D).dot(dX)
    g = dX.T.dot(D).dot(R)

    lsq = np.linalg.lstsq(F, g, rcond=None)
    return lsq[0]

# find the best representable value function
@njit(cache=True)
def lsv(X: Matrix, d: Vector, v_star: Vector):
    D = np.diag(d)

    v_hat = np.linalg.pinv(X.T.dot(D).dot(X)).dot(X.T).dot(D).dot(v_star)

    return v_hat
