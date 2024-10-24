from typing import Callable, Sequence
import numpy as np
from numpy.typing import ArrayLike
from tqdm import tqdm

def rk4_method(
    F: Callable[[float, ArrayLike], ArrayLike],
    y0: ArrayLike,
    t_vect: Sequence[float] = None,
    t0: float = None,
    t_max: float = None,
    e: float = None,
    disable=True,
):
    """
    Runge Kutta 4th Order

    calculates RungeKutta 4th order with fixed timesteps.
    Either accepts a prepared time point list t_vec or otherwise
    a starttime t0 endtime t_max and a time intervall

    Arguments:
        F -- Functional F(t,y(t)) for Equation y`(t) = F(t,y(t))
        y0 -- start condition vector

    Keyword Arguments:
        t_vect -- time points (default: {None}) should be in form array([t0, t1, t2,...,tn])
        t0 -- start time (default: {None})
        t_max -- end time (default: {None})
        e -- time intervall (default: {None})

    Returns:
        tuple (t_vect, y(t) as numpy array)
    """
    # time checking
    if t_vect is not None:
        # func is with t_vect
        tarr = t_vect
        t0 = t_vect[0]
        e = tarr[1] - tarr[0]
        t_max = t_vect[-1]
    elif t0 is not None and t_max is not None and e is not None:
        # we have to create t_vect
        tarr = np.arange(t0, t_max + e, e)
    else:
        # time params missing
        return None

    # basic dimensionalities
    N_t = np.size(tarr, 0)
    N_dim = len(y0)

    # prepeation
    y = np.zeros((N_t, N_dim))
    y[0] = y0

    for n in tqdm(range(1, N_t), desc="rk4-timeloop", disable=False):
        cur_t = tarr[n - 1]
        cur_y = y[n - 1]
        # for j in range(len(c)):
        #    k.append(F(cur_t + c[i]*e,))
        k1 = F(cur_t, cur_y)
        k2 = F(cur_t + rk4_c_j[1] * e, cur_y + e * (rk4_a_ij[1][0] * k1))
        k3 = F(
            cur_t + rk4_c_j[2] * e,
            cur_y + e * (rk4_a_ij[2][0] * k1 + rk4_a_ij[2][1] * k2),
        )
        k4 = F(
            cur_t + rk4_c_j[3] * e,
            cur_y
            + e
            * (
                rk4_a_ij[3][0] * k1 + rk4_a_ij[3][1] * k2 + rk4_a_ij[3][2] * k3
            ),
        )

        y[n] = cur_y + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4) * e

        tarr[n] = cur_t + e

    return tarr, y.T