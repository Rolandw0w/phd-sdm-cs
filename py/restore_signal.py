import gurobipy as gp
from gurobipy import GRB

import numpy as np

import scipy.optimize

from sklearn.linear_model import OrthogonalMatchingPursuit, orthogonal_mp
from time import time


class GurobiInFeasibleException(Exception):
    pass


def restore_cs1_signal(non_zero_features, sdm_signal, transformation, restoration_type="omp",
                       error_handler=print, length=600, **kwargs) -> np.ndarray:
    try:
        if isinstance(non_zero_features, tuple):
            len_non_zero_features = len(non_zero_features[0])
            if len_non_zero_features == 0:
                raise ValueError("No features in array")
        elif isinstance(non_zero_features, int):
            len_non_zero_features = non_zero_features

        set_sdm_signal = set(sdm_signal)
        if set_sdm_signal == {0}:
            cs1_signal = np.zeros((transformation.shape[1],))
        else:
            if restoration_type == "omp":
                matrix = transformation.copy().astype(np.float64)
                arr = np.array(sdm_signal.copy())

                omp = OrthogonalMatchingPursuit(n_nonzero_coefs=len_non_zero_features)
                omp.fit(matrix, arr)

                inds = omp.coef_.argsort()[-len_non_zero_features:][::-1]
                cs1_signal = np.zeros((transformation.shape[1],))
                cs1_signal[inds] = 1
            elif restoration_type == "linprog":
                matrix = np.vstack([transformation.astype(int), np.ones((length,)).astype(int)])
                # arr = np.rint(np.append(sdm_signal, len_non_zero_features)).astype(int)
                arr = np.append(sdm_signal, len_non_zero_features)
                method = kwargs.get("method") or "interior-point"
                solution = scipy.optimize.linprog(c=np.ones((matrix.shape[1],)), A_eq=matrix, b_eq=arr,
                                                  method=method)
                solution_x = solution.x
                if solution_x is None:
                    solution = scipy.optimize.linprog(c=np.ones((matrix.shape[1],)), A_eq=matrix, b_eq=arr)
                    solution_x = solution.x
                inds = np.abs(solution_x).argsort()[-len_non_zero_features:][::-1]
                cs1_signal = np.zeros((length,))
                cs1_signal[inds] = 1
            else:
                raise Exception(f"Unknown restoration type: {restoration_type}")

        return cs1_signal.astype(np.int8)
    except Exception as error:
        if callable(error_handler):
            error_handler(error)
        else:
            print(error)

        return np.zeros((transformation.shape[1],)).astype(np.int8)


def restore_cs1_signal_gurobi(non_zero_features, sdm_signal, transformation, error_handler=print) -> np.ndarray:
    try:
        if isinstance(non_zero_features, tuple):
            len_non_zero_features = len(non_zero_features[0])
            if len_non_zero_features == 0:
                raise ValueError("No features in array")
        elif isinstance(non_zero_features, int):
            len_non_zero_features = non_zero_features

        set_sdm_signal = set(sdm_signal)
        if set_sdm_signal == {0}:
            cs1_signal = np.zeros((transformation.shape[1], ), dtype=np.int8)
        else:
            model = gp.Model("mip1")
            model.setParam('OutputFlag', 0)
            model.setParam('MIPFocus', 3)

            vars = []
            for i in range(transformation.shape[1]):
                x = model.addVar(vtype=GRB.BINARY, name=f"x_{i}")
                vars.append(x)

            ys = []
            for i in range(transformation.shape[0]):
                y = model.addVar(vtype=GRB.INTEGER)
                ys.append(y)

            # sums = []
            for i in range(transformation.shape[0]):
                s = sum([transformation[i][j] * vars[j] for j in range(transformation.shape[1])])
                # sums.append(s)
                # model.addConstr(s == sdm_signal[i], f"c_{i}")
                model.addConstr(s - sdm_signal[i] <= ys[i], f"c0_{i}")
                model.addConstr(sdm_signal[i] - s <= ys[i], f"c1_{i}")

            model.addConstr(sum(vars) == len_non_zero_features, f"c_sum")

            model.setObjective(sum(ys), GRB.MINIMIZE)


            # l = [gp.abs_(sums[i] - sdm_signal[i]) for i in range(transformation.shape[0])]
            # model.setObjective(gp.sum_(l[1:], start=l[0]), GRB.MINIMIZE)

            model.optimize()
            if model.status == GRB.OPTIMAL:
                cs1_signal = np.array([x.x for n, x in enumerate(vars)], dtype=np.int8)
            else:
                raise GurobiInFeasibleException()
                # cs1_signal = np.zeros((transformation.shape[1], ), dtype=np.int8)

        return cs1_signal
    except GurobiInFeasibleException:
        raise
    except Exception as error:
        if callable(error_handler):
            error_handler("err", error)
        else:
            print("err", error)

        return np.zeros((transformation.shape[1], ), dtype=np.int8)


def restore_cs1_signal_cosamp(non_zero_features, sdm_signal, transformation, error_handler=print,
                              tol=1e-10, precision=1e-12, max_iter=1000, transformation_transposed=None):
    """
    @Brief:  "CoSaMP: Iterative signal recovery from incomplete and inaccurate
             samples" by Deanna Needell & Joel Tropp

    @Input:  Phi - Sampling matrix
             u   - Noisy sample vector
             s   - Sparsity vector

    @Return: A s-sparse approximation "a" of the target signal
    """
    if isinstance(non_zero_features, tuple):
        len_non_zero_features = len(non_zero_features[0])
        if len_non_zero_features == 0:
            raise ValueError("No features in array")
    elif isinstance(non_zero_features, int):
        len_non_zero_features = non_zero_features
    try:
        max_iter -= 1  # Correct the while loop
        a = np.zeros(transformation.shape[1])
        v = sdm_signal
        iterations = 0
        halt = False

        # speedup
        if transformation_transposed is None:
            transformation_transposed = np.transpose(transformation)
        sdm_signal_norm = np.linalg.norm(sdm_signal)
        while not halt:
            iterations += 1
            # print("Iteration {}\r".format(iter))

            y = abs(np.dot(transformation_transposed, v))
            # Omega = [i for (i, val) in enumerate(y) if val > np.sort(y)[::-1][2*len_non_zero_features] and val > precision]  # equivalent to below
            # Omega = np.argwhere(y >= np.maximum(np.sort(y)[::-1][2*len_non_zero_features], precision))
            Omega = np.flatnonzero(y >= np.maximum(np.sort(y)[::-1][2*len_non_zero_features], precision))
            T = np.union1d(Omega, a.nonzero()[0])
            # T = np.union1d(Omega, T)
            b = np.dot(np.linalg.pinv(transformation[:, T]), sdm_signal)
            igood = (abs(b) > np.sort(abs(b))[::-1][len_non_zero_features]) & (abs(b) > precision)
            T = T[igood]
            a[T] = b[igood]
            v = sdm_signal - np.dot(transformation[:, T], b[igood])

            halt = np.linalg.norm(v)/sdm_signal_norm < tol or iterations > max_iter

        # print(iterations, end=" ")

        inds = np.abs(a).argsort()[-len_non_zero_features:][::-1]
        cs1_signal = np.zeros((transformation.shape[1],), dtype=np.int8)
        cs1_signal[inds] = 1
        return cs1_signal, iterations
    except Exception as error:
        # if callable(error_handler):
        #     error_handler("err", error)
        # else:
        #     print("err", error)
        cs1_signal = np.zeros((transformation.shape[1], ), dtype=np.int8)

        return cs1_signal, -1
