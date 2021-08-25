import numpy as np

import scipy.optimize

from sklearn.linear_model import OrthogonalMatchingPursuit, orthogonal_mp
from time import time


def restore_cs1_signal(non_zero_features, sdm_signal, transformation, restoration_type="omp",
                       error_handler=print, **kwargs) -> np.ndarray:
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
                cs1_signal = np.zeros((transformation.shape[1],)).astype(int)
                cs1_signal[inds] = 1
            elif restoration_type == "linprog":
                matrix = np.vstack([transformation.astype(int), np.ones((600,)).astype(int)])
                # arr = np.rint(np.append(sdm_signal, len_non_zero_features)).astype(int)
                arr = np.append(sdm_signal, len_non_zero_features)
                method = kwargs.get("method") or "interior-point"
                t = time()
                solution = scipy.optimize.linprog(c=np.ones((matrix.shape[1],)), A_eq=matrix, b_eq=arr,
                                                  method=method)
                solution_x = solution.x
                if solution_x is None:
                    solution = scipy.optimize.linprog(c=np.ones((matrix.shape[1],)), A_eq=matrix, b_eq=arr)
                    solution_x = solution.x
                inds = np.abs(solution_x).argsort()[-len_non_zero_features:][::-1]
                cs1_signal = np.zeros((600,)).astype(int)
                cs1_signal[inds] = 1
            else:
                raise Exception(f"Unknown restoration type: {restoration_type}")

        return cs1_signal
    except Exception as error:
        if callable(error_handler):
            error_handler(error)
        else:
            print(error)

        return np.zeros((transformation.shape[1],))
