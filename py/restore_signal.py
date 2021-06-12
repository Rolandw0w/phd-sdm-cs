import numpy as np

from sklearn.linear_model import OrthogonalMatchingPursuit


def restore_cs1_signal(non_zero_features, sdm_signal, transformation, error_handler=print) -> np.ndarray:
    try:
        len_non_zero_features = len(non_zero_features[0])
        if len_non_zero_features == 0:
            raise ValueError("No features in array")

        set_sdm_signal = set(sdm_signal)
        if set_sdm_signal == {0}:
            cs1_signal = np.zeros((transformation.shape[1],))
        else:
            omp = OrthogonalMatchingPursuit(n_nonzero_coefs=len_non_zero_features)
            omp.fit(transformation, sdm_signal)
            cs1_signal = omp.coef_
            cs1_signal[cs1_signal != 0] = 1

        return cs1_signal
    except Exception as error:
        if callable(error_handler):
            error_handler(error)
        else:
            print(error)

        return np.zeros((transformation.shape[1],))
