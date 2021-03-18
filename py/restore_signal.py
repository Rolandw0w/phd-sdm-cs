from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit

import data_wrangling as dw
from utils import calculate_l1


def restore_signal(true_signal, sdm_signal, transformation, error_handler=None) -> np.ndarray:
    try:
        non_zero_features = true_signal.nonzero()
        len_non_zero_features = len(non_zero_features[0])
        if len_non_zero_features == 0:
            raise ValueError("No features in array")

        set_sdm_signal = set(sdm_signal)
        if set_sdm_signal == {0}:
            raise RuntimeError("SDM signal is zero")

        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=len_non_zero_features)
        omp.fit(transformation, sdm_signal)
        restored_signal = omp.coef_
        restored_signal[restored_signal != 0] = 1

        return restored_signal
    except Exception as error:
        if callable(error_handler):
            error_handler(error)
        else:
            print(error)


def process():
    transformation = dw.get_transformation()
    features = dw.get_features()
    transformed = np.matmul(transformation, features)

    all_l1s = {}
    avg_l1s = {}
    all_grouped = {}
    for images_count in range(500, 9001, 500):
        grouped = {}
        restored = dw.get_restored(f"data/restored_K_10_I_{images_count}.txt")
        l1s = []

        feat_0 = []
        rest_0 = []

        for i in range(images_count):
            try:
                feat_i = features[:, i]

                set_feat_i = set(feat_i)
                if set_feat_i == {0}:
                    feat_0.append(i)
                    continue

                f_nz = feat_i.nonzero()
                len_f_nz = len(f_nz[0])

                double_i = restored[i]
                set_double_i = set(double_i)
                if set_double_i == {0}:
                    rest_0.append(i)
                    continue

                omp = OrthogonalMatchingPursuit(n_nonzero_coefs=len_f_nz)
                omp.fit(transformation, double_i)
                coef = omp.coef_
                coef[coef != 0] = 1
                l1 = calculate_l1(feat_i, coef)
                l1s.append(l1)

                grouped.setdefault(len_f_nz, [])
                grouped[len_f_nz].append(l1)
            except Exception as e:
                print(e)

        avg_l1 =np.mean(l1s)
        print(f"I={images_count} avg_L1={avg_l1}")
        avg_l1s[images_count] = avg_l1
        all_l1s[images_count] = l1s
        all_grouped[images_count] = grouped

    with open("C:\\Development\\PhD\\Code\\reports\\labels_knots.txt", "r") as f_labels:
        content = f_labels.read()
        report = eval(content)

        for key, y_label, title, image_name in [
            ("avg_dist", "Average L1", "Labeled approach vs Compressed Sensing (avg L1)", "average_L1"),
            ("fn_avg", "Average False Negative", "Labeled approach vs Compressed Sensing (avg FN)", "average_false_negative"),
            ("fp_avg", "Average False Positive", "Labeled approach vs Compressed Sensing (avg FP)", "average_false_positive"),
        ]:
            dists = {r["image_count"]: r[key] for r in report}

            x = sorted(dists.keys())
            y1 = [dists[k] for k in x]
            y2 = [avg_l1s[k] for k in x]

            x_ticks = x
            y_ticks = range(14)
            x_tick_labels = [str(k) for k in x_ticks]
            y_tick_labels = [str(k) for k in y_ticks]

            plt.xticks(x_ticks, fontsize=6)
            # plt.xticklabels(x_tick_labels)
            plt.yticks(y_ticks)
            # plt.yticklabels(y_tick_labels)

            plt.plot(x, y1, marker="o")
            plt.plot(x, y2, marker="o")

            plt.legend(["Labeled approach", "Compressed Sensing"])
            plt.xlabel("Number of images restored")
            plt.ylabel(y_label)
            plt.title(title)
            # plt.show()
            plt.savefig(f"img/{image_name}.png")
            plt.close()
        print()
