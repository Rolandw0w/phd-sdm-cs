import argparse
import logging
import multiprocessing as mp
import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), ".."))
sys.path.append(os.path.join(os.getcwd(), "..", ".."))

from matplotlib import pyplot as plt
import numpy as np

from py import data_wrangling as dw
from py.restore_signal import restore_cs1_signal
from py.utils import calculate_l1

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s: %(message)s')
logger = logging.getLogger("COMPARATIVE_EXPERIMENT")


def get_labels_signals(input_path: str,
                       labels_mask_length: int,
                       image_nums: list) -> dict:
    labels_signals_map = {}
    for image_num in image_nums:
        logger.info(f"Started reading labels signals: image_num={image_num}")
        labels_path = os.path.join(input_path, f"labels_stat_K_{labels_mask_length}_I_{image_num}.csv")
        if not os.path.isfile(labels_path):
            msg = f"File {labels_path} not found"
            raise ValueError(msg)

        labels_signals = np.genfromtxt(labels_path, delimiter=",")
        labels_signals_map[image_num] = labels_signals

    return labels_signals_map


def get_cs1_signals(input_path: str,
                    features: np.array,
                    cs1_mask_range: list,
                    image_num: int,
                    skip_indices: set) -> dict:
    cs1_signal = {}
    for mask_length in cs1_mask_range:
        logger.info(f"Starting signal restoration: mask_length={mask_length}, image_num={image_num}")
        cs1_path = os.path.join(input_path, f"cs1_K_{mask_length}_I_{image_num}.csv")
        if not os.path.isfile(cs1_path):
            msg = f"File {cs1_path} not found"
            raise ValueError(msg)

        cs1_matrix_path = os.path.join(input_path, f"cs1_matrix_K_{mask_length}_I_{image_num}.csv")
        if not os.path.isfile(cs1_matrix_path):
            msg = f"File {cs1_matrix_path} not found"
            raise ValueError(msg)

        cs1 = np.genfromtxt(cs1_path, delimiter=",")
        cs1_matrix = np.genfromtxt(cs1_matrix_path, delimiter=",")

        restored = []
        for i in range(image_num):
            if i in skip_indices:
                restored.append(None)
                continue

            features_i = features[:, i]
            features_i_non_zero = features_i.nonzero()

            cs1_i = cs1[i]
            cs1_restored_signal = restore_cs1_signal(features_i_non_zero, cs1_i, cs1_matrix)
            restored.append(cs1_restored_signal)

        cs1_signal[mask_length] = restored
        logger.info(f"Finished signal restoration: mask_length={mask_length}, image_num={image_num}")

    return cs1_signal


def save_plots(plots_path: str,
               image_nums: list,
               features: np.array,
               labels_signals_map: dict, cs1_signals_map: dict,
               cs1_mask_range: list,
               skip_indices: set):
    for mask_length in cs1_mask_range:
        cs1_img_l1s = {image_num: {} for image_num in image_nums}
        labels_img_l1s = {image_num: {} for image_num in image_nums}
        for image_num in image_nums:
            cs1_l1s = []
            cs1_fp = 0
            cs1_fn = 0
            labels_l1s = []
            labels_fp = 0
            labels_fn = 0

            norm = image_num
            for i in range(image_num):
                if i in skip_indices:
                    norm -= 1
                    continue

                features_i = features[:, i]
                labels_i = labels_signals_map[image_num][i]
                cs1_i = cs1_signals_map[image_num][mask_length][i]

                cs1_l1 = calculate_l1(features_i, cs1_i)
                cs1_l1s.append(cs1_l1)

                labels_l1 = calculate_l1(features_i, labels_i)
                labels_l1s.append(labels_l1)

                for j in range(len(features_i)):
                    if features_i[j] == 0 and cs1_i[j] == 1:
                        cs1_fp += 1
                    if features_i[j] == 1 and cs1_i[j] == 0:
                        cs1_fn += 1
                    if features_i[j] == 0 and labels_i[j] == 1:
                        labels_fp += 1
                    if features_i[j] == 1 and labels_i[j] == 0:
                        labels_fn += 1

            cs1_img_l1s[image_num]["avg_l1"] = np.mean(cs1_l1s)
            labels_img_l1s[image_num]["avg_l1"] = np.mean(labels_l1s)
            cs1_img_l1s[image_num]["fn_avg"] = cs1_fn / norm
            labels_img_l1s[image_num]["fn_avg"] = labels_fn / norm
            cs1_img_l1s[image_num]["fp_avg"] = cs1_fp / norm
            labels_img_l1s[image_num]["fp_avg"] = labels_fp / norm

        for key, y_label, title, image_name in [
            ("avg_l1", "Average L1", "Labeled approach vs Compressed Sensing (avg L1)",
             f"average_L1_K_{mask_length}"),
            ("fn_avg", "Average False Negative", "Labeled approach vs Compressed Sensing (avg FN)",
             f"average_false_negative_K_{mask_length}"),
            ("fp_avg", "Average False Positive", "Labeled approach vs Compressed Sensing (avg FP)",
             f"average_false_positive_K_{mask_length}"),
        ]:
            y_cs1 = np.array([cs1_img_l1s[image_num][key] for image_num in image_nums])
            y_labels = np.array([labels_img_l1s[image_num][key] for image_num in image_nums])
            x = image_nums

            x_ticks = x
            x_tick_labels = [str(k) for k in x_ticks]

            y_min = np.floor(min(y_cs1.min(), y_labels.min()))
            y_max = np.ceil(max(y_cs1.max(), y_labels.max()))
            step = (y_max - y_min) / 20
            y_ticks = np.arange(y_min, y_max + 1e-10, step)
            y_tick_labels = ['%.2f' % k for k in y_ticks]

            plt.xticks(x_ticks, fontsize=6, labels=x_tick_labels)
            plt.yticks(y_ticks, labels=y_tick_labels)

            plt.plot(x, y_labels, marker="o")
            plt.plot(x, y_cs1, marker="o")

            plt.legend(["Labeled approach", "Compressed Sensing"])
            plt.xlabel("Number of images restored")
            plt.ylabel(y_label)
            plt.title(title)

            plot_path = os.path.abspath(os.path.join(plots_path, f"{image_name}.png"))
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Image {plot_path} was saved")


def process(features_path: str,
            input_path: str,
            labels_mask_length: int,
            image_nums: list,
            plots_path: str,
            cs1_mask_range: list,
            multi_process: bool = False):
    features = dw.get_features(features_path)
    skip_indices = set()
    for i in range(max(image_nums)):
        f_i = features[:, i]
        f_i_non_zero = f_i.nonzero()[0]
        if len(f_i_non_zero) == 0:
            skip_indices.add(i)

    labels_signals_map = get_labels_signals(input_path, labels_mask_length, image_nums)

    def _process_cs1(img_num):
        return get_cs1_signals(input_path, features, cs1_mask_range, image_num, skip_indices)

    cs1_signals_map = {}
    if not multi_process:
        for image_num in image_nums:
            cs1_signals = _process_cs1(image_num)
            cs1_signals_map[image_num] = cs1_signals
    else:
        process_num = len(image_nums)
        with mp.Pool(process_num) as pool:
            params = [(input_path, features, cs1_mask_range, img_num, skip_indices)
                      for img_num in image_nums]
            cs1_signals_list = pool.starmap(get_cs1_signals, params)
            cs1_signals_map = dict(zip(image_nums, cs1_signals_list))

    save_plots(plots_path, image_nums, features, labels_signals_map, cs1_signals_map, cs1_mask_range, skip_indices)


def main():
    parser = argparse.ArgumentParser()

    cwd = os.getcwd()

    default_features_path = os.path.abspath(os.path.join(cwd, "..", "data/features.bin"))
    features_path_help = f"Path to .bin with features (default is {default_features_path})"
    parser.add_argument("--features", type=str, help=features_path_help, default=default_features_path)

    default_input = os.path.abspath(os.path.join(cwd, "..", "output"))
    input_help = f"Path to directory with SDM responses (default is ({default_input})"
    parser.add_argument("--input", type=str, help=input_help, default=default_input)

    default_labels_mask_length = 2
    labels_mask_length_help = f"Mask length used for Labelled approach (default is {default_labels_mask_length})"
    parser.add_argument("--labels_mask_length", type=int,
                        help=labels_mask_length_help, default=default_labels_mask_length)

    default_cs1_min_mask_length = None
    cs1_min_mask_length_help = f"Min mask length used for Compressed Sensing approach (default is {default_cs1_min_mask_length})"
    parser.add_argument("--cs1_min_mask_length", type=int,
                        help=cs1_min_mask_length_help, default=default_cs1_min_mask_length)

    default_cs1_max_mask_length = None
    cs1_max_mask_length_help = f"Max mask length used for Compressed Sensing approach (default is {default_cs1_max_mask_length})"
    parser.add_argument("--cs1_max_mask_length", type=int,
                        help=cs1_max_mask_length_help, default=default_cs1_max_mask_length)

    default_cs1_opt_mask_length = 12
    cs1_opt_mask_length_help = f"Optimal mask length used for Compressed Sensing approach (default is {default_cs1_opt_mask_length})"
    parser.add_argument("--cs1_opt_mask_length", type=int,
                        help=cs1_opt_mask_length_help, default=default_cs1_opt_mask_length)

    default_min_image_num = 500
    min_image_num_help = f"Min image num to use for analysis and plots (default is {default_min_image_num})"
    parser.add_argument("--min_image_num", type=int, help=min_image_num_help, default=default_min_image_num)

    default_max_image_num = 9_000
    max_image_num_help = f"Max image num to use for analysis and plots (default is {default_max_image_num})"
    parser.add_argument("--max_image_num", type=int, help=max_image_num_help, default=default_max_image_num)

    default_image_num_step = 500
    image_num_step_help = f"Difference between two subsequent image nums (default is {default_image_num_step})"
    parser.add_argument("--image_num_step", type=int, help=image_num_step_help, default=default_image_num_step)

    default_plots_path = os.path.abspath(os.path.join(cwd, "..", "plots"))
    plots_path_help = f"Difference between two subsequent image nums (default is {default_plots_path})"
    parser.add_argument("--plots_path", type=str, help=plots_path_help, default=default_plots_path)

    default_multi_process = True
    multi_process_help = f"Difference between two subsequent image nums (default is {default_multi_process})"
    parser.add_argument("--multi_process", type=bool, help=multi_process_help, default=default_multi_process)

    args = parser.parse_args()

    features_path = args.features
    if not os.path.isfile(features_path):
        msg = f"File {features_path} not found"
        raise ValueError(msg)

    input_path = args.input
    if not os.path.isdir(input_path):
        msg = f"Directory {input_path} not found"
        raise ValueError(msg)

    labels_mask_length = args.labels_mask_length
    cs1_min_mask_length = args.cs1_min_mask_length
    cs1_max_mask_length = args.cs1_max_mask_length
    cs1_opt_mask_length = args.cs1_opt_mask_length

    min_image_num = args.min_image_num
    max_image_num = args.max_image_num
    image_num_step = args.image_num_step
    multi_process = args.multi_process

    plots_path = args.plots_path
    if not os.path.isdir(plots_path):
        msg = f"Directory {plots_path} not found"
        raise ValueError(msg)

    if cs1_min_mask_length is None or cs1_max_mask_length is None:
        cs1_min_mask_length = cs1_opt_mask_length
        cs1_max_mask_length = cs1_opt_mask_length

    cs1_mask_range = list(range(cs1_min_mask_length, cs1_max_mask_length + 1))
    image_nums = list(range(min_image_num, max_image_num + 1, image_num_step))
    process(features_path, input_path,
            labels_mask_length,
            image_nums,
            plots_path,
            cs1_mask_range=cs1_mask_range,
            multi_process=multi_process)


if __name__ == "__main__":
    main()
