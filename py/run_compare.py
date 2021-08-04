import argparse
import logging
import os

import numpy as np
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), ".."))
sys.path.append(os.path.join(os.getcwd(), "..", ".."))

from py import data_wrangling as dw, metrics, plots, signals

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s: %(message)s')
logger = logging.getLogger("COMPARATIVE_EXPERIMENT")


def process(features_path: str,
            input_path: str,
            labels_mask_length: int,
            image_nums: list,
            plots_path: str,
            cs1_mask_range: list,
            labels_mask_range: list,
            multi_process: bool = False):
    features = dw.get_features(features_path)
    skip_indices = set()
    for index in range(features.shape[1]):
        feature_array = features[:, index]
        m = feature_array.max()
        if m == 0:
            skip_indices.add(index)

    kanerva_radius_list = [1, 2, 3, 4, 5, 6]
    kanerva_p0s = ["0.990", "0.995"]
    kanerva_signals_map = signals.get_kanerva_signals_all(input_path, kanerva_radius_list, kanerva_p0s, image_nums)

    labels_signals_map = signals.get_labels_signals_all(input_path, labels_mask_range, image_nums)

    cs1_signals_map = signals.get_cs1_signals_all(input_path, cs1_mask_range, image_nums)

    kanerva_metrics_map = metrics.get_kanerva_metrics_all(kanerva_signals_map, features, image_nums)
    labels_metrics_map = metrics.get_labels_metrics_all(labels_signals_map, features, image_nums)
    cs1_metrics_map = metrics.get_cs1_metrics_all(cs1_signals_map, features, image_nums)

    plots.plot_kanerva(plots_path, kanerva_metrics_map, kanerva_radius_list, image_nums)
    plots.plot_labels(plots_path, labels_metrics_map, labels_mask_range, image_nums)
    plots.plot_cs1(plots_path, cs1_metrics_map, cs1_mask_range, image_nums)
    plots.plot_comparison(plots_path, image_nums,
                          kanerva_metrics_map, labels_metrics_map, cs1_metrics_map,
                          1, 2, 14)
    print()
    # save_plots(plots_path, image_nums, features, labels_signals_map, cs1_signals_map, cs1_mask_range, skip_indices)


def process_noisy(features_path: str,
                  input_path: str,
                  labels_mask_length: int,
                  image_nums: list,
                  plots_path: str,
                  cs1_mask_range: list,
                  labels_mask_range: list,
                  multi_process: bool = False,
                  ):
    features = dw.get_features(features_path)

    less_than_3_indices = set()

    image_nums_cp = image_nums.copy()
    image_nums_actual = []
    for index in range(features.shape[1]):
        if index == image_nums_cp[0]:
            image_nums_actual.append(image_nums_cp[0] - len(less_than_3_indices))
            del image_nums_cp[0]

        feature_array = features[:, index]
        non_zero = feature_array.nonzero()
        non_zero_count = non_zero[0].shape[0]
        if non_zero_count < 3:
            less_than_3_indices.add(index)
    image_nums_actual.append(image_nums_cp[0] - len(less_than_3_indices))
    delete_list = sorted(list(less_than_3_indices))
    features_filtered = np.delete(features, delete_list, axis=1)

    # cs1_signals_map = signals.get_cs1_signals_all(input_path, cs1_mask_range, image_nums)

    # cs1_signals_noisy_1_map = signals.calculate_cs1_signals_all(
    #     os.path.abspath(os.path.join(os.getcwd(), "..", "output_noisy")),
    #     features_filtered, cs1_mask_range, image_nums, set(),
    #     write_to_disk=True, mask="_noisy")
    # cs1_signals_noisy_2_map = signals.calculate_cs1_signals_all(
    #     os.path.abspath(os.path.join(os.getcwd(), "..", "output_noisy_2")),
    #     features_filtered, cs1_mask_range, image_nums, set(),
    #     write_to_disk=True, mask="_noisy_2")

    kanerva_metrics_path = os.path.abspath(
        os.path.join(os.getcwd(), "..", "metrics", "kanerva_metrics_3_features.json"))
    try:
        kanerva_metrics_map = metrics.read_metrics(kanerva_metrics_path)
    except Exception as error:
        logger.warning(f"Error while reading Kanerva metrics: {repr(error)}")

        kanerva_radius_list = [1, 2, 3, 4, 5, 6]
        kanerva_p0s = ["0.990", "0.995"]
        kanerva_signals_map = signals.get_kanerva_signals_all(input_path, kanerva_radius_list, kanerva_p0s, image_nums,
                                                              delete_list=delete_list)
        kanerva_metrics_map = metrics.get_kanerva_metrics_all(kanerva_signals_map, features_filtered, image_nums)
        metrics.save_metrics(kanerva_metrics_map, kanerva_metrics_path)

    jaeckel_metrics_path = os.path.abspath(
        os.path.join(os.getcwd(), "..", "metrics", "jaeckel_metrics_3_features.json"))
    try:
        jaeckel_metrics_map = metrics.read_metrics(jaeckel_metrics_path)
    except Exception as error:
        logger.warning(f"Error while reading Jaeckel metrics: {repr(error)}")

        jaeckel_signals_map = signals.get_labels_signals_all(input_path, labels_mask_range, image_nums,
                                                             delete_list=delete_list)
        jaeckel_metrics_map = metrics.get_labels_metrics_all(jaeckel_signals_map, features_filtered, image_nums)
        metrics.save_metrics(jaeckel_metrics_map, jaeckel_metrics_path)

    cs1_noisy_1_metrics_path = os.path.abspath(
        os.path.join(os.getcwd(), "..", "metrics", "cs1_noisy_1_metrics_3_features.json"))
    try:
        cs1_metrics_noisy_1_map = metrics.read_metrics(cs1_noisy_1_metrics_path)
    except Exception as error:
        logger.warning(f"Error while reading CS SDM (1 noisy bit) metrics: {repr(error)}")

        try:
            cs1_signals_noisy_1_map = signals.get_cs1_signals_all(
                os.path.abspath(os.path.join(os.getcwd(), "..", "output_noisy")),
                cs1_mask_range, image_nums, mask="_noisy")
        except Exception as cs1_noisy_1_error:
            logger.warning(f"Error while reading CS SDM (1 noisy bit) signals: {repr(cs1_noisy_1_error)}")

            cs1_signals_noisy_1_map = signals.calculate_cs1_signals_all(
                os.path.abspath(os.path.join(os.getcwd(), "..", "output_noisy")),
                features_filtered, cs1_mask_range, image_nums, set(),
                write_to_disk=True, mask="_noisy"
            )

        cs1_metrics_noisy_1_map = metrics.get_cs1_metrics_all(cs1_signals_noisy_1_map, features_filtered, image_nums)
        metrics.save_metrics(cs1_metrics_noisy_1_map, cs1_noisy_1_metrics_path)

    cs1_noisy_2_metrics_path = os.path.abspath(
        os.path.join(os.getcwd(), "..", "metrics", "cs1_noisy_2_metrics_3_features.json"))
    try:
        cs1_metrics_noisy_2_map = metrics.read_metrics(cs1_noisy_2_metrics_path)
    except Exception as error:
        logger.warning(f"Error while reading CS SDM (1 noisy bit) metrics: {repr(error)}")

        try:
            cs1_signals_noisy_2_map = signals.get_cs1_signals_all(
                os.path.abspath(os.path.join(os.getcwd(), "..", "output_noisy_2")),
                cs1_mask_range, image_nums, mask="_noisy_2")
        except Exception as cs1_noisy_2_error:
            logger.warning(f"Error while reading CS SDM (2 noisy bits) signals: {repr(cs1_noisy_2_error)}")

            cs1_signals_noisy_2_map = signals.calculate_cs1_signals_all(
                os.path.abspath(os.path.join(os.getcwd(), "..", "output_noisy_2")),
                features_filtered, cs1_mask_range, image_nums, set(),
                write_to_disk=True, mask="_noisy_2"
            )

        cs1_metrics_noisy_2_map = metrics.get_cs1_metrics_all(cs1_signals_noisy_2_map, features_filtered, image_nums)
        metrics.save_metrics(cs1_metrics_noisy_2_map, cs1_noisy_2_metrics_path)

    plots.plot_cs1_noisy_1(plots_path, cs1_metrics_noisy_1_map, cs1_mask_range, image_nums)
    plots.plot_cs1_noisy_2(plots_path, cs1_metrics_noisy_2_map, cs1_mask_range, image_nums)
    plots.plot_noisy_comparison(plots_path, image_nums,
                                kanerva_metrics_map, jaeckel_metrics_map,
                                cs1_metrics_noisy_1_map, cs1_metrics_noisy_2_map,
                                1, 2, 12, 12, image_nums_actual=image_nums_actual)
    plots.plot_noisy_comparison_bars(plots_path, [i * 1_500 for i in range(1, 7)],
                                     kanerva_metrics_map, jaeckel_metrics_map,
                                     cs1_metrics_noisy_1_map, cs1_metrics_noisy_2_map,
                                     1, 2, 12, 12, image_nums_actual=[image_nums_actual[i] for i in range(1, 18, 3)])
    print()


def process_cs2_naive(features_path: str,
                      input_path: str,
                      labels_mask_length: int,
                      image_nums: list,
                      plots_path: str,
                      cs1_mask_range: list,
                      labels_mask_range: list,
                      multi_process: bool = False):
    features = dw.get_features(features_path)
    skip_indices = set()
    # for index in range(features.shape[1]):
    #     feature_array = features[:, index]
    #     m = feature_array.max()
    #     if m == 0:
    #         skip_indices.add(index)
    delete_list = []
    features_filtered = np.delete(features, delete_list, axis=1)

    kanerva_metrics_path = os.path.abspath(os.path.join(os.getcwd(), "..", "metrics", "kanerva_metrics_.json"))
    try:
        kanerva_metrics_map = metrics.read_metrics(kanerva_metrics_path)
    except Exception as error:
        logger.warning(f"Error while reading Kanerva metrics: {repr(error)}")

        kanerva_radius_list = [1, 2, 3, 4, 5, 6]
        kanerva_p0s = ["0.990", "0.995"]
        kanerva_signals_map = signals.get_kanerva_signals_all(input_path, kanerva_radius_list, kanerva_p0s, image_nums,
                                                              delete_list=delete_list)
        kanerva_metrics_map = metrics.get_kanerva_metrics_all(kanerva_signals_map, features_filtered, image_nums)
        metrics.save_metrics(kanerva_metrics_map, kanerva_metrics_path)

    jaeckel_metrics_path = os.path.abspath(os.path.join(os.getcwd(), "..", "metrics", "jaeckel_metrics.json"))
    try:
        jaeckel_metrics_map = metrics.read_metrics(jaeckel_metrics_path)
    except Exception as error:
        logger.warning(f"Error while reading Jaeckel metrics: {repr(error)}")

        jaeckel_signals_map = signals.get_labels_signals_all(input_path, labels_mask_range, image_nums,
                                                             delete_list=delete_list)
        jaeckel_metrics_map = metrics.get_labels_metrics_all(jaeckel_signals_map, features_filtered, image_nums)
        metrics.save_metrics(jaeckel_metrics_map, jaeckel_metrics_path)

    cs2_naive_metrics_path = os.path.abspath(os.path.join(os.getcwd(), "..", "metrics", "cs2_naive_metrics.json"))
    try:
        cs2_metrics_naive_map = metrics.read_metrics(cs2_naive_metrics_path)
    except Exception as error:
        logger.warning(f"Error while reading CS2 SDM naive metrics: {repr(error)}")

        try:
            cs2_metrics_naive_map = signals.get_cs1_signals_all(
                os.path.abspath(os.path.join(os.getcwd(), "..", "output")),
                [1, 2, 3, 4, 5], image_nums, mask="cs2_naive_signal")
        except Exception as cs1_noisy_1_error:
            logger.warning(f"Error while reading CS2 SDM naive signals: {repr(cs1_noisy_1_error)}")

            cs2_metrics_naive_map = signals.calculate_cs1_signals_all(
                os.path.abspath(os.path.join(os.getcwd(), "..", "output")),
                features_filtered, [1, 2, 3, 4, 5], image_nums, set(),
                write_to_disk=True, input_prefix="cs2_naive", matrix_prefix="cs2_matrix",
                output_prefix="cs2_naive_signal",
            )

        cs2_metrics_naive_map = metrics.get_cs1_metrics_all(cs2_metrics_naive_map, features_filtered, image_nums)
        metrics.save_metrics(cs2_metrics_naive_map, cs2_naive_metrics_path)

    plots.plot_cs2_naive(plots_path, cs2_metrics_naive_map, [1, 2, 3, 4, 5], image_nums)
    plots.plot_cs2_naive_comparison(plots_path, image_nums,
                                    kanerva_metrics_map, jaeckel_metrics_map, cs2_metrics_naive_map,
                                    1, 2, 3)
    print()


def process_cs2_noisy_1(features_path: str,
                        input_path: str,
                        labels_mask_length: int,
                        image_nums: list,
                        plots_path: str,
                        cs1_mask_range: list,
                        labels_mask_range: list,
                        multi_process: bool = False,
                        ):
    features = dw.get_features(features_path)

    less_than_4_indices = set()

    image_nums_cp = image_nums.copy()
    image_nums_actual = []
    for index in range(features.shape[1]):
        if index == image_nums_cp[0]:
            image_nums_actual.append(image_nums_cp[0] - len(less_than_4_indices))
            del image_nums_cp[0]

        feature_array = features[:, index]
        non_zero = feature_array.nonzero()
        non_zero_count = non_zero[0].shape[0]
        if non_zero_count < 4:
            less_than_4_indices.add(index)
    image_nums_actual.append(image_nums_cp[0] - len(less_than_4_indices))
    delete_list = sorted(list(less_than_4_indices))
    features_filtered = np.delete(features, delete_list, axis=1)

    kanerva_metrics_path = os.path.abspath(
        os.path.join(os.getcwd(), "..", "metrics", "kanerva_metrics_4_features.json"))
    try:
        kanerva_metrics_map = metrics.read_metrics(kanerva_metrics_path)
    except Exception as error:
        logger.warning(f"Error while reading Kanerva metrics: {repr(error)}")

        kanerva_radius_list = [1, 2, 3, 4, 5, 6]
        kanerva_p0s = ["0.990", "0.995"]
        kanerva_signals_map = signals.get_kanerva_signals_all(input_path, kanerva_radius_list, kanerva_p0s, image_nums,
                                                              delete_list=delete_list)
        kanerva_metrics_map = metrics.get_kanerva_metrics_all(kanerva_signals_map, features_filtered, image_nums)
        metrics.save_metrics(kanerva_metrics_map, kanerva_metrics_path)

    jaeckel_metrics_path = os.path.abspath(
        os.path.join(os.getcwd(), "..", "metrics", "jaeckel_metrics_4_features.json"))
    try:
        jaeckel_metrics_map = metrics.read_metrics(jaeckel_metrics_path)
    except Exception as error:
        logger.warning(f"Error while reading Jaeckel metrics: {repr(error)}")

        jaeckel_signals_map = signals.get_labels_signals_all(input_path, labels_mask_range, image_nums,
                                                             delete_list=delete_list)
        jaeckel_metrics_map = metrics.get_labels_metrics_all(jaeckel_signals_map, features_filtered, image_nums)
        metrics.save_metrics(jaeckel_metrics_map, jaeckel_metrics_path)

    cs2_noisy_1_metrics_path = os.path.abspath(
        os.path.join(os.getcwd(), "..", "metrics", "cs2_noisy_1_metrics_4_features.json"))
    try:
        cs2_metrics_noisy_1_map = metrics.read_metrics(cs2_noisy_1_metrics_path)
    except Exception as error:
        logger.warning(f"Error while reading CS2 SDM (1 noisy bit) metrics: {repr(error)}")

        try:
            cs2_signals_noisy_1_map = signals.get_cs1_signals_all(
                os.path.abspath(os.path.join(os.getcwd(), "..", "output")),
                cs1_mask_range, image_nums, mask="_noisy")
        except Exception as cs1_noisy_1_error:
            logger.warning(f"Error while reading CS2 SDM (1 noisy bit) signals: {repr(cs1_noisy_1_error)}")

            cs2_signals_noisy_1_map = signals.calculate_cs1_signals_all(
                os.path.abspath(os.path.join(os.getcwd(), "..", "output")),
                features_filtered, cs1_mask_range, image_nums, set(),
                write_to_disk=True, input_prefix="cs2_noisy_1", matrix_prefix="cs2_noisy_1_matrix", output_prefix="cs2_noisy_1_signal"
            )

        cs2_metrics_noisy_1_map = metrics.get_cs1_metrics_all(cs2_signals_noisy_1_map, features_filtered, image_nums)
        metrics.save_metrics(cs2_metrics_noisy_1_map, cs2_noisy_1_metrics_path)

    cs2_noisy_2_metrics_path = os.path.abspath(
        os.path.join(os.getcwd(), "..", "metrics", "cs2_noisy_2_metrics_4_features.json"))
    try:
        cs2_metrics_noisy_2_map = metrics.read_metrics(cs2_noisy_2_metrics_path)
    except Exception as error:
        logger.warning(f"Error while reading CS2 SDM (2 noisy bits) metrics: {repr(error)}")

        try:
            cs2_signals_noisy_2_map = signals.get_cs1_signals_all(
                os.path.abspath(os.path.join(os.getcwd(), "..", "output")),
                cs1_mask_range, image_nums, mask="_noisy")
        except Exception as cs1_noisy_2_error:
            logger.warning(f"Error while reading CS2 SDM (2 noisy bits) signals: {repr(cs1_noisy_2_error)}")

            cs2_signals_noisy_2_map = signals.calculate_cs1_signals_all(
                os.path.abspath(os.path.join(os.getcwd(), "..", "output")),
                features_filtered, cs1_mask_range, image_nums, set(),
                write_to_disk=True, input_prefix="cs2_noisy_2", matrix_prefix="cs2_noisy_2_matrix", output_prefix="cs2_noisy_2_signal"
            )

        cs2_metrics_noisy_2_map = metrics.get_cs1_metrics_all(cs2_signals_noisy_2_map, features_filtered, image_nums)
        metrics.save_metrics(cs2_metrics_noisy_2_map, cs2_noisy_2_metrics_path)
        
    plots.plot_cs2_noisy_1(plots_path, image_nums,
                           kanerva_metrics_map, jaeckel_metrics_map,
                           cs2_metrics_noisy_1_map, cs2_metrics_noisy_2_map,
                           1, 2, 3, 3, image_nums_actual=image_nums_actual)
    print()


def process_distro(plots_path: str, features_path: str):
    features = dw.get_features(features_path)
    plots.plot_feature_distribution(plots_path, features)


def process_errors(plots_path: str, features_path: str):
    features = dw.get_features(features_path)

    less_than_4_indices = set()

    for index in range(features.shape[1]):
        feature_array = features[:, index]
        non_zero = feature_array.nonzero()
        non_zero_count = non_zero[0].shape[0]
        if non_zero_count <= 2:
            less_than_4_indices.add(index)
    delete_list = sorted(list(less_than_4_indices))
    features_filtered = np.delete(features, delete_list, axis=1)

    # cs2_signals_noisy_2_map = signals.calculate_cs1_signals_all(
    #     os.path.abspath(os.path.join(os.getcwd(), "..", "output")),
    #     features_filtered, [3], [9_000], set(),
    #     write_to_disk=True,
    #     input_prefix="cs2_naive_geq_4", matrix_prefix="cs2_geq_4_matrix", output_prefix="cs2_naive_geq_4_signal"
    # )

    signals = np.genfromtxt("/home/rolandw0w/Development/PhD/output/cs2_naive_geq_4_signal_K_3_I_9000.csv", delimiter=",")
    image_name = "err_distro_per_feat_count"
    plots.plot_error_distro_num_features(plots_path, features_filtered, signals, image_name)
    plots.plot_error_distribution_1(plots_path, features_filtered)
    plots.plot_error_distribution_2(plots_path, features_filtered)


def process_s1(plots_path: str,
            features_path: str,
            image_nums: list,
            ):
    features = dw.get_features(features_path)

    less_than_4_indices = set()

    for index in range(features.shape[1]):
        feature_array = features[:, index]
        non_zero = feature_array.nonzero()
        non_zero_count = non_zero[0].shape[0]
        if non_zero_count <= 2:
            less_than_4_indices.add(index)
    delete_list = sorted(list(less_than_4_indices))
    features_filtered = np.delete(features, delete_list, axis=1)

    cs2_noisy_2_metrics_path = os.path.abspath(
        os.path.join(os.getcwd(), "..", "metrics", "cs2_geq_3_s1_metrics_4_features.json"))
    try:
        cs2_metrics_noisy_2_map = metrics.read_metrics(cs2_noisy_2_metrics_path)
    except Exception as error:
        logger.warning(f"Error while reading CS2 SDM (2 noisy bits) metrics: {repr(error)}")

        try:
            cs2_signals_noisy_2_map = signals.get_cs1_signals_all(
                os.path.abspath(os.path.join(os.getcwd(), "..", "output")),
                [3, 4, 5], image_nums, mask="_geq_3_s1")
        except Exception as cs1_noisy_2_error:
            logger.warning(f"Error while reading CS2 SDM (2 noisy bits) signals: {repr(cs1_noisy_2_error)}")

            cs2_signals_noisy_2_map = signals.calculate_cs1_signals_all(
                os.path.abspath(os.path.join(os.getcwd(), "..", "output")),
                features_filtered, [3, 4, 5], image_nums, set(),
                write_to_disk=True, input_prefix="cs2_geq_3_s1", matrix_prefix="cs2_geq_3_s1_matrix", output_prefix="cs2_geq_3_s1_signal"
            )

        cs2_metrics_noisy_2_map = metrics.get_cs1_metrics_all(cs2_signals_noisy_2_map, features_filtered, image_nums)
        metrics.save_metrics(cs2_metrics_noisy_2_map, cs2_noisy_2_metrics_path)

    sgnls = np.genfromtxt("/home/rolandw0w/Development/PhD/output/cs2_geq_3_s1_signal_K_3_I_9000.csv", delimiter=",")
    image_name = "err_distro_per_feat_count_s1"
    plots.plot_error_distro_num_features(plots_path, features_filtered, sgnls, image_name)
    print(cs2_metrics_noisy_2_map)


def process_s2(plots_path: str,
               features_path: str,
               image_nums: list,
               ):
    features = dw.get_features(features_path)

    less_than_4_indices = set()

    for index in range(features.shape[1]):
        feature_array = features[:, index]
        non_zero = feature_array.nonzero()
        non_zero_count = non_zero[0].shape[0]
        if non_zero_count <= 2:
            less_than_4_indices.add(index)
    delete_list = sorted(list(less_than_4_indices))
    features_filtered = np.delete(features, delete_list, axis=1)

    cs2_noisy_2_metrics_path = os.path.abspath(
        os.path.join(os.getcwd(), "..", "metrics", "cs2_s2_naive_geq_3_metrics.json"))
    try:
        cs2_metrics_noisy_2_map = metrics.read_metrics(cs2_noisy_2_metrics_path)
    except Exception as error:
        logger.warning(f"Error while reading CS2 SDM (2 noisy bits) metrics: {repr(error)}")

        try:
            cs2_signals_noisy_2_map = signals.get_cs1_signals_all(
                os.path.abspath(os.path.join(os.getcwd(), "..", "output")),
                list(range(8, 17)), image_nums, mask="_cs2_s2_geq_3")
        except Exception as cs1_noisy_2_error:
            logger.warning(f"Error while reading CS2 SDM (2 noisy bits) signals: {repr(cs1_noisy_2_error)}")

            cs2_signals_noisy_2_map = signals.calculate_cs1_signals_all(
                os.path.abspath(os.path.join(os.getcwd(), "..", "output")),
                features_filtered, list(range(8, 17)), image_nums, set(),
                write_to_disk=True, input_prefix="cs2_s2_naive_geq_3", matrix_prefix="cs2_s2_geq_3_matrix",
                output_prefix="cs2_s2_naive_geq3_signal"
            )

        cs2_metrics_noisy_2_map = metrics.get_cs1_metrics_all(cs2_signals_noisy_2_map, features_filtered, image_nums)
        metrics.save_metrics(cs2_metrics_noisy_2_map, cs2_noisy_2_metrics_path)

    sgnls = np.genfromtxt("/home/rolandw0w/Development/PhD/output/cs2_geq_3_s1_signal_K_3_I_9000.csv", delimiter=",")
    image_name = "err_distro_per_feat_count_s2"
    plots.plot_error_distro_num_features(plots_path, features_filtered, sgnls, image_name)
    print(cs2_metrics_noisy_2_map)


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

    default_cs1_min_mask_length = 8
    cs1_min_mask_length_help = f"Min mask length used for Compressed Sensing approach (default is {default_cs1_min_mask_length})"
    parser.add_argument("--cs1_min_mask_length", type=int,
                        help=cs1_min_mask_length_help, default=default_cs1_min_mask_length)

    default_cs1_max_mask_length = 16
    cs1_max_mask_length_help = f"Max mask length used for Compressed Sensing approach (default is {default_cs1_max_mask_length})"
    parser.add_argument("--cs1_max_mask_length", type=int,
                        help=cs1_max_mask_length_help, default=default_cs1_max_mask_length)

    default_cs1_opt_mask_length = None
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

    default_mode = "naive"
    mode_help = f"Mode (naive/noisy, default is {default_mode})"
    parser.add_argument("--mode", type=str, help=mode_help, default=default_mode)

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

    if args.mode == "naive":
        process(features_path, input_path,
                labels_mask_length,
                image_nums,
                plots_path,
                cs1_mask_range=cs1_mask_range,
                labels_mask_range=[1, 2, 3, 4, 5],
                multi_process=multi_process,
                )
    elif args.mode == "noisy":
        process_noisy(features_path, input_path,
                      labels_mask_length,
                      image_nums,
                      plots_path,
                      cs1_mask_range=cs1_mask_range,
                      labels_mask_range=[1, 2, 3, 4, 5],
                      multi_process=multi_process,
                      )
    elif args.mode == "cs2_naive":
        process_cs2_naive(features_path, input_path,
                          labels_mask_length,
                          image_nums,
                          plots_path,
                          cs1_mask_range=cs1_mask_range,
                          labels_mask_range=[1, 2, 3, 4, 5],
                          multi_process=multi_process,
                          )
    elif args.mode == "cs2_noisy_1":
        process_cs2_noisy_1(features_path, input_path,
                            labels_mask_length,
                            image_nums,
                            plots_path,
                            cs1_mask_range=[3],
                            labels_mask_range=[1, 2, 3, 4, 5],
                            multi_process=multi_process,
                            )
    elif args.mode == "distro":
        process_distro(plots_path, features_path)
    elif args.mode == "errors":
        process_errors(plots_path, features_path)
    elif args.mode == "s1":
        process_s1(plots_path, features_path, image_nums)
    elif args.mode == "s2":
        process_s2(plots_path, features_path, image_nums)
    else:
        msg = f"Mode should be one of [\"naive\", \"noisy\", \"cs2_naive\"], got {args.mode}"
        raise ValueError(msg)


if __name__ == "__main__":
    main()
