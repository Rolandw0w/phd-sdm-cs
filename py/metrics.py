import logging
import multiprocessing as mp

import numpy as np

from py.utils import calculate_l1, perf_measure

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s: %(message)s')
logger = logging.getLogger("METRICS")


def get_kanerva_metrics(kanerva_signals_map: dict, features: np.array, image_num: int):
    l1s_avg_map = {}
    fn_avg_map = {}
    fp_avg_map = {}
    logger.info(f"Started calculating Kanerva metrics: image_num={image_num}")
    for radius, map1 in kanerva_signals_map.items():
        for p0, signals in map1.items():
            l1s = []
            fns = []
            fps = []
            for i in range(image_num):
                features_i = features[:, i]
                signal_i = signals[i]

                l1 = calculate_l1(features_i, signal_i)
                l1s.append(l1)

                fn, fp, tn, tp = perf_measure(features_i, signal_i)
                fns.append(fn)
                fps.append(fp)

            l1s_avg_map.setdefault(radius, {})
            l1s_avg_map[radius][p0] = np.mean(l1s)

            fn_avg_map.setdefault(radius, {})
            fn_avg_map[radius][p0] = np.mean(fns)

            fp_avg_map.setdefault(radius, {})
            fp_avg_map[radius][p0] = np.mean(fps)

    logger.info(f"Finished calculating Kanerva metrics: image_num={image_num}")

    return l1s_avg_map, fn_avg_map, fp_avg_map


def get_kanerva_metrics_all(kanerva_signals_map: dict, features: np.ndarray, image_nums: list,
                            processes: int = -1,
                            ) -> dict:
    l1s_avg_map = {}
    fn_avg_map = {}
    fp_avg_map = {}

    if processes is None:
        for image_num, map1 in kanerva_signals_map.items():
            l1s_avg, fn_avg, fp_avg = get_kanerva_metrics(map1, features, image_num)
            l1s_avg_map[image_num] = l1s_avg
            fn_avg_map[image_num] = fn_avg
            fp_avg_map[image_num] = fp_avg
    else:
        process_num = len(kanerva_signals_map) if processes == -1 else processes
        with mp.Pool(process_num) as pool:
            params = [(kanerva_signals_map[image_num], features, image_num) for image_num in image_nums]
            kanerva_metrics_list = pool.starmap(get_kanerva_metrics, params)
            for i in range(len(image_nums)):
                image_num = image_nums[i]
                l1s_avg, fn_avg, fp_avg = kanerva_metrics_list[i]
                l1s_avg_map[image_num] = l1s_avg
                fn_avg_map[image_num] = fn_avg
                fp_avg_map[image_num] = fp_avg

    metrics_map = {
        "avg_l1": l1s_avg_map,
        "fn_avg": fn_avg_map,
        "fp_avg": fp_avg_map,
    }
    return metrics_map


def get_labels_metrics(masks_signals: dict, features: np.array, image_num: int):
    l1s_avg_map = {}
    fn_avg_map = {}
    fp_avg_map = {}
    logger.info(f"Started calculating Labels metrics: image_num={image_num}")
    for mask, signals in masks_signals.items():
        l1s = []
        fns = []
        fps = []
        for i in range(image_num):
            features_i = features[:, i]
            signal_i = signals[i]

            l1 = calculate_l1(features_i, signal_i)
            l1s.append(l1)

            fn, fp, tn, tp = perf_measure(features_i, signal_i)
            fns.append(fn)
            fps.append(fp)

        l1s_avg_map.setdefault(mask, {})
        l1s_avg_map[mask] = np.mean(l1s)

        fn_avg_map.setdefault(mask, {})
        fn_avg_map[mask] = np.mean(fns)

        fp_avg_map.setdefault(mask, {})
        fp_avg_map[mask] = np.mean(fps)

    logger.info(f"Finished calculating Labels metrics: image_num={image_num}")
    return l1s_avg_map, fn_avg_map, fp_avg_map


def get_labels_metrics_all(masks_signals: dict, features: np.ndarray, image_nums: list, processes: int = -1) -> dict:
    l1s_avg_map = {}
    fn_avg_map = {}
    fp_avg_map = {}

    if processes is None:
        for image_num, map1 in masks_signals.items():
            l1s_avg, fn_avg, fp_avg = get_labels_metrics(map1, features, image_num)
            l1s_avg_map[image_num] = l1s_avg
            fn_avg_map[image_num] = fn_avg
            fp_avg_map[image_num] = fp_avg
    else:
        process_num = len(masks_signals) if processes == -1 else processes
        with mp.Pool(process_num) as pool:
            params = [(masks_signals[image_num], features, image_num) for image_num in image_nums]
            labels_metrics_list = pool.starmap(get_labels_metrics, params)
            for i in range(len(image_nums)):
                image_num = image_nums[i]
                l1s_avg, fn_avg, fp_avg = labels_metrics_list[i]
                l1s_avg_map[image_num] = l1s_avg
                fn_avg_map[image_num] = fn_avg
                fp_avg_map[image_num] = fp_avg

    metrics_map = {
        "avg_l1": l1s_avg_map,
        "fn_avg": fn_avg_map,
        "fp_avg": fp_avg_map,
    }
    return metrics_map


def get_cs1_metrics(masks_signals: dict, features: np.array, image_num: int):
    l1s_avg_map = {}
    fn_avg_map = {}
    fp_avg_map = {}
    logger.info(f"Started calculating CS1 metrics: image_num={image_num}")
    for mask_length, signals in masks_signals.items():
        l1s = []
        fns = []
        fps = []
        for i in range(image_num):
            features_i = features[:, i]
            signal_i = signals[i]

            l1 = calculate_l1(features_i, signal_i)
            l1s.append(l1)

            fn, fp, tn, tp = perf_measure(features_i, signal_i)
            fns.append(fn)
            fps.append(fp)

        l1s_avg_map.setdefault(mask_length, {})
        l1s_avg_map[mask_length] = np.mean(l1s)

        fn_avg_map.setdefault(mask_length, {})
        fn_avg_map[mask_length] = np.mean(fns)

        fp_avg_map.setdefault(mask_length, {})
        fp_avg_map[mask_length] = np.mean(fps)

    logger.info(f"Finished calculating CS1 metrics: image_num={image_num}")

    return l1s_avg_map, fn_avg_map, fp_avg_map


def get_cs1_metrics_all(masks_signals: dict, features: np.ndarray, image_nums: list, processes: int = -1) -> dict:
    l1s_avg_map = {}
    fn_avg_map = {}
    fp_avg_map = {}

    if processes is None:
        for image_num, map1 in masks_signals.items():
            l1s_avg, fn_avg, fp_avg = get_labels_metrics(map1, features, image_num)
            l1s_avg_map[image_num] = l1s_avg
            fn_avg_map[image_num] = fn_avg
            fp_avg_map[image_num] = fp_avg
    else:
        process_num = len(masks_signals) if processes == -1 else processes
        with mp.Pool(process_num) as pool:
            params = [(masks_signals[image_num], features, image_num) for image_num in image_nums]
            cs1_metrics_list = pool.starmap(get_cs1_metrics, params)
            for i in range(len(image_nums)):
                image_num = image_nums[i]
                l1s_avg, fn_avg, fp_avg = cs1_metrics_list[i]
                l1s_avg_map[image_num] = l1s_avg
                fn_avg_map[image_num] = fn_avg
                fp_avg_map[image_num] = fp_avg

    metrics_map = {
        "avg_l1": l1s_avg_map,
        "fn_avg": fn_avg_map,
        "fp_avg": fp_avg_map,
    }
    return metrics_map
