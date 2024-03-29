import logging
import multiprocessing as mp
import os

import numpy as np

from restore_signal import restore_cs1_signal


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s: %(message)s')
logger = logging.getLogger("SIGNALS_READER")


def get_kanerva_signals(input_path: str,
                        radius_list: list,
                        p0s: list,
                        image_num: int,
                        delete_list: list = None,
                        ) -> dict:
    logger.info(f"Starting reading Kanerva signals: image_num={image_num}")
    signals_map = {}
    for radius in radius_list:
        for p0 in p0s:
            file_name = f"kanerva_D_{radius}_I_{image_num}_p0_{p0}.csv"
            file_path = os.path.join(input_path, file_name)

            signals = np.genfromtxt(file_path, delimiter=",")
            if isinstance(delete_list, list):
                delete_indices = [x for x in delete_list if x < image_num]
                signals = np.delete(signals, delete_indices, axis=0)
            signals_map.setdefault(p0, {})
            signals_map[p0][radius] = signals

    logger.info(f"Finished reading Kanerva signals: image_num={image_num}")
    return signals_map


def get_kanerva_signals_all(input_path: str,
                            radius_list: list,
                            p0s: list,
                            image_nums: list,
                            processes: int = -1,
                            delete_list: list = None,
                            ) -> dict:
    kanerva_signals_map = {}
    if processes is None:
        for image_num in image_nums:
            kanerva_signals = get_kanerva_signals(input_path, radius_list, p0s, image_num, delete_list=delete_list)
            kanerva_signals_map[image_num] = kanerva_signals
    else:
        process_num = len(image_nums) if processes == -1 else processes
        with mp.Pool(process_num) as pool:
            params = [(input_path, radius_list, p0s, img_num, delete_list) for img_num in image_nums]
            kanerva_signals_list = pool.starmap(get_kanerva_signals, params)
            kanerva_signals_map = dict(zip(image_nums, kanerva_signals_list))

    return kanerva_signals_map


def get_labels_signals(input_path: str,
                       labels_mask_range: list,
                       image_num: int,
                       delete_list: list = None,
                       ) -> dict:
    logger.info(f"Started reading Labels signals: image_num={image_num}")
    labels_signals = {}
    for mask_length in labels_mask_range:
        labels_path = os.path.join(input_path, f"labels_stat_K_{mask_length}_I_{image_num}.csv")
        if not os.path.isfile(labels_path):
            msg = f"File {labels_path} not found"
            raise ValueError(msg)

        labels_signal = np.genfromtxt(labels_path, delimiter=",")
        if isinstance(delete_list, list):
            delete_indices = [x for x in delete_list if x < image_num]
            labels_signal = np.delete(labels_signal, delete_indices, axis=0)
        labels_signals[mask_length] = labels_signal

    logger.info(f"Finished reading Labels signals: image_num={image_num}")
    return labels_signals


def get_labels_signals_all(input_path: str,
                           labels_mask_range: list,
                           image_nums: list,
                           processes: int = -1,
                           delete_list: list = None,
                           ) -> dict:
    labels_signals_map = {}
    if processes is None:
        for image_num in image_nums:
            labels_signals = get_labels_signals(input_path, labels_mask_range, image_num, delete_list=delete_list)
            labels_signals_map[image_num] = labels_signals
    else:
        process_num = len(image_nums) if processes == -1 else processes
        with mp.Pool(process_num) as pool:
            params = [(input_path, labels_mask_range, image_num, delete_list) for image_num in image_nums]
            labels_signals_list = pool.starmap(get_labels_signals, params)
            labels_signals_map = dict(zip(image_nums, labels_signals_list))

    return labels_signals_map


def get_cs1_signals(input_path: str,
                    cs1_mask_range: list,
                    image_num: int,
                    mask: str = "",
                    ) -> dict:
    logger.info(f"Started reading {mask} CS1 signals: image_num={image_num}")
    cs1_signals = {}
    for mask_length in cs1_mask_range:
        cs1_signal_path = os.path.join(input_path, f"{mask}_K_{mask_length}_I_{image_num}.csv")
        if not os.path.isfile(cs1_signal_path):
            msg = f"File {cs1_signal_path} not found"
            raise ValueError(msg)

        cs1_signal = np.genfromtxt(cs1_signal_path, delimiter=",")

        cs1_signals[mask_length] = cs1_signal

    logger.info(f"Finished reading {mask} CS1 signals: image_num={image_num}")
    return cs1_signals


def get_cs1_signals_all(input_path: str,
                        cs1_mask_range: list,
                        image_nums: list,
                        processes: int = -1,
                        mask: str = "",
                        ):
    cs1_signals_map = {}
    if processes is None:
        for image_num in image_nums:
            cs1_signals = get_cs1_signals(input_path, cs1_mask_range, image_num, mask=mask)
            cs1_signals_map[image_num] = cs1_signals
    else:
        process_num = len(image_nums) if processes == -1 else processes
        with mp.Pool(process_num) as pool:
            params = [(input_path, cs1_mask_range, image_num, mask) for image_num in image_nums]
            cs1_signals_list = pool.starmap(get_cs1_signals, params)
            cs1_signals_map = dict(zip(image_nums, cs1_signals_list))

    return cs1_signals_map


def calculate_cs1_signals(input_path: str,
                          features: np.array,
                          cs1_mask_range: list,
                          image_num: int,
                          skip_indices: set,
                          write_to_disk: bool = True,
                          input_prefix: str = "",
                          matrix_prefix: str = "",
                          output_prefix: str = "",
                          ) -> dict:
    cs1_signal = {}
    for mask_length in cs1_mask_range:
        logger.info(f"Starting {input_prefix} signal restoration: mask_length={mask_length}, image_num={image_num}")
        cs1_path = os.path.join(input_path, f"{input_prefix}_K_{mask_length}_I_{image_num}.csv")
        if not os.path.isfile(cs1_path):
            msg = f"File {cs1_path} not found"
            raise ValueError(msg)

        cs1_matrix_path = os.path.join(input_path, f"{matrix_prefix}_K_{mask_length}_I_{image_num}.csv")
        if not os.path.isfile(cs1_matrix_path):
            msg = f"File {cs1_matrix_path} not found"
            raise ValueError(msg)

        cs1 = np.genfromtxt(cs1_path, delimiter=",")
        cs1_matrix = np.genfromtxt(cs1_matrix_path, delimiter=",")

        restored = []
        for i in range(cs1.shape[0]):
            if i in skip_indices:
                restored.append(np.zeros((600,)))
                continue

            features_i = features[:, i]
            features_i_non_zero = features_i.nonzero()

            cs1_i = cs1[i]

            if len(cs1_i.nonzero()[0]) == 0:
                restored.append(np.zeros((600,)))
                continue

            cs1_restored_signal = restore_cs1_signal(features_i_non_zero, cs1_i, cs1_matrix)
            restored.append(cs1_restored_signal)

        if write_to_disk:
            cs1_signal_path = os.path.join(input_path, f"{output_prefix}_K_{mask_length}_I_{image_num}.csv")
            rows = []
            for row in restored:
                try:
                    r = ",".join(map(lambda x: str(int(x)), row))
                    rows.append(r)
                except Exception as e:
                    print(e)
            to_write = "\n".join(rows)
            with open(cs1_signal_path, "w") as cs1_signal_file:
                cs1_signal_file.write(to_write)

        cs1_signal[mask_length] = restored
        logger.info(f"Finished {input_prefix} signal restoration: mask_length={mask_length}, image_num={image_num}")

    return cs1_signal


def calculate_cs1_signals_all(input_path: str,
                              features: np.array,
                              cs1_mask_range: list,
                              image_nums: list,
                              skip_indices: set,
                              write_to_disk: bool = True,
                              input_prefix: str = "",
                              matrix_prefix: str = "",
                              output_prefix: str = "",
                              processes: int = -1,
                              ) -> dict:
    cs1_signals_map = {}
    if processes is None:
        for image_num in image_nums:
            cs1_signals = calculate_cs1_signals(input_path, features, cs1_mask_range, image_num, skip_indices,
                                                write_to_disk=write_to_disk, input_prefix=input_prefix,
                                                matrix_prefix=matrix_prefix, output_prefix=output_prefix)
            cs1_signals_map[image_num] = cs1_signals
    else:
        process_num = len(image_nums) if processes == -1 else processes
        with mp.Pool(process_num) as pool:
            params = [(input_path, features, cs1_mask_range, image_num, skip_indices, write_to_disk,
                       input_prefix, matrix_prefix, output_prefix)
                      for image_num in image_nums]
            cs1_signals_list = pool.starmap(calculate_cs1_signals, params)
            cs1_signals_map = dict(zip(image_nums, cs1_signals_list))

    return cs1_signals_map
