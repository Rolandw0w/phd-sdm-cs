import os

import numpy as np
import sdm as sdm_lib

import data_wrangling as dw


PROLOGUE = """SDM ADDRESS SPACE
SDM-Version: v1.0.0
Format: base64
Order-of-bytes: little-endian
Bits-per-Bitstring: 640
Bits: 600
Sample: 3000000

"""


def rewrite_addresses(input_path: str, output_path: str):
    with open(input_path, "r") as input_addresses_file:
        with open(output_path, "w") as output_addresses_file:
            output_addresses_file.write(PROLOGUE)
            for line in input_addresses_file:
                normalized = line.strip()
                non_zero_indices = [index for index, bit in enumerate(normalized) if bit == "1"]
                bit_string = sdm_lib.Bitstring.init_zeros(600)
                for non_zero_index in non_zero_indices:
                    bit_string.set_bit(non_zero_index, 1)
                b64_bit_string = bit_string.to_b64()
                b64_bit_string_decoded = b64_bit_string.decode()
                output_addresses_file.write(b64_bit_string_decoded + "\n")


def get_model(addresses_path) -> sdm_lib.SDM:
    scanner_type = sdm_lib.SDM_SCANNER_OPENCL
    address_space = sdm_lib.AddressSpace.init_from_b64_file(addresses_path.encode("utf-8"))
    counter = sdm_lib.Counter.init_zero(600, 3_000_000)

    sdm = sdm_lib.SDM(address_space, counter, 8, scanner_type)
    return sdm


def get_bit_strings(features: np.array) -> list:
    bit_strings = []
    for index in range(features.shape[1]):
        feature_array = features[:, index]
        non_zero_indices = feature_array.nonzero()[0]
        bit_string = sdm_lib.Bitstring.init_zeros(600)
        for non_zero_index in non_zero_indices:
            bit_string.set_bit(non_zero_index, 1)
        bit_strings.append(bit_string)

    return bit_strings


def process(input_path: str, addresses_path: str, images_read: int):
    sdm = get_model(addresses_path)
    features = dw.get_features(input_path, left_slice=images_read)

    bit_strings = get_bit_strings(features)

    for bit_string in bit_strings:
        # d = []
        # for i in range(3_000_000):
        #     a = sdm.address_space.get_bitstring(i)
        #     a_bs = a.to_binary()
        #     b_bs = bit_string.to_binary()
        #     dd = bit_string.distance_to(a)
        #     d.append(dd)
        sdm.write(bit_string, bit_string)

    hamming_distances = []
    zeros = 0
    for bit_string in bit_strings:
        restored = sdm.read(bit_string)
        hamming_distance = bit_string.distance_to(restored)
        hamming_distances.append(hamming_distance)
        restored_binary_string = restored.to_binary()
        if set(restored_binary_string) == {"0"}:
            zeros += 1

    mean_distance = np.mean(hamming_distances)
    print(sdm)


if __name__ == "__main__":
    # rewrite_addresses("/home/rolandw0w/Development/PhD/output/kanerva_addresses.csv",
    #                   "/home/rolandw0w/Development/PhD/output/kanerva_addresses.as",
    #                   )
    process("/home/rolandw0w/Development/PhD/data/features.bin",
            "/home/rolandw0w/Development/PhD/output/kanerva_addresses.as",
            500
            )
