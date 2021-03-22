import argparse
import json
import logging
import os
from collections import Counter


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s: %(message)s')
logger = logging.getLogger("FEATURE_PROCESSING")


def read_labels(features_path):
    logger.info(f"Started reading features from {features_path}")
    with open(features_path, "r") as features_file:
        features_str = features_file.read()
        features_obj = json.loads(features_str)
        return features_obj


def filter_labels(input_path: str, score_threshold):
    image_labels = read_labels(input_path)

    descriptions = set()
    filtered_labels = {}
    for image_num, labels in image_labels.items():
        for label in labels:
            if label["score"] < score_threshold:
                continue
            description = label["description"]
            descriptions.add(description)

            filtered_labels.setdefault(image_num, [])
            filtered_labels[image_num].append(label)

    descriptions_list = sorted(list(descriptions))
    descriptions_dict = {description: index for index, description in enumerate(descriptions_list)}
    return descriptions_dict, filtered_labels


def process(features_path: str, score_threshold: float, non_zero_images: int, output_path: str, max_features: int = None):
    image_descriptions, labels = filter_labels(features_path, score_threshold)

    numbers = sorted([int(x) for x in labels.keys()])

    min_number = min(numbers)
    max_number = max(numbers)

    non_zeros = 0
    bytes_list = []
    for image_number in range(min_number, max_number + 1):
        code = ["0"] * len(image_descriptions)
        image_labels = labels.get(str(image_number), [])
        for image_label in image_labels:
            index = image_descriptions.get(image_label["description"], None)
            if index is not None:
                code[index] = "1"

        counter = Counter(code)
        if counter.get("1", 0) > 1:
            non_zeros += 1
        else:
            continue

        code += ["0"] * (8 - len(code) % 8)
        max_features = max_features or len(code)
        for i in range(0, max_features, 8):
            code_byte = "".join(code[i:i+8])
            code_byte_int = int(code_byte, 2)
            bytes_to_write = code_byte_int.to_bytes(1, byteorder="big", signed=False)
            bytes_list.append(bytes_to_write)

        if non_zeros == non_zero_images:
            break

    if len(bytes_list) < non_zero_images:
        msg = "Only {} images with at least one feature were processed (minimum {} was required)".format(
            len(bytes_list), non_zero_images
        )
        logger.error(msg)
        exit(1)

    with open(output_path, "wb") as f_output:
        output = b"".join(bytes_list)
        f_output.write(output)
        logger.info(f"Features were successfully saved to {output_path}")


def main():
    cwd = os.getcwd()
    parser = argparse.ArgumentParser()

    default_input_path = os.path.abspath(os.path.join(cwd, "../../data/features.json"))
    parser.add_argument("--input", help="Path to .json with features", default=default_input_path)

    default_score_threshold = 0.7
    parser.add_argument("--score_threshold", help="Lower threshold for a feature", default=default_score_threshold)

    default_non_zero_images = 9_000
    parser.add_argument("--non_zero_images", help="Required number of images with at least one feature",
                        default=default_non_zero_images)

    default_output_path = os.path.abspath(os.path.join(cwd, "../../data/features.bin"))
    parser.add_argument("--output", help="Path to .bin with features", default=default_output_path)

    default_max_features = 600
    parser.add_argument("--max_features", help="Maximum number of features to process", default=default_max_features)

    arguments = parser.parse_args()

    features_path = arguments.input
    if not os.path.isfile(features_path):
        msg = f"File {features_path} not found"
        logger.error(msg)
        exit(1)

    score_threshold = arguments.score_threshold
    try:
        score_threshold = float(score_threshold)
    except:
        msg = f"score_threshold must be float, got {score_threshold}"
        logger.error(msg)
        exit(1)

    non_zero_images = arguments.non_zero_images
    try:
        non_zero_images = int(non_zero_images)
    except:
        msg = f"non_zero_images must be integer, got {non_zero_images}"
        logger.error(msg)
        exit(1)

    max_features = arguments.max_features
    try:
        max_features = int(max_features)
    except:
        msg = f"max_features must be integer, got {max_features}"
        logger.error(msg)
        exit(1)

    output_path = arguments.output

    process(features_path, score_threshold, non_zero_images, output_path, max_features=max_features)


if __name__ == "__main__":
    main()
