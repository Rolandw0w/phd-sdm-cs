import argparse
import json
import logging
import os
import pickle

import cv2

from google.cloud import vision
from google.cloud.vision import types

import numpy as np


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s: %(message)s')
logger = logging.getLogger("FEATURE_GETTING")


def get_client(gcp_credentials_path):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_credentials_path

    client = vision.ImageAnnotatorClient()
    return client


def to_dict(label_annotation):
    return {
        "description": label_annotation.description,
        "score": label_annotation.score,
        "topicality": label_annotation.topicality
    }


def read_images(input_path: str, n: int = None) -> np.ndarray:
    images_batch = []

    for i in [1, 2, 3, 4, 5]:
        with open(f"{input_path}/data_batch_{i}", "rb") as f_train:
            train = pickle.load(f_train, encoding="bytes")
            images = train[b"data"]
            images_batch.append(images)

    images_list = np.concatenate(images_batch)

    if isinstance(n, int):
        images_slice = images_list[:n]
        return images_slice

    return images_list


def get_image_obj(image: np.ndarray) -> types.Image:
    image_encoded = cv2.imencode('.jpg', image)[1].tostring()
    image_obj = types.Image(content=image_encoded)
    return image_obj


def get_labels(gcp_credentials_path: str, input_path: str, n: int) -> dict:
    client = get_client(gcp_credentials_path)
    images = read_images(input_path, n)
    labels = {}

    for index in range(1, n + 1):
        try:
            image_np = images[index]
            image_obj = get_image_obj(image_np)

            response = client.label_detection(image=image_obj)
            label_annotations = response.label_annotations

            image_labels = map(lambda label_annotation: to_dict(label_annotation), label_annotations)
        except Exception as error:
            logger.warning(f"Failed to process image #{index}: {repr(error)}")
        else:
            labels[index] = image_labels

    return labels


def process(gcp_credentials_path: str, input_path: str, n: int, output_path: str):
    image_labels = get_labels(gcp_credentials_path, input_path, n)
    image_labels_json = json.dumps(image_labels, indent=4)

    with open(output_path, "w") as output_file:
        output_file.write(image_labels_json)
        logger.info(f"Labels were successfully saved to {output_path}")


def main():
    cwd = os.getcwd()
    parser = argparse.ArgumentParser()

    parser.add_argument("gcp_credentials_path", help="Path to Google Cloud Platform credentials.json")

    default_input_path = os.path.abspath(os.path.join(cwd, "../../data"))
    parser.add_argument("--input", help="Path to .bin with images", default=default_input_path)

    default_images_number = 12_000
    parser.add_argument("--N", help="Number of images to pass to Google Vision API", default=default_images_number)

    default_output_path = os.path.abspath(os.path.join(cwd, "../../data/features.json"))
    parser.add_argument("--output", help="Path to .bin with features", default=default_output_path)

    arguments = parser.parse_args()

    gcp_credentials_path = arguments.gcp_credentials_path
    if not os.path.isfile(gcp_credentials_path):
        msg = f"File {gcp_credentials_path} not found"
        logger.error(msg)
        exit(1)

    input_path = arguments.input
    if not os.path.isdir(input_path):
        msg = f"Directory {input_path} not found"
        logger.error(msg)
        exit(1)

    n = arguments.N
    try:
        n = int(n)
    except:
        msg = f"N must be integer, got {n}"
        logger.error(msg)
        exit(1)

    output_path = arguments.output

    process(gcp_credentials_path, input_path, n, output_path)


if __name__ == "__main__":
    main()
