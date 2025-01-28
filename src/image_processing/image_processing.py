import cv2
import numpy as np
from typing import Tuple


def convert_image_to_gray(
    image: np.array
) -> np.array:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def get_binary_image(
    image: np.array,
    binary_threshold: int
) -> np.array:
    binary_threshold = min(binary_threshold, 255)
    _, image = cv2.threshold(image, binary_threshold, 255, cv2.THRESH_BINARY)
    return image


def get_gaussia_blur_image(
    image: np.array,
    blur_kernel_size: Tuple[int, int]
) -> np.array:
    image = cv2.GaussianBlur(image, blur_kernel_size, 0)
    return image