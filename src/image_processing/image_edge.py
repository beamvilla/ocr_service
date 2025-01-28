import cv2
import numpy as np
from typing import List, Tuple

from imutils import grab_contours, resize
from imutils.contours import sort_contours


def get_contours(
    preprocessed_image: np.array
) -> Tuple[np.array]:
    """
    preprocessed_image: image which pass through blur, binary threshold method. 
    """
    contours = cv2.findContours(
        image=255 - preprocessed_image,
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_SIMPLE
    )
    contours = grab_contours(contours)
    contours = sort_contours(contours, method="left-to-right")[0]
    return contours


def get_char_boxes(
    contours: Tuple[np.array],
    image: np.array,
    image_shape: Tuple[int, int] = (32, 32)
) -> Tuple[List[np.array], List[Tuple[int, int, int, int]]]:
    """
    image_shape: (w, h)
    """
    chars = []
    boxes = []
    resized_image_width, resized_image_height = image_shape
  
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        
        boxes.append((x, y, w, h))  # Keep box position for plotting
        pos = image[y:y + h, x:x + w]
        _, bi = cv2.threshold(pos, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)  # Extract only image array

        (tH, tW) = bi.shape
        if tW > tH:
            bi = resize(bi, width=resized_image_width)
        else:
            bi = resize(bi, height=resized_image_height)

        (tH, tW) = bi.shape
        dX = int(max(0, resized_image_width - tW) / 2.0)
        dY = int(max(0, resized_image_height - tH) / 2.0)

        # Border image
        padded = cv2.copyMakeBorder(
            bi,
            top=dY, 
            bottom=dY,
            left=dX, 
            right=dX,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )

        char = cv2.resize(padded, (resized_image_width, resized_image_height))
        char = char.astype(np.float32) / 255.0  # Normalize
        char = np.stack((char, ) * 3, axis = -1) # Expand gray to 3 Channels
        chars.append(char)
    return chars, boxes
    