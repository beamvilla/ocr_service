import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def showImage(
    data: np.array, 
    convert: bool = False, 
    gray: bool = True,
    figsize: Tuple[int, int] = (10, 6)
) -> plt:
  
  plt.figure(figsize=figsize)
  color = None

  if convert is True:
    data = data[:, :, ::-1] # BGR -> RGB

  if gray is True:
    color = "gray"

  plt.imshow(data, cmap=color)
  plt.axis("off")
  return plt.show()


def draw_contours(
    image: np.array,
    contours: Tuple[np.array],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 1
) -> np.array:
    cv2.drawContours(
        image=image, 
        contours=contours,
        contourIdx=-1,
        color=color,
        thickness=thickness
    )
    return image