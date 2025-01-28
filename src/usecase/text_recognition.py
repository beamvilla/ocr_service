import torch
import cv2
import numpy as np
from typing import Tuple, List


from src.config.config import AppConfig
from src.models.text_recognition import TextRecognitionModel
from src.image_processing import get_binary_image, get_char_boxes, get_contours


class TextRecognitionUsecase:
    def __init__(self, config: AppConfig) -> None:
        self.label_names = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        self.config = config
        self.device = torch.device(self.config.DEVICE)
        self.model = TextRecognitionModel().to(self.device)
        self.model.load_state_dict(torch.load(self.config.MODEL_PATH))
        self.model.eval()

    def preprocess_image(self, image: np.array) -> Tuple[np.array, np.array]:
        raw_gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        preprocessed_image = get_binary_image(image=raw_gray_image, binary_threshold=150)
        return preprocessed_image, raw_gray_image
    
    def extract_char_images(
        self,
        preprocessed_image: np.array,
        raw_gray_image: np.array
    ) -> Tuple[torch.Tensor, List[Tuple[int, int, int, int]]]:
        contours = get_contours(preprocessed_image=preprocessed_image)
        chars, boxes = get_char_boxes(
            contours=contours,
            image=raw_gray_image,
            image_shape=(self.config.INPUT_WIDTH, self.config.INPUT_WIDTH)
        )
        chars = torch.tensor(np.array(chars), dtype=torch.float32).unsqueeze(1)
        return chars, boxes

    def get_text(self, image: np.array) -> str:
        preprocessed_image, raw_gray_image = self.preprocess_image(image)
        chars, boxes = self.extract_char_images(
            preprocessed_image=preprocessed_image,
            raw_gray_image=raw_gray_image
        )

        with torch.no_grad():
            chars = chars.to(self.config.DEVICE)
            preds = self.model(chars)
        
        pred_word = ""

        for pred in preds:    
            pred_id = pred.argmax(0).item()
            pred_text = self.label_names[pred_id]

            pred_word += pred_text

        return pred_word