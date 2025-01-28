import torch
import cv2
import sys
sys.path.append("./")

from src.models.text_recognition import TextRecognitionModel
from src.image_processing import *


device = torch.device("cuda")


model = TextRecognitionModel()
model.load_state_dict(torch.load("./models/model.pt"))
model.eval()


image_path = "./dataset/offline_test/Thai_ID_Card_Mockup_1.jpg"
raw_image = cv2.imread(image_path)
raw_gray_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
preprocessed_image = get_binary_image(image=raw_gray_image, binary_threshold=150)
contours = get_contours(preprocessed_image=preprocessed_image)
chars, boxes = get_char_boxes(
    contours=contours,
    image=raw_gray_image,
    image_shape=(28, 28)
)
chars = torch.tensor(chars, dtype=torch.float32).unsqueeze(1)
preds = model(chars)
label_names = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

pred_word = ""

for pred, box in zip(preds, boxes):
    x, y, w, h = box
    
    pred_id = pred.argmax(0).item()
    pred_text = label_names[pred_id]

    pred_word += pred_text
    last_x2 = x + w

print(pred_word)