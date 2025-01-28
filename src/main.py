import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile

from src.usecase import TextRecognitionUsecase
from src.schema import APIResponseBody
from src.config.config import AppConfig


app_config = AppConfig("./config/config.yaml")
text_recognition_usecase = TextRecognitionUsecase(app_config)

app = FastAPI()

@app.post(
    "/ocr_service", 
    response_model=APIResponseBody
)
async def recognize(image: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await image.read()))
    image = np.array(image)
    text_in_image = text_recognition_usecase.get_text(image)
    return {"success": True, "text_in_image": text_in_image}