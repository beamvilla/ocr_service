from pydantic import BaseModel


class APIResponseBody(BaseModel):
    success: bool
    text_in_image: str