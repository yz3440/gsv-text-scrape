from enum import Enum
from typing import List, Dict, Any
from ..engine import OCREngine
from ..models import FlatOCRResult, BoundingBox
from dataclasses import dataclass
from PIL import Image
import numpy as np


class TrOCRModel(Enum):
    MICROSOFT_TROCR_LARGE_PRINTED = "microsoft/trocr-large-printed"


DEFAULT_MODEL = TrOCRModel.MICROSOFT_TROCR_LARGE_PRINTED


class TrOCREngine(OCREngine):
    model: str

    def __init__(self, config: Dict[str, Any] = {}) -> None:

        # Parse model
        model = config.get("model", DEFAULT_MODEL)

        try:
            self.model = model.value
        except KeyError:
            raise ValueError("Unsupported model")

        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self.processor = TrOCRProcessor.from_pretrained(self.model)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.model)

    def recognize(self, image: Image.Image) -> List[FlatOCRResult]:
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        print("Generated text:")
        print(generated_text)

        flat_ocr_results = []

        return flat_ocr_results


@dataclass
class TrOCRResult:
    text: str
    bounding_box: List[List[float]]
    image_width: int
    image_height: int

    def to_flat(self):
        left = min(
            self.bounding_box[0][0],
            self.bounding_box[1][0],
            self.bounding_box[2][0],
            self.bounding_box[3][0],
        )
        right = max(
            self.bounding_box[0][0],
            self.bounding_box[1][0],
            self.bounding_box[2][0],
            self.bounding_box[3][0],
        )
        bottom = max(
            self.bounding_box[0][1],
            self.bounding_box[1][1],
            self.bounding_box[2][1],
            self.bounding_box[3][1],
        )
        top = min(
            self.bounding_box[0][1],
            self.bounding_box[1][1],
            self.bounding_box[2][1],
            self.bounding_box[3][1],
        )

        return FlatOCRResult(
            text=self.text,
            confidence=self.confidence,
            bounding_box=BoundingBox(
                left=left / self.image_width,
                top=top / self.image_height,
                right=right / self.image_width,
                bottom=bottom / self.image_height,
                width=(right - left) / self.image_width,
                height=(bottom - top) / self.image_height,
            ),
            engine="TR_OCR",
        )
