from enum import Enum
from typing import List, Dict, Any
from ..engine import OCREngine
from ..models import FlatOCRResult, BoundingBox
from dataclasses import dataclass
from PIL import Image
import numpy as np


class Florence2OCREngine(OCREngine):
    def __init__(self, config: Dict[str, Any] = {}) -> None:
        from transformers import AutoProcessor, AutoModelForCausalLM

        self.device = self.get_best_device()
        self.dtype = self.get_torch_dtype()

        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-large", torch_dtype=self.dtype, trust_remote_code=True
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-large", trust_remote_code=True
        )
        self.prompt = "<OCR_WITH_REGION>"

    def get_best_device(self):
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return "cpu"

    def get_torch_dtype(self):
        import torch

        if torch.cuda.is_available():
            return torch.float16

        return torch.float32

    def recognize(self, image: Image.Image) -> List[FlatOCRResult]:
        inputs = self.processor(text=self.prompt, images=image, return_tensors="pt").to(
            self.device, self.dtype
        )
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
        )

        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task="<OCR_WITH_REGION>",
            image_size=(image.width, image.height),
        )
        florence2_ocr_results = []
        try:
            for quad_box, label in zip(
                parsed_answer["<OCR_WITH_REGION>"]["quad_boxes"],
                parsed_answer["<OCR_WITH_REGION>"]["labels"],
            ):
                # replace all </s> with empty string
                label = label.replace("</s>", "")
                # replace all <s> with empty string
                label = label.replace("<s>", "")

                florence2_ocr_results.append(
                    Florence2OCRResult(
                        text=label,
                        bounding_box=[
                            [quad_box[0], quad_box[1]],
                            [quad_box[2], quad_box[3]],
                            [quad_box[4], quad_box[5]],
                            [quad_box[6], quad_box[7]],
                        ],
                        image_width=image.width,
                        image_height=image.height,
                    )
                )
        except KeyError:
            print("Error parsing OCR results, returning empty list")

        flat_ocr_results = [
            florence2_ocr_result.to_flat()
            for florence2_ocr_result in florence2_ocr_results
        ]

        return flat_ocr_results


@dataclass
class Florence2OCRResult:
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
            confidence=1.0,
            bounding_box=BoundingBox(
                left=left / self.image_width,
                top=top / self.image_height,
                right=right / self.image_width,
                bottom=bottom / self.image_height,
                width=(right - left) / self.image_width,
                height=(bottom - top) / self.image_height,
            ),
            engine="FLORENCE_2",
        )
