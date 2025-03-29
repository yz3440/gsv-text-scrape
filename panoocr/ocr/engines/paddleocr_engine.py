from enum import Enum
from typing import List, Dict, Any
from ..engine import OCREngine
from ..models import FlatOCRResult, BoundingBox
from dataclasses import dataclass
from PIL import Image
import numpy as np


class PaddleOCRLanguageCode(Enum):
    ENGLISH = "en"
    CHINESE = "ch"
    FRENCH = "french"
    GERMAN = "german"
    KOREAN = "korean"
    JAPANESE = "japan"


DEFAULT_LANGUAGE_PREFERENCE = PaddleOCRLanguageCode.ENGLISH
DEFAULT_RECOGNIZE_UPSIDE_DOWN = False

PP_OCR_V4_SERVER = {
    "detection_model": "https://paddleocr.bj.bcebos.com/models/PP-OCRv4/chinese/ch_PP-OCRv4_det_server_infer.tar",
    "detection_yml": "https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_teacher.yml",
    "recognition_model": "https://paddleocr.bj.bcebos.com/models/PP-OCRv4/chinese/ch_PP-OCRv4_rec_server_infer.tar",
    "recognition_yml": "https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/configs/rec/PP-OCRv4/ch_PP-OCRv4_rec_hgnet.yml",
    "cls_model": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_slim_infer.tar",
}


class PaddleOCREngine(OCREngine):
    language_preference: str
    recognize_upside_down: bool
    use_v4_server: bool

    def __init__(self, config: Dict[str, Any] = {}) -> None:

        # Parse language preference
        language_perference = config.get(
            "language_preference", DEFAULT_LANGUAGE_PREFERENCE
        )
        try:
            self.language_preference = language_perference.value
        except KeyError:
            raise ValueError("Unsupported language code")

        # Parse recognition level
        recognize_upside_down = config.get(
            "recognize_upside_down", DEFAULT_RECOGNIZE_UPSIDE_DOWN
        )

        if isinstance(recognize_upside_down, bool):
            self.recognize_upside_down = recognize_upside_down
        else:
            raise ValueError("recognize_upside_down must be a boolean")

        use_v4_server = config.get("use_v4_server", False)
        if isinstance(use_v4_server, bool):
            self.use_v4_server = use_v4_server
        else:
            raise ValueError("use_v4_server must be a boolean")

        from paddleocr import PaddleOCR

        if not self.use_v4_server:
            self.ocr = PaddleOCR(
                use_angle_cls=self.recognize_upside_down,
                lang=self.language_preference,
                use_gpu=True,
            )
        else:
            # if use v4 server, download the model
            self.__download_v4_server_models()

            self.ocr = PaddleOCR(
                use_angle_cls=self.recognize_upside_down,
                # lang=self.language_preference,
                det_model_dir="./models/PP-OCRv4/chinese/ch_PP-OCRv4_det_server_infer",
                det_algorithm="DB",
                rec_model_dir="./models/PP-OCRv4/chinese/ch_PP-OCRv4_rec_server_infer",
                rec_algorithm="CRNN",
                cls_model_dir="./models/PP-OCRv4/chinese/ch_ppocr_mobile_v2.0_cls_slim_infer",
                use_gpu=True,
            )

    def __download_v4_server_models(self):
        import os
        import requests
        import tarfile

        if not os.path.exists("PP-OCRv4"):
            os.makedirs("PP-OCRv4")

        if not os.path.exists("./models/PP-OCRv4/chinese"):
            os.makedirs("./models/PP-OCRv4/chinese")

        if not os.path.exists(
            "./models/PP-OCRv4/chinese/ch_PP-OCRv4_det_server_infer.tar"
        ):
            print("Downloading detection model")
            r = requests.get(PP_OCR_V4_SERVER["detection_model"], allow_redirects=True)
            open(
                "./models/PP-OCRv4/chinese/ch_PP-OCRv4_det_server_infer.tar", "wb"
            ).write(r.content)

        if not os.path.exists("./models/PP-OCRv4/chinese/ch_PP-OCRv4_det_server_infer"):
            print("Extracting detection model")
            with tarfile.open(
                "./models/PP-OCRv4/chinese/ch_PP-OCRv4_det_server_infer.tar"
            ) as tar:
                tar.extractall("./models/PP-OCRv4/chinese")

        if not os.path.exists(
            "./models/PP-OCRv4/chinese/ch_PP-OCRv4_rec_server_infer.tar"
        ):
            print("Downloading recognition model")
            r = requests.get(
                PP_OCR_V4_SERVER["recognition_model"], allow_redirects=True
            )
            open(
                "./models/PP-OCRv4/chinese/ch_PP-OCRv4_rec_server_infer.tar", "wb"
            ).write(r.content)
        if not os.path.exists("./models/PP-OCRv4/chinese/ch_PP-OCRv4_rec_server_infer"):
            print("Extracting recognition model")
            with tarfile.open(
                "./models/PP-OCRv4/chinese/ch_PP-OCRv4_rec_server_infer.tar"
            ) as tar:
                tar.extractall("./models/PP-OCRv4/chinese")

        if not os.path.exists("./models/PP-OCRv4/chinese/ch_PP-OCRv4_det_teacher.yml"):
            print("Downloading detection yml")
            r = requests.get(PP_OCR_V4_SERVER["detection_yml"], allow_redirects=True)
            open("./models/PP-OCRv4/chinese/ch_PP-OCRv4_det_teacher.yml", "wb").write(
                r.content
            )

        if not os.path.exists("./models/PP-OCRv4/chinese/ch_PP-OCRv4_rec_hgnet.yml"):
            print("Downloading recognition yml")
            r = requests.get(PP_OCR_V4_SERVER["recognition_yml"], allow_redirects=True)
            open("./models/PP-OCRv4/chinese/ch_PP-OCRv4_rec_hgnet.yml", "wb").write(
                r.content
            )

        if not os.path.exists(
            "./models/PP-OCRv4/chinese/ch_ppocr_mobile_v2.0_cls_slim_infer.tar"
        ):
            print("Downloading cls model")
            r = requests.get(PP_OCR_V4_SERVER["cls_model"], allow_redirects=True)
            open(
                "./models/PP-OCRv4/chinese/ch_ppocr_mobile_v2.0_cls_slim_infer.tar",
                "wb",
            ).write(r.content)
        if not os.path.exists(
            "./models/PP-OCRv4/chinese/ch_ppocr_mobile_v2.0_cls_slim_infer"
        ):
            print("Extracting cls model")
            with tarfile.open(
                "./models/PP-OCRv4/chinese/ch_ppocr_mobile_v2.0_cls_slim_infer.tar"
            ) as tar:
                tar.extractall("./models/PP-OCRv4/chinese")

    def recognize(self, image: Image.Image) -> List[FlatOCRResult]:
        image_array = np.array(image)
        slice = {
            "horizontal_stride": 300,
            "vertical_stride": 500,
            "merge_x_thres": 50,
            "merge_y_thres": 35,
        }
        annotations = self.ocr.ocr(image_array, cls=True, slice=slice)
        paddle_ocr_results = []

        for annotation in annotations:
            if not isinstance(annotation, list):
                continue

            for res in annotation:
                boundingbox = res[0]
                text = res[1][0]
                confidence = res[1][1]
                paddle_ocr_results.append(
                    PaddleOCRResult(
                        text=text,
                        confidence=confidence,
                        bounding_box=boundingbox,
                        image_width=image.width,
                        image_height=image.height,
                        use_v4_server=(self.use_v4_server),
                    )
                )

        flat_ocr_results = [
            paddle_ocr_result.to_flat() for paddle_ocr_result in paddle_ocr_results
        ]

        return flat_ocr_results


@dataclass
class PaddleOCRResult:
    text: str
    bounding_box: List[List[float]]
    confidence: float
    image_width: int
    image_height: int
    use_v4_server: bool

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
            engine=("PADDLE_OCR_SERVER_V4" if self.use_v4_server else "PADDLE_OCR"),
        )
