from enum import Enum
from typing import List, Dict, Any, Tuple
from ..engine import OCREngine
from ..models import FlatOCRResult, BoundingBox
from dataclasses import dataclass
from PIL import Image


class MacOCRRecognitionLevel(Enum):
    FAST = "fast"
    ACCURATE = "accurate"


class MacOCRLanguageCode(Enum):
    ENGLISH_US = "en-US"
    FRENCH_FR = "fr-FR"
    ITALIAN_IT = "it-IT"
    GERMAN_DE = "de-DE"
    SPANISH_ES = "es-ES"
    PORTUGUESE_BR = "pt-BR"
    CHINESE_SIMPLIFIED = "zh-Hans"
    CHINESE_TRADITIONAL = "zh-Hant"
    CHINESE_YUE_SIMPLIFIED = "yue-Hans"
    CHINESE_YUE_TRADITIONAL = "yue-Hant"
    KOREAN_KR = "ko-KR"
    JAPANESE_JP = "ja-JP"
    RUSSIAN_RU = "ru-RU"
    UKRAINIAN_UA = "uk-UA"
    THAI_TH = "th-TH"
    VIETNAMESE_VT = "vi-VT"


DEFAULT_LANGUAGE_PREFERENCE = [MacOCRLanguageCode.ENGLISH_US]
DEFAULT_RECOGNITION_LEVEL = MacOCRRecognitionLevel.ACCURATE


class MacOCREngine(OCREngine):
    language_preference: List[str]
    recognition_level: str

    def __init__(self, config: Dict[str, Any] = {}) -> None:

        # Parse language preference
        language_perference = config.get(
            "language_preference", DEFAULT_LANGUAGE_PREFERENCE
        )
        try:
            self.language_preference = [
                language_code.value for language_code in language_perference
            ]
        except KeyError:
            raise ValueError("Unsupported language code")

        # Parse recognition level
        recognition_level = config.get("recognition_level", DEFAULT_RECOGNITION_LEVEL)

        try:
            self.recognition_level = recognition_level.value
        except KeyError:
            raise ValueError("Unsupported recognition level")

    def recognize(self, image: Image.Image) -> List[FlatOCRResult]:
        import ocrmac.ocrmac

        annotations = ocrmac.ocrmac.OCR(
            image,
            recognition_level=self.recognition_level,
            language_preference=self.language_preference,
        ).recognize()

        mac_ocr_results = [
            MacOCRResult(
                text=annotation[0],
                confidence=annotation[1],
                bounding_box=annotation[2],
            )
            for annotation in annotations
        ]

        flat_ocr_results = [
            mac_ocr_result.to_flat() for mac_ocr_result in mac_ocr_results
        ]

        return flat_ocr_results


@dataclass
class MacOCRResult:
    text: str
    bounding_box: Tuple[float, float, float, float]
    confidence: float

    def to_flat(self):
        left = self.bounding_box[0]
        right = self.bounding_box[0] + self.bounding_box[2]
        top = 1 - self.bounding_box[1] - self.bounding_box[3]
        bottom = 1 - self.bounding_box[1]

        return FlatOCRResult(
            text=self.text,
            confidence=self.confidence,
            bounding_box=BoundingBox(
                left=left,
                top=top,
                right=right,
                bottom=bottom,
                width=right - left,
                height=bottom - top,
            ),
            engine="APPLE_VISION_FRAMEWORK",
        )
