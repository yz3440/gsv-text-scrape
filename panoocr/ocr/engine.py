from enum import Enum
from typing import List, Dict, Any
from abc import ABC, abstractmethod
from PIL import Image
from .models import FlatOCRResult


class OCREngineType(Enum):
    MACOCR = "macocr"
    EASYOCR = "easyocr"
    PADDLEOCR = "paddleocr"
    TROCR = "trocr"
    FLORENCE = "florence2"


class OCREngine(ABC):
    @abstractmethod
    def __init__(self, config: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def recognize(self, image: Image.Image = None) -> List[FlatOCRResult]:
        pass
