from enum import Enum
from typing import List, Dict, Any
from ..engine import OCREngine
from ..models import FlatOCRResult, BoundingBox
from dataclasses import dataclass
from PIL import Image
import numpy as np


class EasyOCRLanguageCode(Enum):
    ABAZA = "abq"
    ADYGHE = "ady"
    AFRIKAANS = "af"
    ANGIKA = "ang"
    ARABIC = "ar"
    ASSAMESE = "as"
    AVAR = "ava"
    AZERBAIJANI = "az"
    BELARUSIAN = "be"
    BULGARIAN = "bg"
    BIHARI = "bh"
    BHOJPURI = "bho"
    BENGALI = "bn"
    BOSNIAN = "bs"
    SIMPLIFIED_CHINESE = "ch_sim"
    TRADITIONAL_CHINESE = "ch_tra"
    CHECHEN = "che"
    CZECH = "cs"
    WELSH = "cy"
    DANISH = "da"
    DARGWA = "dar"
    GERMAN = "de"
    ENGLISH = "en"
    SPANISH = "es"
    ESTONIAN = "et"
    PERSIAN = "fa"
    FRENCH = "fr"
    IRISH = "ga"
    GOAN_KONKANI = "gom"
    HINDI = "hi"
    CROATIAN = "hr"
    HUNGARIAN = "hu"
    INDONESIAN = "id"
    INGUSH = "inh"
    ICELANDIC = "is"
    ITALIAN = "it"
    JAPANESE = "ja"
    KABARDIAN = "kbd"
    KANNADA = "kn"
    KOREAN = "ko"
    KURDISH = "ku"
    LATIN = "la"
    LAK = "lbe"
    LEZGHIAN = "lez"
    LITHUANIAN = "lt"
    LATVIAN = "lv"
    MAGAHI = "mah"
    MAITHILI = "mai"
    MAORI = "mi"
    MONGOLIAN = "mn"
    MARATHI = "mr"
    MALAY = "ms"
    MALTESE = "mt"
    NEPALI = "ne"
    NEWARI = "new"
    DUTCH = "nl"
    NORWEGIAN = "no"
    OCCITAN = "oc"
    PALI = "pi"
    POLISH = "pl"
    PORTUGUESE = "pt"
    ROMANIAN = "ro"
    RUSSIAN = "ru"
    SERBIAN_CYRILLIC = "rs_cyrillic"
    SERBIAN_LATIN = "rs_latin"
    NAGPURI = "sck"
    SLOVAK = "sk"
    SLOVENIAN = "sl"
    ALBANIAN = "sq"
    SWEDISH = "sv"
    SWAHILI = "sw"
    TAMIL = "ta"
    TABASSARAN = "tab"
    TELUGU = "te"
    THAI = "th"
    TAJIK = "tjk"
    TAGALOG = "tl"
    TURKISH = "tr"
    UYGHUR = "ug"
    UKRAINIAN = "uk"
    URDU = "ur"
    UZBEK = "uz"
    VIETNAMESE = "vi"


DEFAULT_LANGUAGE_PREFERENCE = [EasyOCRLanguageCode.ENGLISH]


class EasyOCREngine(OCREngine):
    language_preference: List[str]

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

        import easyocr

        self.reader = easyocr.Reader(self.language_preference, gpu=True)

    def recognize(self, image: Image.Image) -> List[FlatOCRResult]:
        image_array = np.array(image)
        annotations = self.reader.readtext(image_array)

        easy_ocr_results = []

        for annotation in annotations:
            boundingbox = annotation[0]
            text = annotation[1]
            confidence = annotation[2]
            easy_ocr_results.append(
                EasyOCRResult(
                    text=text,
                    confidence=confidence,
                    bounding_box=boundingbox,
                    image_width=image.width,
                    image_height=image.height,
                )
            )

        flat_ocr_results = [
            easy_ocr_result.to_flat() for easy_ocr_result in easy_ocr_results
        ]

        return flat_ocr_results


@dataclass
class EasyOCRResult:
    text: str
    bounding_box: List[List[float]]
    confidence: float
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
            engine="easyocr",
        )
