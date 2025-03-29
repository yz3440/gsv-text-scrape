__version__ = "0.0.1"

from .image.models import PanoramaImage, PerspectiveImage, PerspectiveMetadata
from .image.constants import (
    DEFAULT_IMAGE_PERSPECTIVES,
    ZOOMED_IN_IMAGE_PERSPECTIVES,
    ZOOMED_OUT_IMAGE_PERSPECTIVES,
    ZOOMED_OUT_IMAGE_PERSPECTIVES_60,
)

from .ocr.models import FlatOCRResult, SphereOCRResult
from .ocr.engine import OCREngine
from .ocr.engines.macocr_engine import (
    MacOCRLanguageCode,
    MacOCRRecognitionLevel,
)

from .ocr.engines.paddleocr_engine import (
    PaddleOCREngine,
    PaddleOCRLanguageCode,
)

from .ocr.engines.easyocr_engine import (
    EasyOCREngine,
    EasyOCRLanguageCode,
)

from .ocr.engine import OCREngineType
from .ocr.constants import LanguageCode
from .ocr.utils import (
    visualize_ocr_results,
    create_ocr_engine,
    visualize_sphere_ocr_results,
)

from .ocr.duplication_detection import SphereOCRDuplicationDetectionEngine
