from typing import List
from PIL import Image, ImageDraw, ImageFont
from .models import FlatOCRResult, SphereOCRResult
from .engine import OCREngine, OCREngineType
from typing import Any


def create_ocr_engine(engine_type: OCREngineType, config: Any) -> OCREngine:
    if engine_type == OCREngineType.MACOCR:
        from .engines.macocr_engine import MacOCREngine

        print("Initializing OCR Engine: MacOCR")
        return MacOCREngine(config)
    elif engine_type == OCREngineType.EASYOCR:
        from .engines.easyocr_engine import EasyOCREngine

        print("Initializing OCR Engine: EasyOCR")
        return EasyOCREngine(config)
    elif engine_type == OCREngineType.PADDLEOCR:
        from .engines.paddleocr_engine import PaddleOCREngine

        print("Initializing OCR Engine: PaddleOCR")
        return PaddleOCREngine(config)
    elif engine_type == OCREngineType.TROCR:
        from .engines.trocr_engine import TrOCREngine

        print("Initializing OCR Engine: TrOCR")
        return TrOCREngine(config)
    elif engine_type == OCREngineType.FLORENCE:
        from .engines.florence2_engine import Florence2OCREngine

        print("Initializing OCR Engine: Florence2")
        return Florence2OCREngine(config)
    else:
        raise ValueError(f"Unsupported OCR engine type: {engine_type}")


def visualize_ocr_results(
    image: Image.Image,
    ocr_results: List[FlatOCRResult],
    font_size: int = 16,
    highlight_color: str = "red",
    stroke_width: int = 2,
) -> Image.Image:
    """
    Visualize OCR results on an image.

    Args:
        image: The image to visualize OCR results on.
        ocr_results: A list of OCR results.

    Returns:
        Image.Image: The image with OCR results visualized.
    """

    draw = ImageDraw.Draw(image)
    width, height = image.size

    font = ImageFont.load_default(size=font_size)

    for ocr_result in ocr_results:
        draw.rectangle(
            [
                ocr_result.bounding_box.left * width,
                ocr_result.bounding_box.top * height,
                ocr_result.bounding_box.right * width,
                ocr_result.bounding_box.bottom * height,
            ],
            outline=highlight_color,
            width=3,
        )
        # Draw the text
        draw.text(
            (
                ocr_result.bounding_box.left * width,
                ocr_result.bounding_box.top * height
                - font_size
                - stroke_width,  # Move the text up so it doesn't overlap the bounding box
            ),
            ocr_result.text,
            fill=highlight_color,
            stroke_fill="white",
            stroke_width=stroke_width,
            font=font,
        )
        # Draw the confidence
        draw.text(
            (
                ocr_result.bounding_box.left * width,
                ocr_result.bounding_box.bottom * height
                + stroke_width,  # Move the text down so it doesn't overlap the bounding box
            ),
            f"{ocr_result.confidence:.2f}",
            fill=highlight_color,
            stroke_fill="white",
            stroke_width=stroke_width,
            font=font,
        )

    return image


def visualize_sphere_ocr_results(
    image: Image.Image,
    ocr_results: List[SphereOCRResult],
    font_size: int = 16,
    highlight_color: str = "red",
    stroke_width: int = 2,
    inplace: bool = True,
) -> Image.Image:
    """
    Visualize Sphere OCR results on an Equirectangular image.
    This is SLOW and should only be used for debugging purposes.

    Args:
        image: The Equirectangular image to visualize OCR results on.
        ocr_results: A list of Sphere OCR results.

    """
    import numpy as np
    from scipy.ndimage import map_coordinates

    # Convert image to RGBA
    image = image.convert("RGBA")

    def get_ocr_result_image(ocr_result: SphereOCRResult) -> Image.Image:
        PIXEL_PER_DEGREE = 300

        text_image = Image.new(
            "RGBA",
            (
                int(ocr_result.width * PIXEL_PER_DEGREE),
                int(ocr_result.height * PIXEL_PER_DEGREE),
            ),
            (255, 255, 255, 0),
        )

        draw = ImageDraw.Draw(text_image)
        font = ImageFont.load_default(size=ocr_result.height * PIXEL_PER_DEGREE * 0.2)

        draw.rectangle(
            [0, 0, text_image.width, text_image.height],
            outline=highlight_color,
            width=3,
            fill=(255, 255, 255, 0),
        )

        draw.text(
            (text_image.width / 2, text_image.height / 2),
            ocr_result.text,
            fill=highlight_color,
            anchor="mm",
            stroke_fill="white",
            stroke_width=stroke_width,
            font=font,
        )
        return text_image

    def place_ocr_result_on_panorama(panorama_image, ocr_result):
        ocr_result_image = get_ocr_result_image(ocr_result)

        ocr_result_image = np.array(ocr_result_image)

        pano_height, pano_width = panorama_image.shape[:2]
        ocr_result_height, ocr_result_width = ocr_result_image.shape[:2]

        yaw_rad = np.radians(-ocr_result.yaw)
        pitch_rad = np.radians(ocr_result.pitch)
        width_rad = np.radians(ocr_result.width)
        height_rad = np.radians(ocr_result.height)

        # Create coordinate mappings for the panorama
        y_pano, x_pano = np.mgrid[0:pano_height, 0:pano_width]

        # Convert panorama coordinates to spherical coordinates
        lon = (x_pano / pano_width - 0.5) * 2 * np.pi
        lat = (0.5 - y_pano / pano_height) * np.pi

        # Calculate the 3D coordinates on the unit sphere
        x = np.cos(lat) * np.sin(lon)
        y = np.sin(lat)
        z = np.cos(lat) * np.cos(lon)

        # Combine rotation matrices
        sin_yaw, cos_yaw = np.sin(yaw_rad), np.cos(yaw_rad)
        sin_pitch, cos_pitch = np.sin(pitch_rad), np.cos(pitch_rad)

        # Apply rotation
        x_rot = cos_yaw * x + sin_yaw * z
        y_rot = sin_pitch * sin_yaw * x + cos_pitch * y - sin_pitch * cos_yaw * z
        z_rot = -cos_pitch * sin_yaw * x + sin_pitch * y + cos_pitch * cos_yaw * z

        # Project onto the plane
        x_proj = x_rot / (z_rot + 1e-8)  # Add small epsilon to avoid division by zero
        y_proj = y_rot / (z_rot + 1e-8)

        # Scale and shift to image coordinates
        x_img = (x_proj / np.tan(width_rad / 2) + 1) * ocr_result_width / 2
        y_img = (-y_proj / np.tan(height_rad / 2) + 1) * ocr_result_height / 2

        # Create mask for valid coordinates
        mask = (
            (x_img >= 0)
            & (x_img < ocr_result_width)
            & (y_img >= 0)
            & (y_img < ocr_result_height)
            & (z_rot > 0)
        )

        # Use map_coordinates to sample from the image
        warped_channels = []
        for channel in range(ocr_result_image.shape[2]):
            warped_channel = map_coordinates(
                ocr_result_image[:, :, channel],
                [y_img, x_img],
                order=1,
                mode="constant",
                cval=0,
            )
            warped_channels.append(warped_channel)

        warped_image = np.stack(warped_channels, axis=-1)

        # save the warped_image
        # temp_warped_image = Image.fromarray(warped_image)
        # temp_warped_image.save("assets/warped_image.png")

        if not inplace:
            result = panorama_image.copy()
        else:
            result = panorama_image

        # Apply alpha compositing
        alpha = warped_image[:, :, 3] / 255.0
        for c in range(3):  # RGB channels
            result[:, :, c] = result[:, :, c] * (1 - alpha * mask) + warped_image[
                :, :, c
            ] * (alpha * mask)

        # Update alpha channel
        result[:, :, 3] = np.maximum(result[:, :, 3], warped_image[:, :, 3] * mask)

        return result

    new_image = np.array(image)

    for ocr_result in ocr_results:
        if inplace:
            place_ocr_result_on_panorama(new_image, ocr_result)
        else:
            new_image = place_ocr_result_on_panorama(new_image, ocr_result)

    new_image = Image.fromarray(new_image)

    # convert back to rgb
    new_image = new_image.convert("RGB")

    return new_image
