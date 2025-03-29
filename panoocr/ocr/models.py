from typing import List, Tuple
from dataclasses import dataclass
import math


@dataclass
class BoundingBox:
    # distance form the top-left corner of the image
    left: float
    top: float
    right: float
    bottom: float
    width: float
    height: float


@dataclass
class FlatOCRResult:
    text: str
    confidence: float
    bounding_box: BoundingBox
    engine: str | None = None

    def to_dict(self):
        return {
            "text": self.text,
            "confidence": self.confidence,
            "bounding_box": {
                "left": self.bounding_box.left,
                "top": self.bounding_box.top,
                "right": self.bounding_box.right,
                "bottom": self.bounding_box.bottom,
                "width": self.bounding_box.width,
                "height": self.bounding_box.height,
            },
            "engine": self.engine,
        }

    def __uv_to_yaw_pitch(
        self, horizontal_fov: float, vertical_fov: float, u: float, v: float
    ):
        """
        Convert the UV coordinate to yaw and pitch using the camera parameters.
        All the parameters are in degrees.

        Args:
            horizontal_fov (float): horizontal field of view of the camera
            vertical_fov (float): vertical field of view of the camera
            u (float): horizontal coordinate on flat image
            v (float): vertical coordinate on flat image

        Returns:
            Tuple[float, float]: the converted yaw and pitch
        """

        if horizontal_fov is None or vertical_fov is None or u is None or v is None:
            raise ValueError("Missing parameters")

        if horizontal_fov < 0 or vertical_fov < 0:
            raise ValueError("FOV must be positive")

        # Translate the origin to the center of the image
        u = u - 0.5
        v = 0.5 - v

        yaw = math.atan2(2 * u * math.tan(math.radians(horizontal_fov) / 2), 1)
        pitch = math.atan2(2 * v * math.tan(math.radians(vertical_fov) / 2), 1)

        return math.degrees(yaw), math.degrees(pitch)

    def to_sphere(
        self,
        horizontal_fov: float,
        vertical_fov: float,
        yaw_offset: float,
        pitch_offset: float,
    ):
        """
        Convert the flat OCR result to a spherical OCR result using the camera parameters.
        All the parameters are in degrees.

        Args:
            horizontal_fov (float): horizontal field of view of the camera
            vertical_fov (float): vertical field of view of the camera
            yaw_offset (float): horizontal offset of the camera
            pitch_offset (float): vertical offset of the camera

        Returns:
            SphereOCRResult: the converted OCR result
        """

        # check if all the parameters are present
        if (
            horizontal_fov is None
            or vertical_fov is None
            or yaw_offset is None
            or pitch_offset is None
        ):
            print(horizontal_fov, vertical_fov, yaw_offset, pitch_offset)
            raise ValueError("Missing parameters")
        if horizontal_fov < 0 or vertical_fov < 0:
            raise ValueError("FOV must be positive")

        centerX = (self.bounding_box.left + self.bounding_box.right) * 0.5
        centerY = (self.bounding_box.top + self.bounding_box.bottom) * 0.5

        center_yaw, center_pitch = self.__uv_to_yaw_pitch(
            horizontal_fov, vertical_fov, centerX, centerY
        )

        left_yaw, top_pitch = self.__uv_to_yaw_pitch(
            horizontal_fov, vertical_fov, self.bounding_box.left, self.bounding_box.top
        )

        right_yaw, bottom_pitch = self.__uv_to_yaw_pitch(
            horizontal_fov,
            vertical_fov,
            self.bounding_box.right,
            self.bounding_box.bottom,
        )

        width = right_yaw - left_yaw
        height = top_pitch - bottom_pitch

        return SphereOCRResult(
            text=self.text,
            confidence=self.confidence,
            yaw=center_yaw + yaw_offset,
            pitch=center_pitch + pitch_offset,
            width=width,
            height=height,
            engine=self.engine,
        )





@dataclass
class SphereOCRResult:
    text: str
    confidence: float
    yaw: float
    pitch: float
    width: float
    height: float
    engine: str | None = None

    def to_dict(self):
        return {
            "text": self.text,
            "confidence": self.confidence,
            "yaw": self.yaw,
            "pitch": self.pitch,
            "width": self.width,
            "height": self.height,
            "engine": self.engine,
        }
