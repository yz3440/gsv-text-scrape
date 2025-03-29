import math
from dataclasses import dataclass


def degreesToRadians(degrees):
    return degrees * (math.pi / 180)


def radiansToDegrees(radians):
    return radians * (180 / math.pi)


def correct_ocr_coordinates(
    ocr_yaw: float, ocr_pitch: float, street_view_pitch: float, street_view_roll: float
):
    # The original streetview pitch has 90 set as the horizon, so we need to adjust it to 0
    street_view_pitch = 90 - street_view_pitch

    # Convert degrees to radians
    ocr_yaw_rad = degreesToRadians(ocr_yaw)
    ocr_pitch_rad = degreesToRadians(ocr_pitch)
    street_view_pitch_rad = degreesToRadians(street_view_pitch)
    street_view_roll_rad = degreesToRadians(street_view_roll)

    # Calculate corrected yaw
    x = math.cos(ocr_yaw_rad) * math.cos(ocr_pitch_rad)
    y = math.sin(ocr_yaw_rad) * math.cos(ocr_pitch_rad)
    z = math.sin(ocr_pitch_rad)

    # Apply roll rotation
    x2 = x
    y2 = y * math.cos(street_view_roll_rad) - z * math.sin(street_view_roll_rad)
    z2 = y * math.sin(street_view_roll_rad) + z * math.cos(street_view_roll_rad)

    # Apply pitch rotation
    x3 = x2 * math.cos(street_view_pitch_rad) + z2 * math.sin(street_view_pitch_rad)
    y3 = y2
    z3 = -x2 * math.sin(street_view_pitch_rad) + z2 * math.cos(street_view_pitch_rad)

    # Calculate corrected yaw and pitch
    correctedYaw = radiansToDegrees(math.atan2(y3, x3))
    correctedPitch = radiansToDegrees(math.asin(z3))

    return correctedYaw, correctedPitch


def fov_to_zoom_level(fov: float) -> float:
    if fov <= 0 or fov >= 180:
        raise ValueError("FOV must be between 0 and 180 degrees")

    zoom_level = math.log2(360 / fov)
    return round(zoom_level * 100) / 100


@dataclass
class GoogleStreetViewProps:
    lat: float
    lng: float
    panorama_id: str
    heading: float
    pitch: float
    tilt: float
    fov: float
    zoom: float


def get_google_streetview_props(
    panorama_id: str,
    lat: float,
    lng: float,
    ocr_yaw: float,
    ocr_pitch: float,
    street_view_heading: float,
    street_view_pitch: float,
    street_view_roll: float,
    ocr_width: float,
    ocr_height: float,
    FOV_FACTOR: float = 2,
    FOV_MIN: float = 10,
) -> GoogleStreetViewProps:
    ocr_yaw, ocr_pitch = correct_ocr_coordinates(
        ocr_yaw, ocr_pitch, street_view_pitch, street_view_roll
    )

    ocr_bounding_degree = math.ceil(max(ocr_width, ocr_height) * FOV_FACTOR)
    fov = max(ocr_bounding_degree, FOV_MIN)
    zoom = fov_to_zoom_level(fov)

    return GoogleStreetViewProps(
        panorama_id=panorama_id,
        lat=lat,
        lng=lng,
        heading=(ocr_yaw + street_view_heading) % 360,
        pitch=ocr_pitch,
        tilt=ocr_pitch + 90,
        fov=fov,
        zoom=zoom,
    )


def get_google_streetview_url(gsv_prop: GoogleStreetViewProps) -> str:
    return f"https://www.google.com/maps/@{gsv_prop.lat},{gsv_prop.lng},3a,20y,{round(gsv_prop.heading, 2)}h,{round(gsv_prop.tilt, 2)}t/data=!3m6!1e1!3m4!1s{gsv_prop.panorama_id}!2e0!7i16384!8i8192?entry=ttu"


def get_google_streetview_embed_url(
    gsv_prop: GoogleStreetViewProps, GOOGLE_MAPS_API_KEY: str
) -> str:
    if not GOOGLE_MAPS_API_KEY:
        raise ValueError("GOOGLE_MAPS_API_KEY is required")

    return f"https://www.google.com/maps/embed/v1/streetview?key={GOOGLE_MAPS_API_KEY}&location={gsv_prop.lat},{gsv_prop.lng}&pano={gsv_prop.panorama_id}&heading={gsv_prop.heading}&pitch={gsv_prop.pitch}&fov={gsv_prop.fov}&zoom={gsv_prop.zoom}"
