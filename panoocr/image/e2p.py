import numpy as np
import cv2

from . import utils


def e2p(e_img, fov_deg, u_deg, v_deg, out_hw, in_rot_deg=0, mode="bilinear"):
    """
    e_img:   ndarray in shape of [H, W, C]
    fov_deg: scalar or (scalar, scalar) field of view in degrees
    u_deg:   horizontal viewing angle in range [-180, 180]
    v_deg:   vertical viewing angle in range [-90, 90]
    """
    assert e_img.ndim == 3, "Input image must have shape [H, W, C]"
    h, w = e_img.shape[:2]

    # Convert degrees to radians
    try:
        h_fov = np.deg2rad(fov_deg[0])
        v_fov = np.deg2rad(fov_deg[1])
    except TypeError:
        h_fov = v_fov = np.deg2rad(fov_deg)
    in_rot = np.deg2rad(in_rot_deg)

    # Set interpolation mode
    if mode == "bilinear":
        interpolation = cv2.INTER_LINEAR
    elif mode == "nearest":
        interpolation = cv2.INTER_NEAREST
    else:
        raise NotImplementedError("Unknown mode: {}".format(mode))

    # Compute viewing angles in radians
    u = -np.deg2rad(u_deg)
    v = np.deg2rad(v_deg)

    # Compute the XYZ coordinates for the perspective view
    xyz = utils.xyzpers(h_fov, v_fov, u, v, out_hw, in_rot)

    # Convert XYZ to UV coordinates
    uv = utils.xyz2uv(xyz)

    # Convert UV coordinates to pixel coordinates
    coor_xy = utils.uv2coor(uv, h, w)

    # Adjust coordinates for OpenCV (requires float32)
    map_x = coor_xy[..., 0].astype(np.float32)
    map_y = coor_xy[..., 1].astype(np.float32)

    # Use OpenCV's remap function for efficient sampling
    pers_img = cv2.remap(e_img, map_x, map_y, interpolation, borderMode=cv2.BORDER_WRAP)

    return pers_img
