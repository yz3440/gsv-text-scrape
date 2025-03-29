import numpy as np
import cv2


def xyzpers(h_fov, v_fov, u, v, out_hw, in_rot):
    out = np.ones((*out_hw, 3), np.float32)

    x_max = np.tan(h_fov / 2)
    y_max = np.tan(v_fov / 2)
    x_rng = np.linspace(-x_max, x_max, num=out_hw[1], dtype=np.float32)
    y_rng = np.linspace(-y_max, y_max, num=out_hw[0], dtype=np.float32)
    out[..., :2] = np.stack(np.meshgrid(x_rng, -y_rng), -1)

    Rx = rotation_matrix(v, [1, 0, 0])
    Ry = rotation_matrix(u, [0, 1, 0])
    Ri = rotation_matrix(in_rot, np.dot([0, 0, 1.0], Rx).dot(Ry))

    out = out.dot(Rx).dot(Ry).dot(Ri)
    return out


def xyz2uv(xyz):
    """
    Converts XYZ coordinates to UV coordinates.
    xyz: ndarray of shape [..., 3]
    """
    x, y, z = np.split(xyz, 3, axis=-1)
    u = np.arctan2(x, z)
    hyp = np.hypot(x, z)
    v = np.arctan2(y, hyp)
    return np.concatenate([u, v], axis=-1)


def uv2coor(uv, h, w):
    """
    Converts UV coordinates to image pixel coordinates.
    uv: ndarray of shape [..., 2]
    h: height of the equirectangular image
    w: width of the equirectangular image
    """
    u, v = np.split(uv, 2, axis=-1)
    coor_x = (u / (2 * np.pi) + 0.5) * (w - 1)
    coor_y = (-v / np.pi + 0.5) * (h - 1)
    return np.concatenate([coor_x, coor_y], axis=-1)


def rotation_matrix(angle, axis):
    """
    Creates a rotation matrix for a given angle and axis.
    angle: rotation angle in radians
    axis: rotation axis as a 3-element array
    """
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)
    a = np.cos(angle / 2.0)
    b, c, d = -axis * np.sin(angle / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )


def sample_equirec(e_img, coor_xy, order):
    """
    Samples an equirectangular image using given coordinates.
    e_img:   ndarray in shape of [H, W, C]
    coor_xy: ndarray of coordinates with shape [H_out, W_out, 2]
    order:   interpolation order (0: nearest, 1: bilinear)
    """
    h, w = e_img.shape[:2]
    coor_x = coor_xy[..., 0].astype(np.float32)
    coor_y = coor_xy[..., 1].astype(np.float32)

    # Set interpolation mode
    if order == 1:
        interpolation = cv2.INTER_LINEAR
    elif order == 0:
        interpolation = cv2.INTER_NEAREST
    else:
        raise ValueError("Order must be 0 (nearest) or 1 (bilinear)")

    # Use OpenCV's remap function
    sampled_img = cv2.remap(
        e_img, coor_x, coor_y, interpolation, borderMode=cv2.BORDER_WRAP
    )

    return sampled_img
