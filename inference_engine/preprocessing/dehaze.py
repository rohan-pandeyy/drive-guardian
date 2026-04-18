import cv2
import numpy as np


def _get_dark_channel(img: np.ndarray, patch_size: int) -> np.ndarray:
    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    return cv2.erode(min_channel, kernel)


def _estimate_atmospheric_light(img: np.ndarray, dark_channel: np.ndarray, top_percent: float = 0.001) -> np.ndarray:
    h, w = dark_channel.shape
    flat_dark = dark_channel.reshape(-1)
    flat_img = img.reshape(-1, 3)

    num_pixels = max(int(h * w * top_percent), 1)
    indices = np.argpartition(flat_dark, -num_pixels)[-num_pixels:]
    brightest = flat_img[indices]

    # Pick the brightest RGB vector among top dark-channel candidates.
    return brightest[np.argmax(np.sum(brightest, axis=1))]


def dcp_dehaze(
    bgr_frame: np.ndarray,
    omega: float = 0.95,
    t0: float = 0.1,
    patch_size: int = 15,
) -> np.ndarray:
    """Apply Dark Channel Prior dehazing on a BGR uint8 frame.

    This implementation favors realtime throughput over heavy matting refinement.
    """
    if bgr_frame is None or bgr_frame.size == 0:
        return bgr_frame

    if bgr_frame.dtype != np.uint8:
        frame = np.clip(bgr_frame, 0, 255).astype(np.uint8)
    else:
        frame = bgr_frame

    img = frame.astype(np.float32) / 255.0

    dark = _get_dark_channel(img, patch_size)
    atm = _estimate_atmospheric_light(img, dark)

    normalized = img / np.maximum(atm, 1e-6)
    transmission = 1.0 - omega * _get_dark_channel(normalized, patch_size)
    transmission = np.clip(transmission, t0, 1.0)

    # Recover scene radiance J(x) = (I(x) - A) / t(x) + A
    recovered = (img - atm) / transmission[:, :, None] + atm
    recovered = np.clip(recovered, 0.0, 1.0)

    return (recovered * 255.0).astype(np.uint8)
