"""QR texture generation and decoding (the camera-side success test, check 6)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import qrcode


def make_qr_texture(payload: str, path: Path, pixels: int = 512) -> Path:
    """Write a square RGB PNG with the QR code and its white quiet zone.

    Version 1 (21x21 modules) with high error correction keeps modules large, so the
    code still decodes when it is small in the image or slightly tilted.
    """
    qr = qrcode.QRCode(
        version=1, error_correction=qrcode.constants.ERROR_CORRECT_H, box_size=10, border=2
    )
    qr.add_data(payload)
    qr.make(fit=True)
    img = np.array(qr.make_image(fill_color="black", back_color="white").convert("RGB"))
    img = cv2.resize(img, (pixels, pixels), interpolation=cv2.INTER_NEAREST)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return path


_DETECTOR = cv2.QRCodeDetector()


def decode_qr(rgb: np.ndarray) -> tuple[str | None, np.ndarray | None]:
    """Decode a QR code in an RGB image. Returns (payload or None, corner points or None)."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    text, points, _ = _DETECTOR.detectAndDecode(bgr)
    if text:
        return text, points
    # A mirrored image means the texture is applied the wrong way round; flag it clearly
    # instead of silently passing.
    return None, points
