from io import BytesIO

import numpy as np
from PIL import Image, UnidentifiedImageError


def decode_jpeg_to_rgb(image_bytes: bytes) -> np.ndarray:
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            return np.asarray(image.convert("RGB"))
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Invalid JPEG frame.") from exc

