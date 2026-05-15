from __future__ import annotations

from pathlib import Path
from typing import Any


def image_to_tensor(image: Any | None, image_size: int, torch, batched: bool = False):
    if image is None:
        return zero_image_tensor(image_size, torch, batched=batched)

    image = image.convert("RGB").resize((image_size, image_size))
    data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    tensor = data.view(image_size, image_size, 3).permute(2, 0, 1)
    tensor = tensor.float() / 255.0
    return tensor.unsqueeze(0) if batched else tensor


def image_path_to_tensor(path: str | None, image_size: int, torch):
    if not path:
        return zero_image_tensor(image_size, torch)

    image_path = Path(path)
    if not image_path.exists():
        return zero_image_tensor(image_size, torch)

    try:
        from PIL import Image

        with Image.open(image_path) as image:
            return image_to_tensor(image, image_size, torch)
    except (ImportError, OSError):
        return zero_image_tensor(image_size, torch)


def zero_image_tensor(image_size: int, torch, batched: bool = False):
    shape = (1, 3, image_size, image_size) if batched else (3, image_size, image_size)
    return torch.zeros(*shape, dtype=torch.float32)
