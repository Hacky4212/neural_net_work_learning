from __future__ import annotations

import importlib.util


def main() -> None:
    checks = {
        "Pillow": "PIL",
        "PyTorch": "torch",
        "TorchVision": "torchvision",
    }
    for name, module in checks.items():
        ok = importlib.util.find_spec(module) is not None
        print(f"{name}: {'OK' if ok else 'MISSING'}")


if __name__ == "__main__":
    main()
