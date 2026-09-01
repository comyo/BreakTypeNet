"""Minimal model-shape check that does not require the dataset or DINOv2."""

import torch

from model import BreakTypeNet


def main() -> None:
    model = BreakTypeNet().eval()
    with torch.no_grad():
        logits, patch_features = model(torch.zeros(1, 1, 3, 224, 224))
    assert logits.shape == (1, 3)
    assert patch_features.shape == (1, 256, 768)
    print("BreakTypeNet smoke test passed")


if __name__ == "__main__":
    main()
