# BreakTypeNet

BreakTypeNet is a video classifier for recognizing three nearshore wave breaking types: **Spilling**, **Plunging**, and **Surging**. It combines a shared LiteViT frame encoder, an LSTM temporal branch, and training-stage knowledge distillation from a frozen DINOv2 ViT-B/14 teacher.

This repository contains the model and the deterministic six-fold leave-one-site-out (LOSO) training pipeline. Data, model weights, and experiment outputs are not included.

## Model

Each input is a 30 s clip represented by 60 RGB frames at 224 x 224 pixels.

- **LiteViT:** 14 x 14 patches, 768-dimensional embeddings, six Transformer blocks, and 12 attention heads.
- **Spatial branch:** the `[CLS]` token from each frame is classified, and the frame-wise logits are averaged over time.
- **Temporal branch:** patch tokens are spatially averaged into frame-level vectors and passed to a three-layer unidirectional LSTM with a hidden size of 256.
- **Fusion:** the video-level spatial logits and LSTM temporal logits are added before Softmax.
- **Knowledge distillation:** a frozen DINOv2 ViT-B/14 teacher provides temporally averaged patch features for feature imitation and patch-relation losses. The teacher and feature adapter are removed at inference.

The training objective uses equal weights for task supervision, feature imitation, and relational distillation (`alpha = beta = gamma = 1.0`). It does not use teacher logits, temperature-based distillation, KL divergence, or GroupDRO.

## Repository Contents

```text
.
|-- model.py
|-- loss.py
|-- train_loso_breaktypenet.py
|-- smoke_test.py
|-- requirements.txt
|-- LICENSE
`-- README.md
```

## Installation

Python 3.10, PyTorch 2.4.0, and torchvision 0.19.0 were used during development. A CUDA-capable GPU is required for training and teacher-feature extraction.

```bash
conda create -n breaktypenet python=3.10
conda activate breaktypenet
pip install -r requirements.txt
```

PyTorch CUDA builds are platform-specific. If the command above installs a CPU-only build, install the appropriate PyTorch package for your CUDA environment before running the training scripts.

DINOv2 is loaded from the official `facebookresearch/dinov2` repository through `torch.hub`. The first online run therefore requires network access. A local clone can instead be supplied with `--dinov2-repo`.

Check the local model implementation without downloading DINOv2 or the dataset:

```bash
python smoke_test.py
```

## Data

The VWBT-9000 dataset is available from Figshare: https://doi.org/10.6084/m9.figshare.28814993. Data and prepared tensors are not distributed with this repository.

## Citation

Please cite the VWBT-9000 dataset paper when using the data:

> Yin, H., Cai, F., Qi, H., et al. (2025). A Video Dataset for Nearshore Wave Breaking Type Classification. *Scientific Data*, 12, 1722. https://doi.org/10.1038/s41597-025-06005-5

## License

The code in this repository is released under the MIT License. DINOv2 and the VWBT-9000 dataset are subject to their own licenses and terms of use.
