"""Train BreakTypeNet with the six-fold LOSO protocol used in the manuscript."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from torch import nn, optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

from loss import HeterogeneousKnowledgeDistillationLoss
from model import BreakTypeNet, FeatureAdapter


CLASS_TO_ID = {"Spilling": 0, "Plunging": 1, "Surging": 2}
ID_TO_CLASS = {value: key for key, value in CLASS_TO_ID.items()}
SITE_TO_STATION = {
    "SiteA": "NOB",
    "SiteB": "OI",
    "SiteC": "PA",
    "SiteD": "PR",
    "SiteE": "HC",
    "SiteF": "XSW",
}
CAMERA_TO_SITE = {
    "Camera_01": "SiteA", "Camera_02": "SiteA", "Camera_03": "SiteA",
    "Camera_04": "SiteB", "Camera_05": "SiteB",
    "Camera_06": "SiteC",
    "Camera_07": "SiteD",
    "Camera_08": "SiteE", "Camera_09": "SiteE",
    "Camera_10": "SiteE", "Camera_11": "SiteE",
    "Camera_12": "SiteF", "Camera_13": "SiteF",
    "Camera_14": "SiteF", "Camera_15": "SiteF",
}
CLIP_NAME_RE = re.compile(
    r"^(Spilling|Plunging|Surging)-([0-9]+)-(Camera_[0-9]{2})-([0-9]{14}Z)$"
)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@dataclass(frozen=True)
class Sample:
    path: Path
    label_name: str
    label_id: int
    camera: str
    site: str
    station: str
    timestamp: datetime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Six-fold leave-one-site-out training for BreakTypeNet."
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        required=True,
        help="Dataset root containing class/clip directories of JPG frames.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results_loso_breaktypenet"),
        help="Output directory for checkpoints, manifests, logs, and predictions.",
    )
    parser.add_argument(
        "--test-site",
        choices=["all", *SITE_TO_STATION],
        default="all",
        help="Run all six folds or one held-out site.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=1,
        help="Use >1 only when reducing --batch-size for memory; effective batch is their product.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--no-augmentation",
        action="store_true",
        help="Disable random flip, rotation, and color jitter for training clips.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=10,
        help="Refresh batch progress every N batches; use 0 to disable.",
    )
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--teacher-frame-batch", type=int, default=120)
    parser.add_argument(
        "--dinov2-repo",
        type=Path,
        default=None,
        help="Optional local facebookresearch/dinov2 repository; otherwise torch.hub is used.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate files and print all LOSO split counts without loading a model.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume incomplete folds and skip folds with saved test results.",
    )
    parser.add_argument(
        "--track-test-each-epoch",
        action="store_true",
        help="Record held-out test metrics each epoch without using them for selection.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def rng_state(loader: DataLoader) -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "loader_generator": loader.generator.get_state(),
    }


def cpu_byte_rng_state(state: object) -> torch.Tensor:
    if isinstance(state, torch.Tensor):
        return state.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    return torch.as_tensor(state, dtype=torch.uint8, device="cpu").contiguous()


def restore_rng_state(state: dict[str, object], loader: DataLoader) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(cpu_byte_rng_state(state["torch"]))
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all([
            cpu_byte_rng_state(cuda_state) for cuda_state in state["cuda"]
        ])
    loader.generator.set_state(cpu_byte_rng_state(state["loader_generator"]))


def save_checkpoint(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def clip_frames(path: Path) -> list[Path]:
    return sorted(
        frame for frame in path.iterdir()
        if frame.is_file() and frame.suffix.lower() in {".jpg", ".jpeg"}
    )


def scan_samples(image_root: Path) -> list[Sample]:
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image root does not exist: {image_root}")
    samples = []
    rejected = []
    for class_name in CLASS_TO_ID:
        class_dir = image_root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        for path in sorted(item for item in class_dir.iterdir() if item.is_dir()):
            match = CLIP_NAME_RE.match(path.name)
            if not match or match.group(1) != class_name:
                rejected.append(str(path.relative_to(image_root)))
                continue
            label_name, _, camera, timestamp_text = match.groups()
            frames = clip_frames(path)
            if camera not in CAMERA_TO_SITE or len(frames) != 60:
                rejected.append(
                    f"{path.relative_to(image_root)} ({len(frames)} JPG frames)"
                )
                continue
            site = CAMERA_TO_SITE[camera]
            samples.append(Sample(
                path=path.resolve(),
                label_name=label_name,
                label_id=CLASS_TO_ID[label_name],
                camera=camera,
                site=site,
                station=SITE_TO_STATION[site],
                timestamp=datetime.strptime(timestamp_text, "%Y%m%d%H%M%SZ"),
            ))
    if rejected:
        preview = ", ".join(rejected[:5])
        raise ValueError(f"Rejected {len(rejected)} clip directories; examples: {preview}")
    if len(samples) != 9000:
        raise ValueError(f"Expected 9,000 clips, found {len(samples)} in {image_root}")
    samples.sort(key=lambda item: (item.site, item.label_id, item.timestamp, item.path.name))
    return samples


def make_loso_split(
    samples: Iterable[Sample], test_site: str
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    if test_site not in SITE_TO_STATION:
        raise ValueError(f"Unknown test site: {test_site}")
    test = [sample for sample in samples if sample.site == test_site]
    groups: dict[tuple[str, int], list[Sample]] = defaultdict(list)
    for sample in samples:
        if sample.site != test_site:
            groups[(sample.site, sample.label_id)].append(sample)

    train, val = [], []
    for group_key in sorted(groups):
        ordered = sorted(groups[group_key], key=lambda item: (item.timestamp, item.path.name))
        split_at = math.floor(len(ordered) * 0.8)
        if len(ordered) >= 2:
            split_at = min(max(split_at, 1), len(ordered) - 1)
        train.extend(ordered[:split_at])
        val.extend(ordered[split_at:])

    train.sort(key=lambda item: (item.site, item.label_id, item.timestamp, item.path.name))
    val.sort(key=lambda item: (item.site, item.label_id, item.timestamp, item.path.name))
    # Preserve the existing result-file convention: Plunging, Spilling, Surging.
    test.sort(key=lambda item: (item.label_name, item.timestamp, item.path.name))
    if not train or not val or not test:
        raise ValueError(f"Empty split for {test_site}: train={len(train)}, val={len(val)}, test={len(test)}")
    if {sample.path for sample in train} & {sample.path for sample in val + test}:
        raise AssertionError("Train split overlaps validation or test")
    if {sample.path for sample in val} & {sample.path for sample in test}:
        raise AssertionError("Validation split overlaps test")
    return train, val, test


def split_counts(samples: Iterable[Sample]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for sample in samples:
        counts[sample.site][sample.label_name] += 1
    return {site: dict(labels) for site, labels in sorted(counts.items())}


def save_manifest(path: Path, split_name: str, samples: Iterable[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "split", "site", "station", "camera", "true_name", "y_true",
            "timestamp_utc", "clip_path",
        ])
        writer.writeheader()
        for sample in samples:
            writer.writerow({
                "split": split_name,
                "site": sample.site,
                "station": sample.station,
                "camera": sample.camera,
                "true_name": sample.label_name,
                "y_true": sample.label_id,
                "timestamp_utc": sample.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "clip_path": str(sample.path),
            })


class JPGVideoDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        training: bool,
        augment: bool | None = None,
    ) -> None:
        self.samples = samples
        self.training = training
        self.augment = training if augment is None else augment

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, video: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) < 0.5:
            video = torch.flip(video, dims=(-1,))
        angle = float(torch.empty(()).uniform_(-10.0, 10.0))
        video = TF.rotate(
            video, angle=angle, interpolation=InterpolationMode.BILINEAR, fill=0.0
        )
        operations = [
            (TF.adjust_brightness, float(torch.empty(()).uniform_(0.8, 1.2))),
            (TF.adjust_contrast, float(torch.empty(()).uniform_(0.8, 1.2))),
            (TF.adjust_saturation, float(torch.empty(()).uniform_(0.8, 1.2))),
            (TF.adjust_hue, float(torch.empty(()).uniform_(-0.1, 0.1))),
        ]
        for index in torch.randperm(len(operations)).tolist():
            function, value = operations[index]
            video = function(video, value)
        return video.clamp_(0.0, 1.0)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        frames = []
        for frame_path in clip_frames(sample.path):
            with Image.open(frame_path) as image:
                frame = TF.pil_to_tensor(image.convert("RGB"))
            frame = TF.resize(
                frame, [224, 224], interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            frames.append(frame)
        if len(frames) != 60:
            raise ValueError(f"Expected 60 JPG frames: {sample.path}")
        video = torch.stack(frames).float().div_(255.0)
        if self.augment:
            video = self._augment(video)
        video = (video - IMAGENET_MEAN) / IMAGENET_STD
        return video, sample.label_id, index


def make_loader(
    samples: list[Sample],
    training: bool,
    args: argparse.Namespace,
    seed_offset: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(args.seed + seed_offset)
    return DataLoader(
        JPGVideoDataset(
            samples,
            training=training,
            augment=training and not args.no_augmentation,
        ),
        batch_size=args.batch_size,
        shuffle=training,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )


def load_teacher(args: argparse.Namespace, device: torch.device) -> nn.Module:
    if args.dinov2_repo is not None:
        teacher = torch.hub.load(
            str(args.dinov2_repo.resolve()), "dinov2_vitb14", source="local"
        )
    else:
        teacher = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    teacher = teacher.to(device).eval()
    teacher.requires_grad_(False)
    return teacher


@torch.no_grad()
def teacher_temporal_mean_patches(
    teacher: nn.Module, videos: torch.Tensor, frame_batch: int, amp_enabled: bool
) -> torch.Tensor:
    batch_size, sequence_length = videos.shape[:2]
    frames = videos.flatten(0, 1)
    chunks = []
    for frame_chunk in frames.split(frame_batch):
        with autocast(device_type="cuda", enabled=amp_enabled):
            output = teacher.get_intermediate_layers(
                frame_chunk, n=1, reshape=False, return_class_token=True
            )[0]
        patch_tokens = output[0] if isinstance(output, tuple) else output
        if patch_tokens.ndim != 3 or patch_tokens.shape[1] != 256:
            raise ValueError(
                "DINOv2 must return 256 patch tokens with cls excluded; "
                f"received {tuple(patch_tokens.shape)}"
            )
        chunks.append(patch_tokens)
    patches = torch.cat(chunks, dim=0)
    return patches.view(batch_size, sequence_length, 256, -1).mean(dim=1)


def metrics_from_predictions(y_true: list[int], y_pred: list[int]) -> dict[str, object]:
    labels = [0, 1, 2]
    present = sorted(set(y_true))
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "oa": accuracy_score(y_true, y_pred),
        "mf1_three_class": f1_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
        "mf1_present_classes": f1_score(
            y_true, y_pred, labels=present, average="macro", zero_division=0
        ),
        "recall_spilling": recalls[0],
        "recall_plunging": recalls[1],
        "recall_surging": recalls[2],
        "present_class_ids": present,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    samples: list[Sample],
    device: torch.device,
    amp_enabled: bool,
    collect_rows: bool = True,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    model.eval()
    y_true, y_pred, rows = [], [], []
    for videos, labels, sample_indices in loader:
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(device_type="cuda", enabled=amp_enabled):
            logits, _ = model(videos)
            probabilities = torch.softmax(logits, dim=1) if collect_rows else None
        predictions = (
            probabilities.argmax(dim=1) if probabilities is not None
            else logits.argmax(dim=1)
        )
        label_values = labels.cpu().tolist()
        prediction_values = predictions.cpu().tolist()
        y_true.extend(label_values)
        y_pred.extend(prediction_values)
        if collect_rows:
            for label, prediction, probability, sample_index in zip(
                label_values, prediction_values, probabilities.cpu().tolist(),
                sample_indices.tolist()
            ):
                sample = samples[sample_index]
                rows.append({
                    "site": sample.site,
                    "station": sample.station,
                    "camera": sample.camera,
                    "clip_path": str(sample.path),
                    "true_name": sample.label_name,
                    "y_true": label,
                    "pred_name": ID_TO_CLASS[prediction],
                    "y_pred": prediction,
                    "prob_spilling": probability[0],
                    "prob_plunging": probability[1],
                    "prob_surging": probability[2],
                })
    return metrics_from_predictions(y_true, y_pred), rows


def save_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_one_fold(
    test_site: str,
    all_samples: list[Sample],
    teacher: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, object]:
    train_samples, val_samples, test_samples = make_loso_split(all_samples, test_site)
    fold_dir = args.output_root / f"Test_{test_site}_{SITE_TO_STATION[test_site]}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    save_manifest(fold_dir / "train_manifest.csv", "train", train_samples)
    save_manifest(fold_dir / "val_manifest.csv", "val", val_samples)
    save_manifest(fold_dir / "test_manifest.csv", "test", test_samples)

    train_loader = make_loader(
        train_samples, True, args, seed_offset=11
    )
    val_loader = make_loader(val_samples, False, args, seed_offset=22)
    test_loader = make_loader(test_samples, False, args, seed_offset=33)

    student = BreakTypeNet(num_classes=3, embed_dim=384).to(device)
    adapter = FeatureAdapter(384, 768).to(device)
    criterion = HeterogeneousKnowledgeDistillationLoss(
        alpha=1.0, beta=1.0, gamma=1.0
    )
    optimizer = optim.AdamW(
        list(student.parameters()) + list(adapter.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    amp_enabled = device.type == "cuda"
    scaler = GradScaler("cuda", enabled=amp_enabled)

    config = vars(args).copy()
    config.update({
        "image_root": str(args.image_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "dinov2_repo": str(args.dinov2_repo.resolve()) if args.dinov2_repo else None,
        "teacher_protocol": "online_same_frames_as_student",
        "random_augmentation": not args.no_augmentation,
        "checkpoint_selection_metric": "validation_oa",
        "epoch_test_metrics_role": "record_only_not_used_for_selection",
        "test_site": test_site,
        "station": SITE_TO_STATION[test_site],
        "effective_batch_size": args.batch_size * args.accumulation_steps,
        "loss_weights": {"alpha": 1.0, "beta": 1.0, "gamma": 1.0},
        "split_counts": {
            "train": split_counts(train_samples),
            "val": split_counts(val_samples),
            "test": split_counts(test_samples),
        },
    })
    (fold_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    best_val_oa = -1.0
    best_epoch = 0
    stale_epochs = 0
    history = []
    start_epoch = 1
    last_checkpoint_path = fold_dir / "last_training_checkpoint.pt"
    if args.resume and last_checkpoint_path.is_file():
        checkpoint = torch.load(
            last_checkpoint_path, map_location=device, weights_only=False
        )
        stored_config = checkpoint["config"]
        for key in (
            "batch_size", "accumulation_steps", "epochs", "learning_rate",
            "weight_decay", "patience", "random_augmentation",
            "track_test_each_epoch",
        ):
            if stored_config.get(key) != config.get(key):
                raise ValueError(
                    f"Cannot resume {test_site}: config mismatch for {key}: "
                    f"{stored_config.get(key)!r} != {config.get(key)!r}"
                )
        student.load_state_dict(checkpoint["student_state_dict"])
        adapter.load_state_dict(checkpoint["adapter_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        best_val_oa = float(checkpoint["best_val_oa"])
        best_epoch = int(checkpoint["best_epoch"])
        stale_epochs = int(checkpoint["stale_epochs"])
        history = checkpoint["history"]
        start_epoch = int(checkpoint["epoch"]) + 1
        restore_rng_state(checkpoint["rng_state"], train_loader)
        print(
            f"[{test_site}] resumed after epoch {start_epoch - 1}; "
            f"best epoch={best_epoch}, best val_OA={best_val_oa:.4f}, "
            f"stale={stale_epochs}/{args.patience}"
        )
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        student.train()
        adapter.train()
        running = defaultdict(float)
        correct = 0
        seen = 0

        for batch_index, (videos, labels, _) in enumerate(train_loader, start=1):
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast(device_type="cuda", enabled=amp_enabled):
                logits, student_features = student(videos)
                teacher_features = teacher_temporal_mean_patches(
                    teacher, videos, args.teacher_frame_batch, amp_enabled
                )
                loss, components = criterion(
                    logits, teacher_features, student_features, labels, adapter
                )
                scaled_loss = loss / args.accumulation_steps
            scaler.scale(scaled_loss).backward()

            should_step = (
                batch_index % args.accumulation_steps == 0
                or batch_index == len(train_loader)
            )
            if should_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            batch_size = labels.size(0)
            seen += batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            for name, value in components.items():
                running[name] += float(value.detach()) * batch_size

            if args.progress_interval > 0 and (
                batch_index == 1
                or batch_index % args.progress_interval == 0
                or batch_index == len(train_loader)
            ):
                elapsed = time.time() - epoch_start
                eta = elapsed / batch_index * (len(train_loader) - batch_index)
                print(
                    f"\r[{test_site}] epoch {epoch:03d}/{args.epochs} "
                    f"batch {batch_index:,}/{len(train_loader):,} "
                    f"({batch_index / len(train_loader):6.2%}) "
                    f"loss={running['total_loss'] / seen:.4f} "
                    f"OA={correct / seen:.4f} "
                    f"elapsed={format_duration(elapsed)} ETA={format_duration(eta)}",
                    end="",
                    flush=True,
                )

        if args.progress_interval > 0:
            print()
        scheduler.step()
        val_metrics, _ = evaluate(student, val_loader, val_samples, device, amp_enabled)
        epoch_test_metrics = None
        if args.track_test_each_epoch:
            epoch_test_metrics, _ = evaluate(
                student, test_loader, test_samples, device, amp_enabled,
                collect_rows=False,
            )
        record = {
            "epoch": epoch,
            "learning_rate": scheduler.get_last_lr()[0],
            "train_oa": correct / seen,
            "train_total_loss": running["total_loss"] / seen,
            "train_task_loss": running["task_loss"] / seen,
            "train_feature_loss": running["feature_loss"] / seen,
            "train_relation_loss": running["relation_loss"] / seen,
            "val_oa": val_metrics["oa"],
            "val_mf1_three_class": val_metrics["mf1_three_class"],
            "seconds": time.time() - epoch_start,
        }
        if epoch_test_metrics is not None:
            record.update({
                "test_oa": epoch_test_metrics["oa"],
                "test_mf1_three_class": epoch_test_metrics["mf1_three_class"],
                "test_mf1_present_classes": epoch_test_metrics["mf1_present_classes"],
                "test_recall_spilling": epoch_test_metrics["recall_spilling"],
                "test_recall_plunging": epoch_test_metrics["recall_plunging"],
                "test_recall_surging": epoch_test_metrics["recall_surging"],
                "test_confusion_matrix": json.dumps(
                    epoch_test_metrics["confusion_matrix"], separators=(",", ":")
                ),
            })
        history.append(record)
        save_rows(fold_dir / "training_history.csv", history)
        if args.track_test_each_epoch:
            test_history = [
                {key: value for key, value in item.items()
                 if key == "epoch" or key.startswith("test_")}
                for item in history
            ]
            save_rows(fold_dir / "test_metrics_by_epoch.csv", test_history)
        test_text = (
            f" test_OA={record['test_oa']:.4f} "
            f"test_mF1={record['test_mf1_three_class']:.4f}"
            if epoch_test_metrics is not None else ""
        )
        print(
            f"[{test_site}] epoch {epoch:03d}/{args.epochs} "
            f"loss={record['train_total_loss']:.4f} "
            f"train_OA={record['train_oa']:.4f} val_OA={record['val_oa']:.4f} "
            f"val_mF1={record['val_mf1_three_class']:.4f} "
            f"{test_text} "
            f"time={record['seconds']:.1f}s"
        )

        should_stop = False
        if val_metrics["oa"] > best_val_oa:
            best_val_oa = float(val_metrics["oa"])
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(fold_dir / "best_breaktypenet_checkpoint.pt", {
                "epoch": epoch,
                "best_val_oa": best_val_oa,
                "student_state_dict": student.state_dict(),
                "adapter_state_dict": adapter.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": config,
                "class_to_id": CLASS_TO_ID,
            })
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                should_stop = True

        save_checkpoint(last_checkpoint_path, {
            "epoch": epoch,
            "best_val_oa": best_val_oa,
            "best_epoch": best_epoch,
            "stale_epochs": stale_epochs,
            "history": history,
            "student_state_dict": student.state_dict(),
            "adapter_state_dict": adapter.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "rng_state": rng_state(train_loader),
            "config": config,
            "class_to_id": CLASS_TO_ID,
        })
        if should_stop:
            print(f"[{test_site}] early stopping at epoch {epoch}")
            break

    checkpoint = torch.load(
        fold_dir / "best_breaktypenet_checkpoint.pt", map_location=device, weights_only=False
    )
    student.load_state_dict(checkpoint["student_state_dict"])
    test_metrics, prediction_rows = evaluate(
        student, test_loader, test_samples, device, amp_enabled
    )
    save_rows(fold_dir / "test_predictions.csv", prediction_rows)
    np.save(
        fold_dir / "test_confusion_matrix.npy",
        np.asarray(test_metrics["confusion_matrix"], dtype=np.int64),
    )
    summary = {
        "test_site": test_site,
        "station": SITE_TO_STATION[test_site],
        "best_epoch": best_epoch,
        "best_val_oa": best_val_oa,
        **test_metrics,
    }
    (fold_dir / "test_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def print_split_summary(samples: list[Sample]) -> None:
    print(f"Validated {len(samples):,} JPG clips")
    for test_site in SITE_TO_STATION:
        train, val, test = make_loso_split(samples, test_site)
        print(
            f"{test_site} ({SITE_TO_STATION[test_site]}): "
            f"train={len(train):,}, val={len(val):,}, test={len(test):,}"
        )
        print("  test:", split_counts(test))


def pool_saved_predictions(output_root: Path, sites: list[str]) -> dict[str, object]:
    pooled_rows = []
    for site in sites:
        path = output_root / f"Test_{site}_{SITE_TO_STATION[site]}" / "test_predictions.csv"
        with path.open("r", newline="", encoding="utf-8") as handle:
            pooled_rows.extend(csv.DictReader(handle))
    y_true = [int(row["y_true"]) for row in pooled_rows]
    y_pred = [int(row["y_pred"]) for row in pooled_rows]
    metrics = metrics_from_predictions(y_true, y_pred)
    metrics.update({
        "sites": sites,
        "n_predictions": len(pooled_rows),
        "is_complete_six_fold_loso": set(sites) == set(SITE_TO_STATION),
    })
    save_rows(output_root / "pooled_loso_predictions.csv", pooled_rows)
    np.save(
        output_root / "pooled_loso_confusion_matrix.npy",
        np.asarray(metrics["confusion_matrix"], dtype=np.int64),
    )
    (output_root / "pooled_loso_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if metrics["is_complete_six_fold_loso"] and len(pooled_rows) != 9000:
        raise ValueError(
            f"Complete six-fold LOSO must pool 9,000 predictions, got {len(pooled_rows)}"
        )
    return metrics


def main() -> None:
    args = parse_args()
    if min(args.batch_size, args.accumulation_steps, args.epochs, args.patience) <= 0:
        raise ValueError("Batch size, accumulation, epochs, and patience must be positive")
    set_seed(args.seed)
    samples = scan_samples(args.image_root)
    print_split_summary(samples)
    if args.dry_run:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for DINOv2 LOSO training")

    args.output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    print(f"Using {torch.cuda.get_device_name(device)}")
    teacher = load_teacher(args, device)
    sites = list(SITE_TO_STATION) if args.test_site == "all" else [args.test_site]
    summaries = []
    for fold_index, test_site in enumerate(sites):
        set_seed(args.seed + fold_index)
        completed_metrics = (
            args.output_root / f"Test_{test_site}_{SITE_TO_STATION[test_site]}"
            / "test_metrics.json"
        )
        completed_predictions = completed_metrics.with_name("test_predictions.csv")
        if args.resume and completed_metrics.is_file() and completed_predictions.is_file():
            print(f"[{test_site}] complete; skipping saved fold")
            summaries.append(json.loads(completed_metrics.read_text(encoding="utf-8")))
            continue
        summaries.append(
            train_one_fold(
                test_site, samples, teacher, device, args
            )
        )
        torch.cuda.empty_cache()
    save_rows(args.output_root / "loso_fold_metrics.csv", summaries)
    pooled_metrics = pool_saved_predictions(args.output_root, sites)
    print(json.dumps({"folds": summaries, "pooled": pooled_metrics},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
