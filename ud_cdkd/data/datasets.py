from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class FaultSample:
    sample_id: str
    seismic_path: Path
    fault_path: Path | None


def _find_samples(root: Path, split: str, require_labels: bool) -> list[FaultSample]:
    split_root = root / split
    if not split_root.exists():
        raise FileNotFoundError(f"Data split directory does not exist: {split_root}")

    seismic_files = sorted(split_root.glob("*/seis.npy"))
    if not seismic_files:
        seismic_files = sorted(split_root.glob("**/seis.npy"))
    if not seismic_files:
        raise FileNotFoundError(f"No 'seis.npy' files found below {split_root}")

    samples: list[FaultSample] = []
    for seismic_path in seismic_files:
        fault_path = seismic_path.with_name("fault.npy")
        if require_labels and not fault_path.exists():
            raise FileNotFoundError(f"Missing fault label for {seismic_path}: {fault_path}")
        samples.append(
            FaultSample(
                sample_id=seismic_path.parent.name,
                seismic_path=seismic_path,
                fault_path=fault_path if fault_path.exists() else None,
            )
        )
    return samples


def _reorder_to_hwd(array: np.ndarray, axis_order: str) -> np.ndarray:
    order = axis_order.upper()
    if order == "HWD":
        return array
    if order == "DHW":
        return np.transpose(array, (1, 2, 0))
    if order == "WHD":
        return np.transpose(array, (1, 0, 2))
    raise ValueError(f"Unsupported axis_order '{axis_order}'. Use HWD, DHW or WHD.")


def _zscore(volume: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    volume = volume.astype(np.float32, copy=False)
    return (volume - volume.mean()) / (volume.std() + eps)


def _random_crop(image: np.ndarray, mask: np.ndarray | None, patch_size: Sequence[int]) -> tuple[np.ndarray, np.ndarray | None]:
    patch = tuple(int(v) for v in patch_size)
    if image.shape == patch:
        return image, mask
    if any(size < crop for size, crop in zip(image.shape, patch)):
        pads = [(0, max(0, crop - size)) for size, crop in zip(image.shape, patch)]
        image = np.pad(image, pads, mode="edge")
        if mask is not None:
            mask = np.pad(mask, pads, mode="constant")

    starts = [random.randint(0, image.shape[i] - patch[i]) for i in range(3)]
    slices = tuple(slice(start, start + patch[i]) for i, start in enumerate(starts))
    return image[slices], None if mask is None else mask[slices]


def _augment(image: np.ndarray, mask: np.ndarray | None, flip_probability: float, rotation_probability: float, noise_std: float) -> tuple[np.ndarray, np.ndarray | None]:
    for axis in range(3):
        if random.random() < flip_probability:
            image = np.flip(image, axis=axis)
            if mask is not None:
                mask = np.flip(mask, axis=axis)

    if random.random() < rotation_probability:
        axes = random.choice(((0, 1), (0, 2), (1, 2)))
        k = random.randint(1, 3)
        image = np.rot90(image, k=k, axes=axes)
        if mask is not None:
            mask = np.rot90(mask, k=k, axes=axes)

    if noise_std > 0:
        image = image + np.random.normal(0.0, noise_std, size=image.shape).astype(np.float32)
    return np.ascontiguousarray(image), None if mask is None else np.ascontiguousarray(mask)


class SyntheticFaultDataset(Dataset):
    """Synthetic 3-D seismic fault volumes stored as ``seis.npy``/``fault.npy`` pairs."""

    def __init__(
        self,
        samples: Sequence[FaultSample],
        labelled: bool,
        patch_size: Sequence[int] = (128, 128, 128),
        axis_order: str = "DHW",
        augment: bool = False,
        flip_probability: float = 0.5,
        rotation_probability: float = 0.5,
        noise_std: float = 0.0,
    ) -> None:
        if labelled and any(sample.fault_path is None for sample in samples):
            raise ValueError("A labelled dataset contains samples without fault labels.")
        self.samples = list(samples)
        self.labelled = labelled
        self.patch_size = tuple(int(v) for v in patch_size)
        self.axis_order = axis_order
        self.augment = augment
        self.flip_probability = float(flip_probability)
        self.rotation_probability = float(rotation_probability)
        self.noise_std = float(noise_std)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        image = _reorder_to_hwd(np.load(sample.seismic_path), self.axis_order)
        mask = None
        if self.labelled:
            assert sample.fault_path is not None
            mask = _reorder_to_hwd(np.load(sample.fault_path), self.axis_order)
            mask = (mask > 0).astype(np.int64)

        image, mask = _random_crop(image, mask, self.patch_size)
        image = _zscore(image)
        if self.augment:
            image, mask = _augment(
                image,
                mask,
                self.flip_probability,
                self.rotation_probability,
                self.noise_std,
            )

        output: dict[str, torch.Tensor | str] = {
            "image": torch.from_numpy(np.ascontiguousarray(image[None])).float(),
            "sample_id": sample.sample_id,
        }
        if mask is not None:
            output["mask"] = torch.from_numpy(np.ascontiguousarray(mask)).long()
        return output


def _write_split_manifest(path: Path, labelled: Iterable[FaultSample], unlabelled: Iterable[FaultSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "labelled": [sample.sample_id for sample in labelled],
        "unlabelled": [sample.sample_id for sample in unlabelled],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_synthetic_datasets(cfg, write_manifest: bool = True) -> tuple[SyntheticFaultDataset, SyntheticFaultDataset, SyntheticFaultDataset]:
    root = Path(cfg.data.root).expanduser()
    training = _find_samples(root, cfg.data.train_split, require_labels=True)
    validation = _find_samples(root, cfg.data.val_split, require_labels=True)

    rng = random.Random(int(cfg.experiment.seed))
    shuffled = training.copy()
    rng.shuffle(shuffled)
    num_labelled = max(1, int(math.floor(len(shuffled) * float(cfg.data.label_ratio))))
    labelled_samples = sorted(shuffled[:num_labelled], key=lambda sample: sample.sample_id)
    unlabelled_samples = sorted(shuffled[num_labelled:], key=lambda sample: sample.sample_id)

    if write_manifest:
        output_dir = Path(cfg.experiment.output_dir)
        _write_split_manifest(output_dir / "split_manifest.json", labelled_samples, unlabelled_samples)

    common = dict(
        patch_size=cfg.data.patch_size,
        axis_order=cfg.data.axis_order,
        flip_probability=cfg.data.augmentation.flip_probability,
        rotation_probability=cfg.data.augmentation.rotation_probability,
        noise_std=cfg.data.augmentation.noise_std,
    )
    labelled_ds = SyntheticFaultDataset(labelled_samples, labelled=True, augment=True, **common)
    unlabelled_ds = SyntheticFaultDataset(unlabelled_samples, labelled=False, augment=True, **common)
    validation_ds = SyntheticFaultDataset(
        validation,
        labelled=True,
        patch_size=cfg.data.patch_size,
        axis_order=cfg.data.axis_order,
        augment=False,
    )
    return labelled_ds, unlabelled_ds, validation_ds


def load_volume(path: str | Path, axis_order: str = "HWD") -> torch.Tensor:
    """Load and standardise a single ``.npy`` volume as ``(1,1,H,W,D)``."""
    array = np.load(Path(path))
    if array.ndim != 3:
        raise ValueError(f"Expected a 3-D NumPy array, got shape {array.shape}")
    array = _zscore(_reorder_to_hwd(array, axis_order))
    return torch.from_numpy(np.ascontiguousarray(array[None, None])).float()
