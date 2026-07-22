from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ud_cdkd.data.conversion import slices_to_volume, volume_to_slices
from ud_cdkd.data.datasets import build_synthetic_datasets
from ud_cdkd.distillation.direction_controller import (
    DirectionDecision,
    StabilityGuidedDirectionController,
    TransferInstruction,
    select_transfer_instructions,
)
from ud_cdkd.engine.evaluator import evaluate_3d
from ud_cdkd.losses.ducl import DynamicUncertaintyGuidedConsistencyLoss
from ud_cdkd.losses.segmentation import segmentation_loss
from ud_cdkd.losses.supervised_consistency import supervised_cross_dimensional_consistency
from ud_cdkd.models.factory import build_models
from ud_cdkd.utils.checkpoint import load_checkpoint, save_checkpoint
from ud_cdkd.utils.distributed import DistributedContext, barrier


class UDCDKDTrainer:
    """Two-stage trainer matching the terminology and objectives used in the manuscript."""

    def __init__(self, cfg, distributed: DistributedContext) -> None:
        self.cfg = cfg
        self.distributed = distributed
        self.device = distributed.device
        self.output_dir = Path(cfg.experiment.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        labelled_ds, unlabelled_ds, validation_ds = build_synthetic_datasets(
            cfg, write_manifest=distributed.is_main
        )
        self.labelled_sampler = self._sampler(labelled_ds, shuffle=True)
        self.unlabelled_sampler = self._sampler(unlabelled_ds, shuffle=True) if len(unlabelled_ds) else None
        self.stability_sampler = self._sampler(unlabelled_ds, shuffle=False) if len(unlabelled_ds) else None
        self.validation_sampler = self._sampler(validation_ds, shuffle=False)

        loader_kwargs = dict(
            batch_size=int(cfg.training.batch_size),
            num_workers=int(cfg.training.num_workers),
            pin_memory=torch.cuda.is_available(),
            persistent_workers=int(cfg.training.num_workers) > 0,
        )
        self.labelled_loader = DataLoader(
            labelled_ds,
            sampler=self.labelled_sampler,
            shuffle=self.labelled_sampler is None,
            drop_last=True,
            **loader_kwargs,
        )
        self.unlabelled_loader = (
            DataLoader(
                unlabelled_ds,
                sampler=self.unlabelled_sampler,
                shuffle=self.unlabelled_sampler is None,
                drop_last=True,
                **loader_kwargs,
            )
            if len(unlabelled_ds)
            else None
        )
        self.stability_loader = (
            DataLoader(
                unlabelled_ds,
                sampler=self.stability_sampler,
                shuffle=False,
                drop_last=False,
                **loader_kwargs,
            )
            if len(unlabelled_ds)
            else None
        )
        self.validation_loader = DataLoader(
            validation_ds,
            sampler=self.validation_sampler,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )

        self.branch_2d, self.branch_3d = build_models(cfg)
        self.branch_2d.to(self.device)
        self.branch_3d.to(self.device)
        if distributed.enabled:
            self.branch_2d = DistributedDataParallel(
                self.branch_2d,
                device_ids=[distributed.local_rank],
                output_device=distributed.local_rank,
            )
            self.branch_3d = DistributedDataParallel(
                self.branch_3d,
                device_ids=[distributed.local_rank],
                output_device=distributed.local_rank,
            )

        self.optimizer_2d = torch.optim.AdamW(
            [p for p in self.branch_2d.parameters() if p.requires_grad],
            lr=float(cfg.training.learning_rate_2d),
            weight_decay=float(cfg.training.weight_decay),
        )
        self.optimizer_3d = torch.optim.AdamW(
            [p for p in self.branch_3d.parameters() if p.requires_grad],
            lr=float(cfg.training.learning_rate_3d),
            weight_decay=float(cfg.training.weight_decay),
        )

        self.controller = StabilityGuidedDirectionController(
            prediction_momentum=float(cfg.distillation.prediction_ema_momentum),
            statistics_momentum=float(cfg.distillation.statistics_momentum),
        )
        self.ducl = DynamicUncertaintyGuidedConsistencyLoss(
            eta=float(cfg.ducl.eta),
            tau=float(cfg.ducl.tau),
            beta0=float(cfg.ducl.beta0),
            delta0=float(cfg.ducl.delta0),
            uncertainty_weighting=bool(cfg.ducl.uncertainty_weighting),
            entropy_regularisation=bool(cfg.ducl.entropy_regularisation),
            dynamic_scheduling=bool(cfg.ducl.dynamic_scheduling),
        )

        amp_enabled = bool(cfg.training.mixed_precision) and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        self.amp_enabled = amp_enabled
        self.writer = SummaryWriter(self.output_dir / "tensorboard") if distributed.is_main else None
        self.log_path = self.output_dir / "training_log.jsonl"
        self.best_dice = -1.0
        self.start_stage = "warmup"
        self.start_epoch = 1

    def _sampler(self, dataset, shuffle: bool):
        if not self.distributed.enabled:
            return None
        return DistributedSampler(dataset, shuffle=shuffle, drop_last=False)

    @staticmethod
    def _unwrap(model: nn.Module) -> nn.Module:
        return model.module if isinstance(model, DistributedDataParallel) else model

    def _forward_2d_volume(self, volume: torch.Tensor, require_grad: bool = True) -> torch.Tensor:
        sections, layout = volume_to_slices(volume, slice_axis=int(self.cfg.data.slice_axis))
        chunk_size = int(self.cfg.training.slice_chunk_size)
        context = torch.enable_grad() if require_grad else torch.no_grad()
        outputs: list[torch.Tensor] = []
        with context:
            for chunk in torch.split(sections, chunk_size, dim=0):
                outputs.append(self.branch_2d(chunk))
        return slices_to_volume(torch.cat(outputs, dim=0), layout)

    def _supervised_objective(self, batch: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image = batch["image"].to(self.device, non_blocking=True)
        target = batch["mask"].to(self.device, non_blocking=True)
        logits_2d = self._forward_2d_volume(image, require_grad=True)
        logits_3d = self.branch_3d(image)
        loss_2d, components_2d = segmentation_loss(logits_2d, target)
        loss_3d, components_3d = segmentation_loss(logits_3d, target)
        consistency = supervised_cross_dimensional_consistency(
            logits_2d,
            logits_3d,
            confidence_2d=float(self.cfg.distillation.confidence_threshold_2d),
            confidence_3d=float(self.cfg.distillation.confidence_threshold_3d),
        )
        total = loss_2d + loss_3d + float(self.cfg.training.supervised_consistency_weight) * consistency
        return total, {
            "segmentation_2d": loss_2d,
            "segmentation_3d": loss_3d,
            "cross_dimensional_consistency": consistency,
            "dice_2d": components_2d["dice"],
            "dice_3d": components_3d["dice"],
        }

    def _unlabelled_logits(
        self,
        image: torch.Tensor,
        instructions: list[TransferInstruction],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        update_2d = any(item.updated_branch == "2d" for item in instructions)
        update_3d = any(item.updated_branch == "3d" for item in instructions)
        logits_2d = self._forward_2d_volume(image, require_grad=update_2d)
        if update_3d:
            logits_3d = self.branch_3d(image)
        else:
            with torch.no_grad():
                logits_3d = self.branch_3d(image)
        return logits_2d, logits_3d

    def _distillation_objective(
        self,
        batch: dict,
        instructions: list[TransferInstruction],
        epoch: int,
    ) -> tuple[torch.Tensor, dict[str, float | torch.Tensor]]:
        if not instructions:
            zero = torch.zeros((), device=self.device)
            return zero, {"ducl": zero, "ramp_weight": 0.0}

        image = batch["image"].to(self.device, non_blocking=True)
        logits_2d, logits_3d = self._unlabelled_logits(image, instructions)
        logits = {"2d": logits_2d, "3d": logits_3d}
        total = torch.zeros((), device=self.device)
        component_log: dict[str, float | torch.Tensor] = {}
        for instruction in instructions:
            components = self.ducl(
                updated_logits=logits[instruction.updated_branch],
                reference_logits=logits[instruction.reference_branch],
                epoch=epoch,
                total_epochs=int(self.cfg.training.distillation_epochs),
                direction_scale=instruction.scale,
            )
            key = f"{instruction.reference_branch}_to_{instruction.updated_branch}"
            total = total + components.total
            component_log[f"ducl_{key}"] = components.total
            component_log[f"pseudo_label_{key}"] = components.pseudo_label
            component_log[f"entropy_{key}"] = components.entropy_regularisation
            component_log[f"reliability_{key}"] = components.mean_reliability
            component_log["ramp_weight"] = components.ramp_weight
        component_log["ducl"] = total
        return total, component_log

    @torch.no_grad()
    def _estimate_3d_stability(self) -> DirectionDecision:
        if self.stability_loader is None:
            raise RuntimeError("Stability-guided distillation requires unlabelled training volumes.")
        self.branch_3d.eval()
        self.controller.begin_epoch()
        iterator = tqdm(
            self.stability_loader,
            desc="3-D stability",
            disable=not self.distributed.is_main,
            leave=False,
        )
        for batch in iterator:
            image = batch["image"].to(self.device, non_blocking=True)
            probability = torch.softmax(self.branch_3d(image), dim=1)[:, 1]
            self.controller.observe(batch["sample_id"], probability)
        return self.controller.end_epoch(self.distributed)

    def _step(self, loss: torch.Tensor) -> None:
        self.optimizer_2d.zero_grad(set_to_none=True)
        self.optimizer_3d.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        if float(self.cfg.training.gradient_clip_norm) > 0:
            self.scaler.unscale_(self.optimizer_2d)
            self.scaler.unscale_(self.optimizer_3d)
            torch.nn.utils.clip_grad_norm_(self.branch_2d.parameters(), float(self.cfg.training.gradient_clip_norm))
            torch.nn.utils.clip_grad_norm_(self.branch_3d.parameters(), float(self.cfg.training.gradient_clip_norm))
        self.scaler.step(self.optimizer_2d)
        self.scaler.step(self.optimizer_3d)
        self.scaler.update()

    def _next_labelled(self, iterator: Iterator) -> tuple[dict, Iterator]:
        try:
            return next(iterator), iterator
        except StopIteration:
            iterator = iter(self.labelled_loader)
            return next(iterator), iterator

    def _set_epoch(self, epoch: int) -> None:
        for sampler in (self.labelled_sampler, self.unlabelled_sampler):
            if sampler is not None:
                sampler.set_epoch(epoch)

    def _record(self, stage: str, epoch: int, values: dict) -> None:
        serialised = {"stage": stage, "epoch": epoch}
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                serialised[key] = float(value.detach().mean().cpu().item())
            else:
                serialised[key] = float(value) if isinstance(value, (int, float)) else value
        if self.distributed.is_main:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(serialised) + "\n")
            if self.writer is not None:
                global_step = epoch if stage == "warmup" else int(self.cfg.training.warmup_epochs) + epoch
                for key, value in serialised.items():
                    if isinstance(value, float):
                        self.writer.add_scalar(f"{stage}/{key}", value, global_step)

    def _save(self, stage: str, epoch: int, is_best: bool = False) -> None:
        if not self.distributed.is_main:
            return
        state = {
            "stage": stage,
            "epoch": epoch,
            "model_2d": self._unwrap(self.branch_2d).state_dict(),
            "model_3d": self._unwrap(self.branch_3d).state_dict(),
            "optimizer_2d": self.optimizer_2d.state_dict(),
            "optimizer_3d": self.optimizer_3d.state_dict(),
            "scaler": self.scaler.state_dict(),
            "controller": self.controller.state_dict(
                include_prediction_ema=bool(self.cfg.distillation.save_prediction_ema)
            ),
            "best_dice": self.best_dice,
            "config_path": self.cfg.config_path,
        }
        save_checkpoint(self.output_dir / "last_checkpoint.pt", **state)
        if is_best:
            save_checkpoint(self.output_dir / "best_3d_checkpoint.pt", **state)

    def resume(self, checkpoint_path: str | Path) -> None:
        state = load_checkpoint(checkpoint_path, map_location=self.device)
        self._unwrap(self.branch_2d).load_state_dict(state["model_2d"])
        self._unwrap(self.branch_3d).load_state_dict(state["model_3d"])
        self.optimizer_2d.load_state_dict(state["optimizer_2d"])
        self.optimizer_3d.load_state_dict(state["optimizer_3d"])
        self.scaler.load_state_dict(state.get("scaler", {}))
        self.controller.load_state_dict(state.get("controller", {}))
        self.best_dice = float(state.get("best_dice", -1.0))
        self.start_stage = str(state.get("stage", "warmup"))
        self.start_epoch = int(state.get("epoch", 0)) + 1

    def train(self) -> None:
        if self.start_stage == "warmup":
            self._train_warmup(self.start_epoch)
            distillation_start = 1
        else:
            distillation_start = self.start_epoch
        self._train_distillation(distillation_start)
        if self.writer is not None:
            self.writer.close()

    def _train_warmup(self, start_epoch: int = 1) -> None:
        total_epochs = int(self.cfg.training.warmup_epochs)
        for epoch in range(start_epoch, total_epochs + 1):
            self._set_epoch(epoch)
            self.branch_2d.train()
            self.branch_3d.train()
            epoch_loss = 0.0
            iterator = tqdm(self.labelled_loader, desc=f"Warm-up {epoch}/{total_epochs}", disable=not self.distributed.is_main)
            for batch in iterator:
                with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                    loss, components = self._supervised_objective(batch)
                self._step(loss)
                epoch_loss += float(loss.detach().item())
                iterator.set_postfix(loss=f"{loss.detach().item():.4f}")

            values = {"loss": epoch_loss / max(len(self.labelled_loader), 1), **components}
            self._record("warmup", epoch, values)
            if epoch % int(self.cfg.training.checkpoint_interval) == 0 or epoch == total_epochs:
                self._save("warmup", epoch)
            barrier(self.distributed)

    def _train_distillation(self, start_epoch: int = 1) -> None:
        total_epochs = int(self.cfg.training.distillation_epochs)
        strategy = str(self.cfg.distillation.strategy)
        if strategy != "no_distillation" and self.unlabelled_loader is None:
            raise RuntimeError(f"Strategy '{strategy}' requires unlabelled data, but none are available.")

        for epoch in range(start_epoch, total_epochs + 1):
            self._set_epoch(int(self.cfg.training.warmup_epochs) + epoch)
            decision = self._estimate_3d_stability() if strategy == "stability_guided" else None
            instructions = select_transfer_instructions(strategy, epoch, decision)
            self.branch_2d.train()
            self.branch_3d.train()
            labelled_iterator = iter(self.labelled_loader)
            running_total = 0.0

            active_loader = self.unlabelled_loader if instructions else self.labelled_loader
            assert active_loader is not None
            iterator = tqdm(active_loader, desc=f"Distillation {epoch}/{total_epochs}", disable=not self.distributed.is_main)
            last_components: dict = {}
            for batch in iterator:
                if instructions:
                    labelled_batch, labelled_iterator = self._next_labelled(labelled_iterator)
                    unlabelled_batch = batch
                else:
                    labelled_batch = batch
                    unlabelled_batch = None

                with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                    supervised, supervised_components = self._supervised_objective(labelled_batch)
                    if unlabelled_batch is not None:
                        distillation, distillation_components = self._distillation_objective(
                            unlabelled_batch,
                            instructions,
                            epoch,
                        )
                    else:
                        distillation = torch.zeros((), device=self.device)
                        distillation_components = {"ducl": distillation, "ramp_weight": 0.0}
                    loss = supervised + distillation
                self._step(loss)
                running_total += float(loss.detach().item())
                last_components = {**supervised_components, **distillation_components}
                iterator.set_postfix(loss=f"{loss.detach().item():.4f}")

            values: dict = {"loss": running_total / max(len(active_loader), 1), **last_components}
            if decision is not None:
                values.update(
                    {
                        "stability_3d": decision.stability,
                        "normalised_stability_improvement": decision.normalised_improvement,
                        "reverse_activation": decision.reverse_activation,
                        "reference_branch": decision.reference_branch,
                        "updated_branch": decision.updated_branch,
                    }
                )
            self._record("distillation", epoch, values)

            is_best = False
            if epoch % int(self.cfg.training.validation_interval) == 0 or epoch == total_epochs:
                metrics = evaluate_3d(self.branch_3d, self.validation_loader, self.device, self.distributed)
                self._record("validation", epoch, metrics)
                if metrics["dice"] > self.best_dice:
                    self.best_dice = metrics["dice"]
                    is_best = True
            if epoch % int(self.cfg.training.checkpoint_interval) == 0 or epoch == total_epochs or is_best:
                self._save("distillation", epoch, is_best=is_best)
            barrier(self.distributed)
