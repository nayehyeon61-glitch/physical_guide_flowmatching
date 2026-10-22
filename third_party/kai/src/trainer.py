# src/trainer.py
"""
Copyright (c) 2025 MetaEarth-Lab
Licensed under the BSD 3-Clause License. Redistribution and use in source and binary forms, with or without modification, are permitted. See the LICENSE file in the project root for the full license text.
"""
import os
import time
import logging
from typing import Dict, Any, Tuple
from ruamel.yaml.scalarfloat import ScalarFloat
import numpy as np
import torch
import torch.amp as amp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
import wandb

from .utils import (
    DataConfig,
    build_data_loader,
    LpLoss,
    compute_daily_climatology
)
from .models import KAI


def set_seed(seed: int) -> None:
    """
    Seed Python, NumPy, and Torch RNGs for reproducible experiments.

    Args:
        seed (int): Deterministic seed applied to all supported RNG backends.

    Returns:
        None: RNG state is updated globally without returning a value.

    Raises:
        RuntimeError: Propagated if the underlying frameworks fail to set the seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np

    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer:
    """
    Coordinate data loading, model setup, and the end-to-end training workflow.

    Attributes:
        params (Any): Experiment configuration namespace supplying hyperparameters.
        world_rank (int): Global rank used for distributed coordination.
        local_rank (int): Device-local rank used when selecting GPUs.
        device (torch.device): Target device for model weights and tensors.
        max_gradient_norm (float): Clipping threshold applied to gradients.
        iters (int): Global optimization step counter.
        start_epoch (int): Epoch index to resume from when loading checkpoints.
        epoch (int): Current epoch number.
        train_data_loader (DataLoader): Loader that yields training batches.
        train_dataset (Dataset): Underlying training dataset instance.
        train_sampler (Optional[DistributedSampler]): Sampler used for distributed training.
        valid_data_loader (DataLoader): Loader that yields validation batches.
        valid_dataset (Dataset): Underlying validation dataset instance.
        model (torch.nn.Module): Forecasting network being optimized.
        optimizer (torch.optim.Optimizer): Optimizer instance for model parameters.
        gscaler (Optional[amp.GradScaler]): Automatic mixed-precision gradient scaler.
        loss_obj: Primary loss function used for supervision.
        scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): Learning-rate scheduler.
    """

    def __init__(self, params, world_rank: int = 0, local_rank: int = 0):
        """
        Initialize the trainer state, build data loaders, instantiate the model, and
        configure optimization utilities.

        Args:
            params (YParams): Structured configuration with model and data settings.
            world_rank (int): Distributed rank for the current process.
            local_rank (int): Device index within the current node.

        Returns:
            None: The constructor mutates internal attributes only.

        Raises:
            RuntimeError: If dataset discovery, model creation, or checkpoint loading fails.
        """
        self.params = params
        self.world_rank = world_rank
        self.local_rank = local_rank

        self.device = torch.device(
            f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        )

        # The autoregressive rollout used by evaluation/inference feeds the model's
        # own output back in as its next input, which requires
        # in_channels == out_channels. n_history > 0 stacks past frames into the
        # input (in_channels = (n_history + 1) * C), breaking that invariant, so it
        # is not supported.
        if int(getattr(params, "n_history", 0)) != 0:
            raise NotImplementedError(
                f"n_history={params.n_history} is not supported: the autoregressive "
                "rollout requires in_channels == out_channels. Set n_history: 0."
            )

        self.max_gradient_norm = 1.0
        self.iters = 0
        self.start_epoch = 0
        self.epoch = 0
        self.early_stopping_patience = int(
            getattr(params, "early_stopping_patience", 10)
        )
        params["early_stopping_patience"] = self.early_stopping_patience

        # -----------------------------
        # WandB
        # -----------------------------
        if params.log_to_wandb:
            wandb_kwargs = {
                "config": params.params,
                "name": params.name,
                "group": params.group,
                "project": params.project,
            }
            entity = getattr(params, "entity", None)
            if entity and str(entity).lower() not in {"none", "na"}:
                wandb_kwargs["entity"] = entity
            tags = getattr(params, "wandb_tags", None) or getattr(params, "tags", None)
            if tags:
                wandb_kwargs["tags"] = [str(tag) for tag in tags]
            wandb.init(**wandb_kwargs)

        # -----------------------------
        # DataConfig + loaders
        # -----------------------------
        data_cfg = DataConfig(
            dt=params.dt,
            n_history=params.n_history,
            in_channels=params.in_channels,
            out_channels=params.out_channels,
            normalize=(params.normalization == "zscore"),
            normalization=params.normalization,
            global_means_path=params.global_means_path,
            global_stds_path=params.global_stds_path,
            add_grid=params.add_grid,
            gridtype=params.gridtype,
            N_grid_channels=params.N_grid_channels,
            batch_size=params.batch_size,
            num_workers=params.num_data_workers,
        )

        checkpoint_dir = os.path.dirname(params.checkpoint_path)
        self.climatology_path = os.path.join(checkpoint_dir, "daily_climatology.nc")

        # Only rank 0 computes/writes the climatology; writing it from every rank
        # would race on the same output file (temp-file rename). Other ranks wait
        # at the barrier and reuse the file rank 0 produced.
        if world_rank == 0:
            compute_daily_climatology(
                config=data_cfg,
                data_dir=params.train_data_path,
                save_dir=checkpoint_dir,
                filename="daily_climatology.nc",
                channels_ids=params.out_channels,
            )
        if dist.is_initialized():
            dist.barrier()

        self.params.climatology_path = self.climatology_path
        logging.info(f"Climatology available at: {self.climatology_path}")
        logging.info("rank %d: initializing data loaders", world_rank)
        distributed = dist.is_initialized()

        (
            self.train_data_loader,
            self.train_dataset,
            self.train_sampler,
        ) = build_data_loader(
            config=data_cfg,
            data_dir=params.train_data_path,
            distributed=distributed,
            train=True,
        )

        (
            self.valid_data_loader,
            self.valid_dataset,
            _,
        ) = build_data_loader(
            config=data_cfg,
            data_dir=params.valid_data_path,
            distributed=distributed,
            train=False,
        )

        logging.info("rank %d: data loaders ready", world_rank)

        # Save some useful attributes back into params
        params.img_shape_x = self.train_dataset.img_shape_x
        params.img_shape_y = self.train_dataset.img_shape_y

        # -----------------------------
        # Model
        # -----------------------------
        # Infer in/out channels from a sample batch
        sample_inp, sample_tar = next(iter(self.train_data_loader))
        in_chans = sample_inp.shape[1]
        out_chans = sample_tar.shape[1]

        params.N_in_channels = in_chans
        params.N_out_channels = out_chans
        if params.nettype == "kai":
            self.model = KAI(in_chans=in_chans, out_chans=out_chans).to(self.device)
        else:
            raise NotImplementedError(f"nettype='{params.nettype}' not supported yet.")

        if params.enable_nhwc:
            self.model = self.model.to(memory_format=torch.channels_last)

        if params.log_to_wandb:
            wandb.watch(self.model)

        # -----------------------------
        # Optimizer / scaler / DDP
        # -----------------------------
        self.optimizer = AdamW(self.model.parameters(), lr=params.lr)

        self.gscaler = None
        if params.enable_amp:
            self.gscaler = amp.GradScaler("cuda")

        if dist.is_initialized():
            self.model = DDP(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
            )

        # LpLoss computes the relative L2-norm error, which is resolution-invariant.
        self.loss_obj = LpLoss(d=2, p=2, size_average=True, reduction=True)

        # -----------------------------
        # Resume from checkpoint
        # -----------------------------
        # Resume from the latest checkpoint so training continues where it left off.
        if params.resuming and os.path.isfile(params.checkpoint_path):
            logging.info("rank %d: resuming from latest checkpoint %s", world_rank, params.checkpoint_path)
            self.restore_checkpoint(params.checkpoint_path)

        self.epoch = self.start_epoch

        # -----------------------------
        # LR scheduler
        # -----------------------------
        if params.scheduler == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, factor=0.2, patience=5, mode="min"
            )
        elif params.scheduler == "CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=params.max_epochs,
                last_epoch=self.start_epoch - 1,
            )
        else:
            self.scheduler = None

        if params.log_to_screen:
            logging.info(
                "Number of trainable model parameters: %d", self.count_parameters()
            )

        params["n_parameters"] = self.count_parameters()
        if params.log_to_wandb:
            metadata = {"metadata/params": params.n_parameters}
            wandb.config.update(metadata, allow_val_change=True)
            wandb.define_metric("epoch")
            wandb.define_metric("epoch/*", step_metric="epoch")
            wandb.define_metric("train/*", step_metric="epoch")
            wandb.define_metric("valid/*", step_metric="epoch")
            wandb.log(metadata, step=0)

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        """
        Count the number of trainable parameters in the underlying model.

        Args:
            None: The method inspects internal model parameters directly.

        Returns:
            int: Total number of parameters requiring gradients.

        Raises:
            AttributeError: If the model has not been instantiated yet.
        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    def train(self) -> None:
        """
        Execute the multi-epoch training loop with validation, logging, and checkpointing.

        Args:
            None: Uses configuration stored in `self.params`.

        Returns:
            None: Training progress is communicated via logging side effects.

        Raises:
            RuntimeError: If optimization, logging, or checkpoint I/O fails.
        """
        if self.params.log_to_screen:
            logging.info("Starting training loop...")

        best_valid_loss = float("inf")
        epochs_without_improvement = 0
        train_hist = []   # per-epoch mean training loss, for end-of-run loss curves
        valid_hist = []   # per-epoch validation loss
        for epoch in range(self.start_epoch, self.params.max_epochs):
            if dist.is_initialized() and self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(self.device)
            start = time.time()
            tr_time, data_time, train_logs = self.train_one_epoch()
            valid_time, valid_logs = self.validate_one_epoch()
            epoch_time = time.time() - start

            # Track loss history for the end-of-training loss-curve plots
            train_hist.append(float(train_logs["loss"]))
            valid_hist.append(float(valid_logs["valid_loss"]))

            # Scheduler
            if self.params.scheduler == "ReduceLROnPlateau":
                self.scheduler.step(valid_logs["valid_loss"])
            elif self.params.scheduler == "CosineAnnealingLR":
                self.scheduler.step()

            # Early-stopping bookkeeping on validation loss
            current_valid_loss = valid_logs["valid_loss"]
            improved = current_valid_loss < best_valid_loss
            should_save_best = current_valid_loss <= best_valid_loss
            if improved:
                best_valid_loss = current_valid_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Log training loss, validation loss, timing, and GPU usage to wandb
            lr = self.optimizer.param_groups[0]["lr"]
            if self.params.log_to_wandb:
                peak_memory_gb = (
                    torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
                    if torch.cuda.is_available()
                    else 0.0
                )
                wandb.log(
                    {
                        "epoch": self.epoch,
                        "lr": lr,
                        "epoch/lr": lr,
                        "epoch_time_sec": epoch_time,
                        "epoch/time_sec": epoch_time,
                        "train_time_sec": tr_time,
                        "valid_time_sec": valid_time,
                        "data_time_sec": data_time,
                        "peak_memory_gb": peak_memory_gb,
                        "train_loss": train_logs["loss"],
                        "valid_loss": valid_logs["valid_loss"],
                        "train/loss": train_logs["loss"],
                        "valid/loss": valid_logs["valid_loss"],
                        "valid/l1": valid_logs["valid_l1"],
                        "epoch/train_loss": train_logs["loss"],
                        "epoch/valid_loss": valid_logs["valid_loss"],
                        "epoch/valid_l1": valid_logs["valid_l1"],
                        "early_stopping/best_valid_loss": best_valid_loss,
                        "early_stopping/epochs_without_improvement": epochs_without_improvement,
                        "early_stopping/patience": self.early_stopping_patience,
                    },
                    step=self.epoch,
                )

            # Checkpointing (only on rank 0)
            if self.world_rank == 0 and self.params.save_checkpoint:
                self.save_checkpoint(self.params.checkpoint_path)
                if should_save_best:
                    self.save_checkpoint(self.params.best_checkpoint_path)

            if self.params.log_to_screen and self.world_rank == 0:
                logging.info(
                    "Epoch %d finished in %.2f sec | Train loss: %.4f | Valid loss: %.4f | Best valid loss: %.4f | Early stop wait: %d/%d",
                    epoch + 1,
                    epoch_time,
                    train_logs["loss"],
                    valid_logs["valid_loss"],
                    best_valid_loss,
                    epochs_without_improvement,
                    self.early_stopping_patience,
                )

            # Early stopping — break when validation loss stalls for `patience` epochs
            stop_training = (
                self.early_stopping_patience > 0
                and epochs_without_improvement >= self.early_stopping_patience
            )
            if dist.is_initialized():
                stop_tensor = torch.tensor(
                    int(stop_training if self.world_rank == 0 else 0),
                    device=self.device,
                )
                dist.broadcast(stop_tensor, src=0)
                stop_training = bool(stop_tensor.item())

            if stop_training:
                if self.params.log_to_screen and self.world_rank == 0:
                    logging.info(
                        "Early stopping triggered at epoch %d after %d epochs without validation improvement. Best valid loss: %.4f",
                        epoch + 1,
                        epochs_without_improvement,
                        best_valid_loss,
                    )
                if self.params.log_to_wandb and self.world_rank == 0:
                    wandb.log({"early_stopping/stopped": 1}, step=self.epoch)
                break

        # -----------------------------
        # End-of-training loss curves
        # -----------------------------
        # Saved to disk (PNG + CSV) and logged to W&B as a native line plot so
        # train/valid loss can be compared across models.
        if self.world_rank == 0 and train_hist:
            try:
                from .eval_logging import log_loss_curves

                log_loss_curves(
                    train_hist=train_hist,
                    valid_hist=valid_hist,
                    output_dir=self.params.experiment_dir,
                    logger=logging,
                    log_to_wandb=bool(self.params.log_to_wandb),
                )
            except Exception as e:  # never let plotting abort a finished run
                logging.warning(f"Failed to write loss curves: {e}")

    # ------------------------------------------------------------------
    def train_one_epoch(self) -> Tuple[float, float, Dict[str, float]]:
        """
        Iterate over the training DataLoader once while tracking timings and loss.

        Args:
            None: Consumes batches from `self.train_data_loader`.

        Returns:
            Tuple[float, float, Dict[str, float]]: Training time, data loading time,
                and a dictionary with aggregated training metrics.

        Raises:
            RuntimeError: Propagated when forward, backward, or optimizer steps fail.
        """
        self.epoch += 1
        self.model.train()

        tr_time = 0.0
        data_time = 0.0
        last_loss = None
        running_loss = 0.0
        n_batches = 0

        for i, data in enumerate(self.train_data_loader, 0):
            self.iters += 1
            data_start = time.time()
            inp, tar = map(lambda x: x.to(self.device, dtype=torch.float32), data)

            if self.params.enable_nhwc:
                inp = inp.to(memory_format=torch.channels_last)
                tar = tar.to(memory_format=torch.channels_last)

            data_time += time.time() - data_start
            tr_start = time.time()

            self.optimizer.zero_grad(set_to_none=True)

            if self.params.enable_amp and self.gscaler is not None:
                with amp.autocast(device_type='cuda'):
                    gen = self.model(inp)
                    loss = self.loss_obj(gen, tar)

                self.gscaler.scale(loss).backward()
                self.gscaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), self.max_gradient_norm)
                self.gscaler.step(self.optimizer)
                self.gscaler.update()
            else:
                gen = self.model(inp)
                loss = self.loss_obj(gen, tar)
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.max_gradient_norm)
                self.optimizer.step()

            tr_time += time.time() - tr_start
            last_loss = loss.item()
            running_loss += last_loss
            n_batches += 1

            if dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0
            print(f"[rank {rank}] Epoch {self.epoch} - Batch {i} - Loss: {last_loss:.6f}")

        # Mean training loss over the epoch.
        logs = {"loss": running_loss / n_batches if n_batches > 0 else float("nan")}

        # Average logs across ranks
        if dist.is_initialized():
            for key in sorted(logs.keys()):
                t = torch.tensor(logs[key], device=self.device)
                dist.all_reduce(t)
                logs[key] = (t / dist.get_world_size()).item()

        if self.params.log_to_wandb:
            wandb.log(logs, step=self.epoch)

        return tr_time, data_time, logs

    # ------------------------------------------------------------------
    def validate_one_epoch(self) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate the current model on the validation split without gradient tracking.

        Args:
            None: Uses the internally stored validation DataLoader.

        Returns:
            Tuple[float, Dict[str, float]]: Wall-clock validation time and metrics dictionary.

        Raises:
            RuntimeError: Propagated if inference or metric accumulation fails.
        """
        self.model.eval()
        valid_loss = 0.0
        valid_l1 = 0.0
        valid_steps = 0

        start = time.time()
        with torch.no_grad():
            for i, data in enumerate(self.valid_data_loader, 0):
                inp, tar = map(lambda x: x.to(self.device, dtype=torch.float32), data)
                gen = self.model(inp)

                loss = self.loss_obj(gen, tar)
                l1 = torch.nn.functional.l1_loss(gen, tar)

                valid_loss += loss.item()
                valid_l1 += l1.item()
                valid_steps += 1

        valid_time = time.time() - start
        if valid_steps > 0:
            valid_loss /= valid_steps
            valid_l1 /= valid_steps

        logs = {"valid_loss": valid_loss, "valid_l1": valid_l1}

        # Average metrics across ranks so every rank agrees on the global
        # validation loss used for scheduling and early-stopping decisions.
        if dist.is_initialized():
            for key in sorted(logs.keys()):
                t = torch.tensor(logs[key], device=self.device)
                dist.all_reduce(t)
                logs[key] = (t / dist.get_world_size()).item()

        if self.params.log_to_wandb:
            wandb.log(logs, step=self.epoch)

        return valid_time, logs

    # ------------------------------------------------------------------
    def save_checkpoint(self, checkpoint_path: str, model=None) -> None:
        """
        Persist the training state dictionary to disk.

        Args:
            checkpoint_path (str): Filesystem path where the checkpoint will be saved.
            model (Optional[torch.nn.Module]): Model whose weights should be stored;
                defaults to the trainer's current model.

        Returns:
            None: Side effect writes a serialized checkpoint file.

        Raises:
            OSError: If the checkpoint cannot be written to disk.
        """
        if model is None:
            model = self.model

        state = {
            "iters": self.iters,
            "epoch": self.epoch,
            "model_state": model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(state, checkpoint_path)

    # ------------------------------------------------------------------

    def restore_checkpoint(self, checkpoint_path: str) -> None:
        """
        Loads a persisted checkpoint and restores the model, optimizer, and training counters.

        Args:
            checkpoint_path (str): Path to a checkpoint produced by `save_checkpoint`.

        Returns:
            None

        Raises:
            FileNotFoundError: If the checkpoint path does not point to a file.
            RuntimeError: If state_dict keys are incompatible with the current model.
        """
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' does not exist.")
        torch.serialization.add_safe_globals([ScalarFloat])
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model_state = checkpoint["model_state"]
        current_state = self.model.state_dict()

        try:
            self.model.load_state_dict(model_state)
        except RuntimeError:
            # Handle DDP/non-DDP key mismatches
            new_state_dict = {}
            if all(k.startswith("module.") for k in model_state.keys()) and \
            not all(k.startswith("module.") for k in current_state.keys()):
                # DDP → non-DDP: strip 'module.'
                new_state_dict = {k[7:]: v for k, v in model_state.items()}
            elif not any(k.startswith("module.") for k in model_state.keys()) and \
                all(k.startswith("module.") for k in current_state.keys()):
                # non-DDP → DDP: add 'module.'
                new_state_dict = {f"module.{k}": v for k, v in model_state.items()}
            else:
                # Fallback: use as is
                new_state_dict = model_state

            self.model.load_state_dict(new_state_dict)

        # Restore optimizer state if available
        if hasattr(self, "optimizer") and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Restore training counters
        self.iters = checkpoint.get("iters", 0)
        self.start_epoch = checkpoint.get("epoch", 0)
        self.epoch = self.start_epoch

        logging.info(f"Checkpoint '{checkpoint_path}' successfully restored.")

