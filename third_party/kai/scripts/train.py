# scripts/train.py
"""
Copyright (c) 2025 MetaEarth-Lab
Licensed under the BSD 3-Clause License. Redistribution and use in source and binary forms, with or without modification, are permitted. See the LICENSE file in the project root for the full license text.
"""

import os
from pathlib import Path
import sys
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))
from src.evaluator import run_post_training_evaluation
import argparse
import logging

import numpy as np
import torch
import torch.distributed as dist
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict

from src.utils import YParams, setup_logging
from src.trainer import Trainer
from src.trainer import set_seed


def main():
    """
    Launch the training workflow using CLI-provided configuration.

    Args:
        None: The function relies on command-line arguments parsed internally.

    Returns:
        None: This function triggers side effects such as training and logging.

    Raises:
        RuntimeError: Propagated if distributed initialization or trainer setup fails.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", default="01", type=str)
    parser.add_argument("--yaml_config", default="./configs/KAI.yaml", type=str)
    parser.add_argument("--config", default="default", type=str)
    parser.add_argument("--enable_amp", action="store_true")
    parser.add_argument("--epsilon_factor", default=0.0, type=float)
    parser.add_argument("--seed", default=777, type=int)
    args = parser.parse_args()
    # -----------------------------
    # Load YAML config
    # -----------------------------
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    params["epsilon_factor"] = args.epsilon_factor

    # -----------------------------
    # Distributed setup
    # -----------------------------
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    params["world_size"] = world_size

    world_rank = 0
    local_rank = 0
    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_rank = dist.get_rank()
        params["global_batch_size"] = params.batch_size
        params["batch_size"] = int(params.batch_size // world_size)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        torch.backends.cudnn.benchmark = True
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")


    set_seed(args.seed)

    # -----------------------------
    # Experiment directory & paths
    # -----------------------------
    run_name = f"{args.config}_{args.run_num}"
    params["name"] = run_name
    # W&B run metadata — prefer config values, fall back to sensible defaults.
    params["group"] = getattr(params, "wandb_group", None) or f"KAI_{args.config}"
    params["project"] = getattr(params, "wandb_project", None) or "KAI"
    params["entity"] = getattr(params, "wandb_entity", None) or "NA"

    # Honor the W&B mode (online | offline | disabled) without changing code paths.
    wandb_mode = getattr(params, "wandb_mode", None)
    if wandb_mode:
        os.environ["WANDB_MODE"] = str(wandb_mode)

    base_exp_dir = params.exp_dir  # from YAML
    exp_dir = os.path.join(base_exp_dir, run_name)

    if world_rank == 0:
        os.makedirs(exp_dir, exist_ok=True)
        

    params["experiment_dir"] = os.path.abspath(exp_dir)
    params["checkpoint_path"] = os.path.join(exp_dir, "ckpt.tar")
    params["best_checkpoint_path"] = os.path.join(exp_dir, "best_ckpt.tar")
    log_path = os.path.join(exp_dir, "out.log")
    setup_logging(rank=world_rank, log_file=log_path if world_rank == 0 else None)
    params.log()
    resuming = os.path.isfile(params.checkpoint_path)
    params["resuming"] = resuming
    params["local_rank"] = local_rank
    params["enable_amp"] = args.enable_amp
    # Only log to wandb/screen from rank 0
    params["log_to_wandb"] = (world_rank == 0) and params["log_to_wandb"]
    params["log_to_screen"] = (world_rank == 0) and params["log_to_screen"]

    # Ensure channels are plain Python lists
    params["in_channels"] = list(params.in_channels)
    params["out_channels"] = list(params.out_channels)

    # Build and run trainer
    trainer = Trainer(params, world_rank=world_rank, local_rank=local_rank)
    trainer.train()

    if world_rank == 0 and getattr(params, 'enable_evaluation', False):
        logging.info("")
        logging.info("=" * 80)
        logging.info("STARTING POST-TRAINING EVALUATION")
        logging.info("=" * 80)
        
        # Determine checkpoint path
        if hasattr(params, 'eval_checkpoint_path') and params.eval_checkpoint_path:
            eval_ckpt = params.eval_checkpoint_path
        else:
            eval_ckpt = params.best_checkpoint_path
        
        # Verify checkpoint exists
        if not os.path.exists(eval_ckpt):
            logging.error(f"Checkpoint not found: {eval_ckpt}")
            logging.error("Skipping evaluation.")
        else:
            try:
                # Unwrap the DDP wrapper (if any) so evaluation runs on the bare model.
                eval_model = getattr(trainer.model, "module", trainer.model)
                # Run evaluation
                eval_results = run_post_training_evaluation(
                    model=eval_model,
                    config=params,
                    logger=logging,
                    checkpoint_path=eval_ckpt
                )
                
                logging.info("")
                logging.info("=" * 80)
                logging.info("POST-TRAINING EVALUATION COMPLETED SUCCESSFULLY")
                logging.info("=" * 80)
                
            except Exception as e:
                logging.error(f"Evaluation failed: {str(e)}")
                logging.error("Traceback:")
                import traceback
                traceback.print_exc()

    if world_rank == 0:
        logging.info("Training run complete (rank %d).", world_rank)

    # Clean shutdown of the process group in distributed runs.
    if dist.is_initialized():
        dist.destroy_process_group()



if __name__ == "__main__":
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    sys.path.insert(0, str(project_root))
    main()
