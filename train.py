#!/usr/bin/env python3
import os
import sys
import argparse
import shutil
import torch

# Ensure our local pidsmaker is in the import path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from pidsmaker.config import get_runtime_required_args, get_yml_cfg, set_task_to_done
from pidsmaker.tasks import (
    construction,
    transformation,
    featurization,
    feat_inference,
    batching,
    training,
)
from pidsmaker.utils.utils import log, set_seed

import wandb

def main():
    wandb.init(mode="disabled")
    parser = argparse.ArgumentParser(description="Train Velox GNN pipeline on Parquet files.")
    parser.add_argument("--data-dir", default="./data", help="Directory containing preprocessed Parquet files.")
    parser.add_argument("--artifacts-dir", default="./artifacts", help="Directory to store pipeline artifacts.")
    args = parser.parse_args()

    data_dir_abs = os.path.abspath(args.data_dir)
    artifacts_dir_abs = os.path.abspath(args.artifacts_dir)

    os.makedirs(artifacts_dir_abs, exist_ok=True)

    # Initialize the base configuration using CLI arguments matching Velox EAUDIT_LOG run
    cli_args = [
        "velox", "EAUDIT_LOG",
        "--cpu",
        "--database_host", "none", # triggers Mock DB connector fallback
        "--artifact_dir", artifacts_dir_abs,
        "--force_restart", "construction,transformation,featurization,feat_inference,batching,training"
    ]
    
    pids_args = get_runtime_required_args(args=cli_args)
    cfg = get_yml_cfg(pids_args)

    # Force database configuration to use our custom Parquet mock directory
    cfg.database.data_dir = data_dir_abs
    cfg._use_cpu = True

    # Set seed for reproducibility
    set_seed(cfg)

    # Sequence of training pipeline tasks
    tasks = [
        ("construction", construction),
        ("transformation", transformation),
        ("featurization", featurization),
        ("feat_inference", feat_inference),
        ("batching", batching),
        ("training", training),
    ]

    log("=== Starting Velox GNN training pipeline ===")
    for name, module in tasks:
        log(f"Running task: {name}...")
        module.main(cfg)
        set_task_to_done(getattr(cfg, name)._task_path)
        log(f"Task '{name}' completed successfully.")

    log("=== Training pipeline finished successfully! ===")

if __name__ == "__main__":
    main()
