#!/usr/bin/env python3
import os
import sys
import argparse
import shutil
import torch

# Ensure our local pidsmaker is in the import path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from pidsmaker.config import get_runtime_required_args, get_yml_cfg, set_task_to_done
from pidsmaker.tasks import evaluation
from pidsmaker.utils.utils import log, set_seed

import wandb

def main():
    wandb.init(mode="disabled")
    parser = argparse.ArgumentParser(description="Evaluate Velox GNN pipeline on Parquet files.")
    parser.add_argument("--data-dir", default="./data", help="Directory containing preprocessed Parquet files.")
    parser.add_argument("--artifacts-dir", default="./artifacts", help="Directory to store pipeline artifacts.")
    parser.add_argument("--eval-on-train", action="store_true", help="Evaluate on the train set instead of the test set.")
    args = parser.parse_args()

    data_dir_abs = os.path.abspath(args.data_dir)
    artifacts_dir_abs = os.path.abspath(args.artifacts_dir)

    # Initialize the base configuration using CLI arguments matching Velox EAUDIT_LOG run
    cli_args = [
        "velox", "EAUDIT_LOG",
        "--cpu",
        "--database_host", "none", # triggers Mock DB connector fallback
        "--artifact_dir", artifacts_dir_abs,
        "--force_restart", "evaluation"
    ]
    
    pids_args = get_runtime_required_args(args=cli_args)
    cfg = get_yml_cfg(pids_args)

    # Force database configuration to use our custom Parquet mock directory
    cfg.database.data_dir = data_dir_abs
    cfg._use_cpu = True

    # Set seed for reproducibility
    set_seed(cfg)

    # If evaluating on train split, copy the train split losses to the test split location
    if args.eval_on_train:
        train_losses_path = os.path.join(cfg.training._edge_losses_dir, "train")
        test_losses_path = os.path.join(cfg.training._edge_losses_dir, "test")
        if os.path.exists(train_losses_path):
            log(f"Evaluating on Train Set: Copying '{train_losses_path}' to '{test_losses_path}'...")
            if os.path.exists(test_losses_path):
                shutil.rmtree(test_losses_path)
            shutil.copytree(train_losses_path, test_losses_path)
        else:
            log(f"Warning: Train losses path '{train_losses_path}' not found. Defaulting to standard test evaluation.")

    log("=== Starting Velox GNN evaluation pipeline ===")
    evaluation.main(cfg)
    set_task_to_done(cfg.evaluation._task_path)
    log("=== Evaluation pipeline finished successfully! ===")

if __name__ == "__main__":
    main()
