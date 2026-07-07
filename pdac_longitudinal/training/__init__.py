"""Training utilities for the PDAC longitudinal framework."""

from pdac_longitudinal.training.checkpointing import (
    CheckpointPaths,
    load_checkpoint,
    save_checkpoint,
)
from pdac_longitudinal.training.metrics import concordance_index
from pdac_longitudinal.training.loop import fit, train_step, val_step
from pdac_longitudinal.visualisation.attention_viz import render_attention_maps
from pdac_longitudinal.training.cv import (
    build_cv_loaders,
    build_dataset_kwargs,
    build_shared_segmenter,
    cache_all_cases,
    release_segmenter,
    shard_filter_fn,
)
from pdac_longitudinal.training.wandb_setup import (
    finish_wandb,
    init_wandb,
    log_attention_images,
    log_metrics,
    log_seg_overlay,
)

__all__ = [
    # Checkpointing
    "CheckpointPaths",
    "load_checkpoint",
    "save_checkpoint",
    # Metrics
    "concordance_index",
    # Loop
    "fit",
    "train_step",
    "val_step",
    # CV / data building
    "build_cv_loaders",
    "build_dataset_kwargs",
    "build_shared_segmenter",
    "cache_all_cases",
    "release_segmenter",
    "shard_filter_fn",
    # Attention viz
    "render_attention_maps",
    # W&B
    "finish_wandb",
    "init_wandb",
    "log_attention_images",
    "log_metrics",
    "log_seg_overlay",
]
