"""Typed configuration schema for the PDAC longitudinal framework."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import yaml


# Well-known sections

@dataclass(frozen=True)
class DataConfig:
    """Filesystem paths + dataset layout.

    Attributes:
        root_dir: Root directory of raw imaging data.
        splits_file: Path to the train/val/test split file.
        nifti_dir: Composer output directory override.
        labels_csv: Path to the clinical labels CSV.
        cache_dir: Directory for the preprocessed CT cache.
        segmenter_weights_path: Path to the pretrained PanTS segmenter checkpoint.
        phase: CT phase to load.
        allowed_regions: Composer region tags to include.
        post_nat_tps: Post-NAT timepoint tags accepted as T1, tried in order.
            Default `("t1",)` uses only a true t1; add more (e.g. "t2") to fall
            back to a later scan when t1 is missing.
        exclude_cases: Patient IDs to exclude.
        include_cohorts: Restrict training/eval to these cohorts. Empty =
            all cohorts.
        clinical_completeness_weighting: Scale each z-scored clinical
            feature by its train-fold observed fraction.
        clinical_missingness_flags: Add a `<col>__isna` indicator feature
            for every clinical column with missing values.
        radiomic_pca_components: Fold-internal PCA on the radiomic token
            (0 = off).
        cache_version: Embedded in the cache filename; bump to invalidate
            old caches.
        t0_filename: Raw T0 CT filename template.
        t1_filename: Raw T1 CT filename template.
    """
    root_dir: Path = Path("./data/raw")
    splits_file: Optional[Path] = None
    nifti_dir: Optional[Path] = None
    labels_csv: Optional[Path] = None
    cache_dir: Optional[Path] = None
    segmenter_weights_path: Optional[Path] = None

    # Composer-specific filtering
    phase: str = "venous"
    allowed_regions: Tuple[str, ...] = (
        "abdomen_high_conf", "abdomen", "abdomen_partial",
    )
    post_nat_tps: Tuple[str, ...] = ("t1",)
    exclude_cases: Tuple[str, ...] = ()
    include_cohorts: Tuple[str, ...] = ()

    clinical_completeness_weighting: bool = False
    clinical_missingness_flags: bool = False
    radiomic_pca_components: int = 0
    cache_version: str = "v3.5"

    # Raw filename templates
    t0_filename: str = "T0.nii.gz"
    t1_filename: str = "T1.nii.gz"


@dataclass(frozen=True)
class PreprocessingConfig:
    """CT preprocessing + patch geometry.

    Attributes:
        ct_clip_min: Lower HU clip bound.
        ct_clip_max: Upper HU clip bound.
        ct_norm_mean: Fixed normalisation mean; unset derives stats from
            the volume.
        ct_norm_std: Fixed normalisation std; unset derives stats from the
            volume.
        ct_norm_shared_pair: When set (and mean/std unset), z-score both
            timepoints from shared foreground-voxel stats.
        target_spacing: Resample target spacing in mm (x, y, z).
        patch_size: Cropped patch size in voxels (x, y, z).
        crop_to_foreground: Crop each volume to its foreground bbox.
        foreground_margin_voxels: Margin added around the foreground bbox.
        shared_crop_frame: Crop T1 using T0's bbox so both patches cover
            the same anatomy.
        reuse_saved_segs: Reuse cached segmentation files from earlier
            cache builds, skipping PanTS.
        viz_cache_every: Render an ROI-overlay PNG every Nth case to
            `<cache_dir>/_viz/` (0 = off).
        pants_target_spacing_zyx: PanTS inference target spacing (z, y, x),
            from the checkpoint plans.
        pants_patch_zyx: PanTS inference patch size (z, y, x).
        pants_step: PanTS sliding-window step fraction.
    """
    ct_clip_min: float = -150.0
    ct_clip_max: float = 250.0
    ct_norm_mean: Optional[float] = None
    ct_norm_std:  Optional[float] = None
    ct_norm_shared_pair: bool = True
    target_spacing: Tuple[float, float, float] = (1.5, 1.5, 1.5)
    patch_size: Tuple[int, int, int] = (192, 192, 192)
    crop_to_foreground: bool = True
    foreground_margin_voxels: int = 100
    shared_crop_frame: bool = True
    reuse_saved_segs: bool = True
    viz_cache_every: int = 0
    pants_target_spacing_zyx: Tuple[float, float, float] = (1.0, 0.787, 0.801)
    pants_patch_zyx: Tuple[int, int, int] = (128, 192, 288)
    pants_step: float = 0.5


@dataclass(frozen=True)
class AugmentationConfig:
    """Training-time augmentation magnitudes (z-score units; applied per sample).

    Attributes:
        flip_x_prob: Probability of a left-right flip.
        intensity_prob: Probability of applying noise + brightness jitter.
        noise_sigma: Additive Gaussian noise sigma.
        brightness_bias: Uniform +/- bias additive shift.
        spatial_prob: Probability of the spatial deformation transform
            (axes: X=L-R, Y=A-P, Z=cranio-caudal).
        axial_deg: +/- rotation about Z (degrees).
        tilt_deg: +/- rotation about X and Y (degrees).
        scale_range: Isotropic zoom range.
        elastic_prob: Probability of elastic deformation.
        elastic_voxels: Elastic displacement std, in voxels.
        elastic_sigma: Elastic smoothing sigma, in voxels.
        mult_bright_prob: Probability of multiplicative brightness jitter.
        mult_bright_range: Multiplicative brightness factor range.
        contrast_prob: Probability of contrast jitter.
        contrast_range: Contrast factor range.
        blur_prob: Probability of Gaussian blur.
        blur_sigma: Blur sigma range.
        lowres_prob: Probability of simulated low-resolution downsampling.
        lowres_range: Low-resolution downsampling factor range.
    """
    flip_x_prob: float = 0.5
    intensity_prob: float = 0.7
    noise_sigma: float = 0.05
    brightness_bias: float = 0.10

    # Spatial deformation
    spatial_prob: float = 0.0
    axial_deg: float = 25.0
    tilt_deg: float = 7.0
    scale_range: Tuple[float, float] = (0.9, 1.1)
    elastic_prob: float = 0.0
    elastic_voxels: float = 3.0
    elastic_sigma: float = 12.0

    # Acquisition-variation intensity aug
    mult_bright_prob: float = 0.0
    mult_bright_range: Tuple[float, float] = (0.85, 1.15)
    contrast_prob: float = 0.0
    contrast_range: Tuple[float, float] = (0.75, 1.25)
    blur_prob: float = 0.0
    blur_sigma: Tuple[float, float] = (0.5, 1.25)
    lowres_prob: float = 0.0
    lowres_range: Tuple[float, float] = (0.5, 1.0)


@dataclass(frozen=True)
class ModulesConfig:
    """Which input branches are active in this run.

    Attributes:
        imaging: Enable the CT imaging branch.
        clinical: Enable the clinical feature branch.
        anatomy: Enable the anatomy feature branch.
        vessel: Enable the vessel-tumour metrics branch.
        radiomics: Enable the radiomics feature branch.
    """
    imaging:   bool = True
    clinical:  bool = True
    anatomy:   bool = True
    vessel:    bool = True
    radiomics: bool = False


@dataclass(frozen=True)
class TrainingConfig:
    """Optimization + scheduling.

    Attributes:
        task: "survival" -> Cox PH loss + C-index; "binary" -> BCE past
            `survival_horizon_months` + ROC-AUC.
        survival_horizon_months: Horizon for the binary task and its AUC.
        binary_pos_weight: Binary-task BCE positive-class weight: "auto",
            a float, or null.
        clinical_baseline_enabled: Fit an inline lifelines CoxPH on the
            same fold as a clinical-only reference.
        clinical_baseline_results: When set, overrides the inline fit with
            a results.json path.
        attn_guidance_enabled: Supervise cross-timepoint attention to
            concentrate on the ROI union.
        attn_guidance_coef: Weight of the attention-guidance loss term.
        attn_guidance_stage: Stage(s) whose attention is supervised: an
            int (-1 = last) or a list of stage indices.
        attn_guidance_roi: ROI regions forming the guidance target mask:
            tumour (mask_it), peritumoural (mask_pt1/2/3), tvi (mask_tvi),
            liver (liver_t0/t1).
        attn_guidance_roi_weights: Per-region weights aligned with
            `attn_guidance_roi`. Empty = single union.
        phase_adversarial_enabled: Enable the phase-adversarial GRL.
        phase_adv_coef: Weight of the phase-adversarial loss term.
        output_dir: Root output directory for run artifacts.
        num_epochs: Number of training epochs.
        batch_size: Training batch size.
        optimizer: Optimizer name.
        learning_rate: Peak learning rate.
        weight_decay: AdamW weight decay.
        scheduler: LR scheduler name.
        warmup_epochs: Linear warmup epochs before the cosine schedule.
        use_amp: Enable mixed-precision training.
        amp_dtype: "float16" or "bfloat16".
        unfreeze_encoder_at_epoch: Epoch at which the encoder is unfrozen.
        unfreeze_encoder_stages: Encoder stages to unfreeze at
            fine-tuning. Empty = unfreeze all stages.
        save_every_n_epochs: Checkpoint save interval, in epochs.
        early_stopping_patience: Epochs without val improvement before
            stopping.
        seed: Random seed.
        num_workers: DataLoader worker count.
        device: Torch device string; null -> auto.
        attention_viz_every: Render attention overlays every N epochs.
        attention_viz_cases: Number of val cases to render attention for.
        attention_viz_case_ids: Specific case IDs to render attention for;
            None = the first `attention_viz_cases` val cases.
        attention_viz_skip_stages: Stages not materialised during the viz
            pass.
        max_seg_tiles: Cap on nnU-Net RAM use (tile count).
        max_cases: Cap dataset size, for debugging.
    """
    task: str = "survival"
    survival_horizon_months: float = 12.0
    binary_pos_weight: str = "auto"
    clinical_baseline_enabled: bool = True
    clinical_baseline_results: Optional[Path] = None
    attn_guidance_enabled: bool = False
    attn_guidance_coef: float = 0.0
    attn_guidance_stage: Union[int, List[int]] = -1
    attn_guidance_roi: Tuple[str, ...] = ("tumour", "peritumoural")
    attn_guidance_roi_weights: Tuple[float, ...] = ()
    phase_adversarial_enabled: bool = False
    phase_adv_coef: float = 0.001
    output_dir: Path = Path("./outputs")
    num_epochs: int = 100
    batch_size: int = 4
    optimizer: str = "adamw"
    learning_rate: float = 5.0e-5
    weight_decay: float = 1.0e-3
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    use_amp: bool = True
    amp_dtype: str = "float16"
    unfreeze_encoder_at_epoch: int = 40
    unfreeze_encoder_stages: List[int] = field(default_factory=lambda: [3, 4, 5])
    save_every_n_epochs: int = 10
    early_stopping_patience: int = 20
    seed: int = 42

    # Misc training behaviour
    num_workers: int = 8
    device: Optional[str] = None
    attention_viz_every: int = 5
    attention_viz_cases: int = 4
    attention_viz_case_ids: Optional[List[str]] = None
    attention_viz_skip_stages: Tuple[int, ...] = (0,)
    max_seg_tiles: Optional[int] = 300
    max_cases: Optional[int] = None


@dataclass(frozen=True)
class CVConfig:
    """K-fold cross-validation orchestration.

    Attributes:
        enabled: Enable cross-validation.
        n_folds: Number of folds.
        fold: Fold index; set per array task.
        seed: Fold-split random seed.
        folds_file: Persisted fold assignment path (None = auto-derived
            next to `splits_file`).
    """
    enabled: bool = False
    n_folds: int = 5
    fold: Optional[int] = None
    seed: int = 42
    folds_file: Optional[str] = None


@dataclass(frozen=True)
class AnalysisConfig:
    """Options for the analyze and evaluate commands.

    Attributes:
        shap: analyze: also run GradientSHAP.
        on_test: analyze: use the held-out test set, not val.
        perm_through_pca: analyze: per-named-feature delta-C through the
            fold PCA.
        tta: evaluate: average each prediction with its L-R flip.
    """
    shap: bool = False
    on_test: bool = False
    perm_through_pca: bool = False
    tta: bool = False


@dataclass(frozen=True)
class WandbConfig:
    """Experiment tracking.

    Attributes:
        enabled: Enable W&B logging.
        project: W&B project name.
        entity: W&B entity/team name.
        mode: W&B mode ("online", "offline", "disabled").
        run_name: Explicit run name; None -> auto-generated.
        dir: Local directory for W&B run files.
        tags: Run tags.
    """
    enabled: bool = True
    project: str = "pdac-longitudinal"
    entity: Optional[str] = None
    mode: str = "online"
    run_name: Optional[str] = None
    dir: Optional[Path] = None
    tags: Mapping[str, str] = field(default_factory=dict)


# Top-level config

@dataclass(frozen=True)
class Config:
    """Top-level config tree.

    Attributes:
        data: Filesystem paths + dataset layout.
        preprocessing: CT preprocessing + patch geometry.
        augmentation: Training-time augmentation magnitudes.
        modules: Which input branches are active.
        training: Optimization + scheduling.
        cv: K-fold cross-validation orchestration.
        analysis: Options for the analyze and evaluate commands.
        wandb: Experiment tracking.
        encoder: Encoder architecture section, an open pass-through dict.
        attention: Attention module section, an open pass-through dict.
        fusion: Fusion module section, an open pass-through dict.
        roi_pipeline: ROI pipeline section, an open pass-through dict.
        radiomics: Radiomics section, an open pass-through dict.
        registration: Registration section, an open pass-through dict.
        _raw: Raw source dict, kept for debugging.
    """
    data: DataConfig                       = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig     = field(default_factory=PreprocessingConfig)
    augmentation: AugmentationConfig       = field(default_factory=AugmentationConfig)
    modules: ModulesConfig                 = field(default_factory=ModulesConfig)
    training: TrainingConfig               = field(default_factory=TrainingConfig)
    cv: CVConfig                           = field(default_factory=CVConfig)
    analysis: AnalysisConfig               = field(default_factory=AnalysisConfig)
    wandb: WandbConfig                     = field(default_factory=WandbConfig)

    encoder:        Dict[str, Any]         = field(default_factory=dict)
    attention:      Dict[str, Any]         = field(default_factory=dict)
    fusion:         Dict[str, Any]         = field(default_factory=dict)
    roi_pipeline:   Dict[str, Any]         = field(default_factory=dict)
    radiomics:      Dict[str, Any]         = field(default_factory=dict)
    registration:   Dict[str, Any]         = field(default_factory=dict)

    _raw: Dict[str, Any]                   = field(default_factory=dict, repr=False)

    # Loaders

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load a Config from a YAML file.

        Args:
            path: Path to the YAML config file.
        """
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Config":
        """Build a Config from a plain dict (post yaml.safe_load).

        Args:
            raw: Config as a plain nested dict.
        """
        raw = _expand_paths(raw)
        kwargs: Dict[str, Any] = {}
        for f in fields(cls):
            if f.name.startswith("_"):
                continue
            section_raw = raw.get(f.name, {})
            if is_dataclass(f.default_factory()) if callable(f.default_factory) else False:  # type: ignore[arg-type]
                kwargs[f.name] = _build_dataclass(f.default_factory, section_raw, f.name)  # type: ignore[arg-type]
            else:
                # Architectural / open dict section
                kwargs[f.name] = dict(section_raw) if section_raw else {}
        kwargs["_raw"] = dict(raw)
        return cls(**kwargs)

    # Convenience: per-section override after load

    def with_overrides(self, **section_overrides: Mapping[str, Any]) -> "Config":
        """Return a new Config with one or more sections replaced.

        Args:
            section_overrides: Section name -> dict of field overrides to
                merge into that section.
        """
        merged_raw = dict(self._raw)
        for section, overrides in section_overrides.items():
            current = dict(merged_raw.get(section, {}))
            current.update(overrides or {})
            merged_raw[section] = current
        return Config.from_dict(merged_raw)


# Internals

def _build_dataclass(factory, raw: Mapping[str, Any], section_name: str):
    """Construct a dataclass from a YAML section, validating field names.

    Args:
        factory: Zero-arg default factory for the target dataclass.
        raw: Section's raw dict, as loaded from YAML.
        section_name: Section name, used in the unknown-key error message.

    Raises:
        ValueError: If `raw` contains keys not defined on the target
            dataclass.
    """
    target_cls = factory().__class__
    valid_names = {f.name for f in fields(target_cls)}
    unknown = set(raw.keys()) - valid_names
    if unknown:
        raise ValueError(
            f"Unknown keys in [{section_name}]: {sorted(unknown)}. "
            f"Known keys: {sorted(valid_names)}"
        )
    typed: Dict[str, Any] = {}
    for name, value in raw.items():
        if value is None:
            continue
        typed[name] = _coerce(value, _annotation_for(target_cls, name))
    return target_cls(**typed)


def _annotation_for(dc_cls, field_name: str):
    """Return the declared type annotation for `field_name` on `dc_cls`, or None.

    Args:
        dc_cls: Dataclass to inspect.
        field_name: Name of the field to look up.
    """
    for f in fields(dc_cls):
        if f.name == field_name:
            return f.type
    return None


def _expand_paths(value: Any) -> Any:
    """Recursively expand `~` and `$VAR` in every leaf string of a nested structure.

    Args:
        value: Nested dict / list / scalar to expand paths within.
    """
    import os
    if isinstance(value, str):
        if "~" in value or "$" in value:
            return os.path.expandvars(os.path.expanduser(value))
        return value
    if isinstance(value, Mapping):
        return {k: _expand_paths(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_paths(v) for v in value]
    return value


def _coerce(value: Any, annotation: Any) -> Any:
    """Best-effort coercion for Path strings and tuples from lists.

    Args:
        value: Raw value from YAML to coerce.
        annotation: Target dataclass field's type annotation.

    Returns:
        The coerced value, or `value` unchanged if no coercion applies.
    """
    if annotation is None:
        return value
    ann_str = str(annotation)

    # Path; expand ~ and $vars.
    if "Path" in ann_str and isinstance(value, (str, Path)):
        import os
        return Path(os.path.expandvars(os.path.expanduser(str(value))))

    # Tuple[...]
    if ann_str.startswith("Tuple") or ann_str.startswith("tuple"):
        if isinstance(value, (list, tuple)):
            return tuple(value)

    return value
