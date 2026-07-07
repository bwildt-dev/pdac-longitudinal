# PDAC longitudinal framework

Survival prediction from longitudinal CT in pancreatic ductal adenocarcinoma
(PDAC). Each patient has two scans: baseline (T0) and post-neoadjuvant (T1).
A nnU-Net siamese backbone encodes both with a cross-timepoint attention, and the
result is combined with per-compartment radiomics and clinical covariates to
predict survival from the T1 scan onwards.

## Install

```bash
uv sync
```

This installs the package and its `pdac_longitudinal` command.

The segmenter weights ship separately as a release. Download **PeriPancSeg v1.0**
from this repository's Releases page and point both `weights_path` fields at it:
`data.segmenter_weights_path` runs the segmenter during preprocessing, and
`encoder.weights_path` loads the checkpoint's encoder into the trainable model.

The clinical labels and the preprocessed CT cache are cohort-specific patient
data, so you provide those yourself. See `configs/example_config.yaml` for where
all the paths go.

Radiomics is an optional extra (pyradiomics compiles from source, so it needs a
C compiler):

```bash
uv sync --extra radiomics
```

With it installed, `pdac_longitudinal preprocess` extracts per-compartment
radiomics automatically as its final step.

## Command-line interface

A single `pdac_longitudinal` entry point dispatches to the framework's commands:

```bash
pdac_longitudinal preprocess      --config configs/my_config.yaml --shard 0/4   # build the .npz cache (+ radiomics)
pdac_longitudinal train           --config configs/my_config.yaml --cv-fold 0   # train cross-validation fold 0
pdac_longitudinal evaluate        --config configs/my_config.yaml --run-dir outputs/my_run   # held-out test ensemble
pdac_longitudinal analyze         --config configs/my_config.yaml --checkpoint <ckpt> --cv-fold 0   # feature importance
pdac_longitudinal verify          --config configs/my_config.yaml --cv-fold 0   # validate a config, report split sizes
```

Run `pdac_longitudinal <command> --help` for a command's options. Every run is fully
described by its YAML config; only operational knobs live on the command line.

`labels.csv` (clinical covariates + survival labels) isn't built by the package. It's
cohort-specific ETL you write yourself. The only required columns are `patient_id`,
`cohort`, `time_months`, `status`; every other column is treated as a clinical feature.

### A typical cross-validation run

```bash
# 1. Build the cache once (parallelise across N processes with --shard K/N).
#    With modules.radiomics on, this also extracts per-compartment radiomics.
pdac_longitudinal preprocess --config configs/my_config.yaml

# 2. Train each fold (one process per fold).
for k in 0 1 2 3 4; do
  pdac_longitudinal train --config configs/my_config.yaml --cv-fold "$k"
done

# 3. Average the fold checkpoints on the held-out test set.
pdac_longitudinal evaluate --config configs/my_config.yaml --run-dir outputs/my_run
```

## Configuration

`configs/example_config.yaml` documents every section: data paths, the ROI
pipeline (peritumoural rings + tumour-vessel interface), preprocessing,
augmentation, optional deformable registration, the encoder, cross-timepoint
attention, radiomics, the fusion head, module toggles, training, cross-validation,
and analysis. Copy it, adjust the paths, and pass it to every command.

## Project layout

```
pdac_longitudinal/
  cli/            command entry points (train, evaluate, analyze, preprocess, verify)
  config.py       typed configuration schema
  data/           dataset, clinical registry, splits, augmentation
  preprocess/     segmentation, ROI masks, anatomy/vessel/radiomic features, cache build
  registration/   optional deformable T1->T0 registration
  models/         encoder, cross-timepoint attention, longitudinal model, GRL
  fusion/         token-fusion head
  losses/         Cox PH / binary survival, attention guidance
  radiomics/      per-compartment radiomic extraction
  baselines/      clinical-only Cox baseline
  training/       training loop, cross-validation, ensemble evaluation, setup
  analysis/       feature-importance (permutation + GradientSHAP)
  visualisation/  attention and ROI overlays
```

The DICOM to NIfTI dataset assembly lives in a separate package,
[dataset-composer](https://github.com/bwildt-dev/dataset-composer), consumed
here as a dependency.
