# Model Card

This card covers two models: the **PeriPancSeg** segmenter distributed as a
release with this repository, and the **longitudinal survival model** that the
framework trains from your own data (no survival weights are distributed).

## Intended use

- **Research use only.** Both models are intended for methodological research on
  longitudinal PDAC imaging: retrospective survival modelling, ablation studies,
  and reproduction or extension of the associated work.
- PeriPancSeg produces peripancreatic anatomy segmentations that drive the
  preprocessing pipeline (ROI masks, anatomy and vessel features).
- The survival model predicts a relative risk score from a baseline (T0) and
  post-neoadjuvant (T1) CT pair, fused with per-compartment radiomics and
  clinical covariates.

## Not for clinical use

Neither model is a medical device and neither is validated for clinical
decision-making, diagnosis, prognosis, or treatment planning. Outputs are risk
scores and segmentations for research, not patient care. Do not use them to
inform decisions about individual patients.

## Models

**PeriPancSeg** (released weights)
- 3D nnU-Net (ResEnc-L), 5-fold ensemble, single-channel CT input.
- Segments 18 peripancreatic structures plus background (pancreas, tumour, the
  peripancreatic vessels and ducts, and neighbouring upper-abdominal organs).
- Trained at ~1 mm spacing; see the release notes for the exact configuration.

**Longitudinal survival model** (trained by you, not distributed)
- Frozen PeriPancSeg encoder in a siamese configuration over T0 and T1, combined
  by a cross-timepoint attention transformer.
- Fused with per-compartment radiomics and clinical covariates by a token-fusion
  head; trained with a Cox partial-likelihood (or binary-horizon) objective.

## Training data

- **PeriPancSeg** is derived from the PanTS dataset (Li et al., NeurIPS 2025).
  It inherits PanTS's non-commercial terms; see the license section.
- **Survival model:** no trained weights are shipped. You train it on your own
  cohort of paired pre/post-neoadjuvant CT with survival labels. The framework
  was developed on a private PDAC cohort that is not part of this repository.

## Limitations

- Segmentation quality depends on scan protocol, contrast phase, and field of
  view; out-of-distribution scans (unusual acquisition, non-abdominal crops,
  heavy artefact) may segment poorly and silently degrade downstream features.
- The survival model is trained within a batch's own risk set and on modest
  cohort sizes; absolute risk calibration is not guaranteed and results do not
  transfer across cohorts without re-training and re-validation.
- The pipeline assumes one baseline and one post-neoadjuvant scan per patient.
- Predictions reflect biases in the training cohort and are not fairness-audited.

## Segmentation dependency

The framework requires the PeriPancSeg weights to run: segmentation is the only
GPU-mandatory step and produces every ROI mask and anatomy/vessel feature the
model consumes. Point `data.segmenter_weights_path` and `encoder.weights_path`
at a downloaded fold. Without these weights the preprocessing pipeline cannot
build its cache.

## License

- **Code:** MIT (see `LICENSE`). Covers this framework's own source only.
- **PeriPancSeg weights:** research/non-commercial use only. Derived from PanTS
  (CC BY-NC-ND 4.0, https://github.com/MrGiovanni/PanTS); attribute PanTS and
  observe its non-commercial and no-derivatives terms when using or
  redistributing the weights.

The MIT code license does not extend to the model weights. The two are
distributed under different terms and should be treated separately.
