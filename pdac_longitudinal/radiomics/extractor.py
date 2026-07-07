"""PyRadiomics feature extraction per ROI compartment (IT, PT_ring1–3, TVI)."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)

DEFAULT_FEATURE_CLASSES: List[str] = [
    "firstorder",
    "shape",
    "glcm",
    "glrlm",
    "glszm",
    "ngtdm",
    "gldm",
]

# PyRadiomics raises cryptic errors for near-empty masks.
MIN_MASK_VOXELS: int = 10


def _numpy_to_sitk(
    array: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
    is_mask: bool = False,
) -> sitk.Image:
    """Convert `(X, Y, Z)` NumPy array to a SimpleITK image with correct spacing.

    Args:
        array: Volume in `(X, Y, Z)` axis order.
        spacing_xyz: Voxel spacing `(x, y, z)` in mm.
        is_mask: If True, cast to `uint8`; otherwise `float32`.
    """
    if is_mask:
        arr = array.astype(np.uint8)
    else:
        arr = array.astype(np.float32)

    # GetImageFromArray expects (Z, Y, X); transpose from our (X, Y, Z) convention.
    sitk_img = sitk.GetImageFromArray(arr.transpose(2, 1, 0))
    sitk_img.SetSpacing(tuple(float(s) for s in spacing_xyz))
    return sitk_img


class RadiomicsExtractor:
    """PyRadiomics feature extractor for all ROI compartments.

    Args:
        feature_classes: PyRadiomics feature classes to enable; defaults to
            `DEFAULT_FEATURE_CLASSES`.
        settings_file: Optional PyRadiomics settings file; overrides the
            individual `binWidth`/`resegment_range` args when given.
        binWidth: Histogram bin width for texture features, in HU.
        sigma_values: LoG filter sigma values (mm); enables the LoG image type
            when given.
        resegment_range: `[min, max]` HU range for mask resegmentation before
            extraction.
        label_value: Mask label value identifying the ROI to extract from.
        verbose: Enable PyRadiomics' own verbose logging.

    Raises:
        ImportError: If PyRadiomics is not installed.
    """

    def __init__(
        self,
        feature_classes: Optional[List[str]] = None,
        settings_file: Optional[Union[str, Path]] = None,
        binWidth: float = 25.0,
        sigma_values: Optional[List[float]] = None,
        resegment_range: Optional[List[float]] = None,
        label_value: int = 1,
        verbose: bool = False,
    ) -> None:
        try:
            from radiomics import featureextractor
        except ImportError as exc:
            raise ImportError(
                "PyRadiomics is required for feature extraction.  "
                "Install with: uv add pyradiomics"
            ) from exc

        self.label_value = label_value
        self.feature_classes = feature_classes or DEFAULT_FEATURE_CLASSES

        settings: Dict = {
            "binWidth": binWidth,
            "label": label_value,
            "verbose": verbose,
        }
        if resegment_range is not None:
            settings["resegmentRange"] = resegment_range

        if settings_file is not None:
            self._extractor = featureextractor.RadiomicsFeatureExtractor(
                str(settings_file)
            )
            self._extractor.settings["label"] = label_value
        else:
            self._extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

        self._extractor.disableAllFeatures()
        for cls in self.feature_classes:
            self._extractor.enableFeatureClassByName(cls)

        self._extractor.disableAllImageTypes()
        self._extractor.enableImageTypeByName("Original")
        if sigma_values:
            self._extractor.enableImageTypeByName(
                "LoG", customArgs={"sigma": sigma_values}
            )
            logger.info("LoG filter enabled with sigma=%s mm", sigma_values)

        logger.info(
            "RadiomicsExtractor initialised: classes=%s", self.feature_classes
        )

    def extract_single(
        self,
        ct_array: np.ndarray,
        mask_array: np.ndarray,
        spacing_xyz: Tuple[float, float, float],
        compartment_name: str = "ROI",
    ) -> Dict[str, float]:
        """Extract radiomic features from one CT volume and one binary mask.

        Args:
            ct_array: CT volume in `(X, Y, Z)` axis order.
            mask_array: Binary ROI mask, same shape as `ct_array`.
            spacing_xyz: Voxel spacing `(x, y, z)` in mm.
            compartment_name: Label used in warnings/log messages.

        Returns:
            Dict mapping feature name to value; empty if the mask has fewer
            than `MIN_MASK_VOXELS` voxels or extraction fails.
        """
        n_voxels = int(mask_array.sum())
        if n_voxels < MIN_MASK_VOXELS:
            warnings.warn(
                f"[{compartment_name}] Mask has only {n_voxels} voxels "
                f"(minimum {MIN_MASK_VOXELS}); skipping extraction."
            )
            return {}

        ct_sitk  = _numpy_to_sitk(ct_array,   spacing_xyz, is_mask=False)
        mask_sitk = _numpy_to_sitk(mask_array, spacing_xyz, is_mask=True)

        try:
            result = self._extractor.execute(ct_sitk, mask_sitk, label=self.label_value)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"[{compartment_name}] PyRadiomics extraction failed: {exc}"
            )
            return {}

        features = {
            k: float(v)
            for k, v in result.items()
            if not k.startswith("diagnostics_") and not isinstance(v, str)
        }
        logger.debug(
            "[%s] Extracted %d features from %d voxels",
            compartment_name, len(features), n_voxels,
        )
        return features

    def extract_all_compartments(
        self,
        ct_array: np.ndarray,
        roi_masks: Dict[str, np.ndarray],
        spacing_xyz: Tuple[float, float, float],
        compartments: Optional[List[str]] = None,
        prefix: str = "",
    ) -> Dict[str, float]:
        """Extract features from all ROI compartments.

        Args:
            ct_array: CT volume in `(X, Y, Z)` axis order.
            roi_masks: Dict mapping compartment name to binary mask.
            spacing_xyz: Voxel spacing `(x, y, z)` in mm.
            compartments: Compartment names to extract; defaults to all keys
                in `roi_masks`.
            prefix: Prepended to every output key.

        Returns:
            Flat dict with keys `'{prefix}{comp}_{feat}'`, one entry per
            (compartment, feature) pair.
        """
        if compartments is None:
            compartments = list(roi_masks.keys())

        all_features: Dict[str, float] = {}
        for comp in compartments:
            if comp not in roi_masks:
                warnings.warn(
                    f"Compartment '{comp}' not found in roi_masks; skipping."
                )
                continue
            raw = self.extract_single(
                ct_array=ct_array,
                mask_array=roi_masks[comp],
                spacing_xyz=spacing_xyz,
                compartment_name=comp,
            )
            for feat_name, val in raw.items():
                key = f"{prefix}{comp}_{feat_name}"
                all_features[key] = val

        logger.info(
            "Extracted %d features across %d compartments (prefix='%s')",
            len(all_features), len(compartments), prefix,
        )
        return all_features
