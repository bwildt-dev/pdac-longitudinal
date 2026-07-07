"""Wrapper around the pretrained PanTS nnU-Net v2 ResEncL for 19-class abdominal segmentation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import SimpleITK as sitk
import torch

logger = logging.getLogger(__name__)

PANTS_LABELS: Dict[str, int] = {
    "background": 0,
    "pancreas": 1,
    "tumor": 2,
    "superior_mesenteric_artery": 3,
    "celiac_artery": 4,
    "veins": 5,
    "postcava": 6,
    "pancreatic_duct": 7,
    "common_bile_duct": 8,
    "duodenum": 9,
    "stomach": 10,
    "colon": 11,
    "spleen": 12,
    "liver": 13,
    "kidney_left": 14,
    "kidney_right": 15,
    "adrenal_gland_left": 16,
    "adrenal_gland_right": 17,
    "gall_bladder": 18,
}

VESSEL_LABELS: Dict[str, int] = {
    "superior_mesenteric_artery": 3,
    "celiac_artery": 4,
    "veins": 5,
    "postcava": 6,
}

BILIARY_LABELS: Dict[str, int] = {
    "pancreatic_duct": 7,
    "common_bile_duct": 8,
}

class PanTSSegmenter:
    """Full-model PanTS segmenter wrapping `nnUNetPredictor`.

    Args:
        device: Torch device to run inference on. Defaults to CUDA if available, else CPU.
        perform_everything_on_device: If `True`, the per-volume aggregation buffer lives on
            GPU; fastest, but VRAM scales with input size and can OOM on large CTs.

    Raises:
        FileNotFoundError: If `weights_path` does not exist.
    """

    def __init__(
        self,
        weights_path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        tile_step_size: float = 0.5,
        use_gaussian: bool = True,
        use_mirroring: bool = False,
        perform_everything_on_device: bool = False,
        verbose: bool = False,
    ) -> None:
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"PanTS weights not found: {self.weights_path}"
            )

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device) if isinstance(device, str) else device

        logger.info(
            "Initialising PanTSSegmenter on %s (aggregate_on_device=%s)",
            self.device, perform_everything_on_device,
        )

        ckpt = torch.load(
            self.weights_path, map_location="cpu", weights_only=False
        )

        self._predictor = self._build_predictor(
            ckpt=ckpt,
            tile_step_size=tile_step_size,
            use_gaussian=use_gaussian,
            use_mirroring=use_mirroring,
            perform_everything_on_device=perform_everything_on_device,
            verbose=verbose,
        )

    def _build_predictor(
        self,
        ckpt: dict,
        tile_step_size: float,
        use_gaussian: bool,
        use_mirroring: bool,
        perform_everything_on_device: bool,
        verbose: bool,
    ):
        """Wire up nnUNetPredictor from the checkpoint's stored init_args."""
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
        from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

        init_args = ckpt["init_args"]
        plans = init_args["plans"]
        dataset_json = init_args["dataset_json"]
        trainer_name = ckpt.get("trainer_name", "nnUNetTrainer")
        mirroring_axes = ckpt.get("inference_allowed_mirroring_axes", None)

        plans_manager = PlansManager(plans)
        configuration_manager = plans_manager.get_configuration("3d_fullres")

        network = get_network_from_plans(
            arch_class_name=configuration_manager.network_arch_class_name,
            arch_kwargs=configuration_manager.network_arch_init_kwargs,
            arch_kwargs_req_import=configuration_manager.network_arch_init_kwargs_req_import,
            input_channels=1,
            output_channels=len(dataset_json["labels"]),
            allow_init=True,
            deep_supervision=False,     # single output at inference time
        )
        network.eval()

        raw_weights = ckpt["network_weights"]
        weights = {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in raw_weights.items()
        }

        predictor = nnUNetPredictor(
            tile_step_size=tile_step_size,
            use_gaussian=use_gaussian,
            use_mirroring=use_mirroring,
            perform_everything_on_device=perform_everything_on_device,
            device=self.device,
            verbose=verbose,
            verbose_preprocessing=verbose,
        )
        predictor.manual_initialization(
            network=network,
            plans_manager=plans_manager,
            configuration_manager=configuration_manager,
            parameters=[weights],
            dataset_json=dataset_json,
            trainer_name=trainer_name,
            inference_allowed_mirroring_axes=mirroring_axes,
        )

        logger.info(
            "nnUNetPredictor ready — patch_size=%s, spacing=%s mm",
            configuration_manager.patch_size,
            configuration_manager.spacing,
        )
        return predictor

    def segment(
        self,
        ct_path: Union[str, Path, sitk.Image],
    ) -> Tuple[np.ndarray, Tuple[float, float, float], np.ndarray]:
        """Segment a CT volume with the PanTS model.

        Returns:
            `(seg_array, spacing_zyx, affine)`: int32 label map `(Z, Y, X)`, voxel spacing
            in mm, and the 4x4 RAS affine.
        """
        if isinstance(ct_path, sitk.Image):
            sitk_img = ct_path
        else:
            sitk_img = sitk.ReadImage(str(ct_path))

        ct_array = sitk.GetArrayFromImage(sitk_img).astype(np.float32)  # (Z, Y, X)

        # GetSpacing returns (sx, sy, sz); nnunetv2 expects (sz, sy, sx).
        sx, sy, sz = sitk_img.GetSpacing()
        spacing_zyx = (float(sz), float(sy), float(sx))

        affine = _sitk_to_affine(sitk_img)

        image_array = ct_array[np.newaxis]  # (1, Z, Y, X)

        seg = self._predictor.predict_single_npy_array(
            input_image=image_array,
            image_properties={"spacing": list(spacing_zyx)},
            segmentation_previous_stage=None,
            output_file_truncated=None,
            save_or_return_probabilities=False,
        )
        logger.info(
            "Segmentation complete: shape=%s, unique labels=%s",
            seg.shape,
            sorted(np.unique(seg).tolist()),
        )
        return seg.astype(np.int32), spacing_zyx, affine

    def release(self) -> None:
        """Move the network off GPU and drop predictor references."""
        try:
            net = getattr(self._predictor, "network", None)
            if net is not None:
                net.to("cpu")
        except Exception as exc:
            logger.warning("Segmenter release: network.to('cpu') failed: %s", exc)
        self._predictor = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("PanTSSegmenter released")

    def extract_masks(
        self,
        seg_array: np.ndarray,
        include_biliary: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Extract binary masks from a label map.

        Includes `tumor`, `vessel_union`, each vessel, and (if `include_biliary`) the
        pancreatic-duct/common-bile-duct masks.
        """
        masks: Dict[str, np.ndarray] = {}

        masks["tumor"] = seg_array == PANTS_LABELS["tumor"]

        for name, label_id in VESSEL_LABELS.items():
            masks[name] = seg_array == label_id

        vessel_union = np.zeros_like(seg_array, dtype=bool)
        for name in VESSEL_LABELS:
            vessel_union |= masks[name]
        masks["vessel_union"] = vessel_union

        if include_biliary:
            for name, label_id in BILIARY_LABELS.items():
                masks[name] = seg_array == label_id

        for name, arr in masks.items():
            logger.debug("  %-35s  %7d voxels", name, arr.sum())

        return masks


def _sitk_to_affine(sitk_img: sitk.Image) -> np.ndarray:
    """Build a 4×4 NIfTI-style RAS affine from a SimpleITK image (converts LPS->RAS)."""
    origin = np.array(sitk_img.GetOrigin())
    spacing = np.array(sitk_img.GetSpacing())
    direction = np.array(sitk_img.GetDirection()).reshape(3, 3)

    affine = np.eye(4)
    affine[:3, :3] = direction * spacing[np.newaxis, :]
    affine[:3, 3] = origin

    # Flip x and y to convert LPS (SimpleITK) -> RAS (nibabel).
    lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
    return lps_to_ras @ affine
