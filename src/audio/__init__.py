# Audio processing utilities: alignment, masking, crossfade, metrics
from src.audio.alignment import MelRegion, create_error_regions, ms_to_mel_frames
from src.audio.crossfade import crossfade_chunks
from src.audio.masking import apply_mask_to_mel, build_inpainting_mask
from src.audio.metrics import QualityMetrics, compute_secs, convergence_score

__all__ = [
    "MelRegion",
    "QualityMetrics",
    "apply_mask_to_mel",
    "build_inpainting_mask",
    "compute_secs",
    "convergence_score",
    "create_error_regions",
    "crossfade_chunks",
    "ms_to_mel_frames",
]
