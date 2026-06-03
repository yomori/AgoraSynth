"""Amortized non-Gaussian Compton-y synthesis via flow matching with WPH-feature loss."""

from importlib.metadata import PackageNotFoundError, version

from .data import (
    extract_patches,
    gaussianize_patches,
    gaussianized_to_physical,
    project_healpix_to_zea,
    random_sphere_directions,
)
from .flow_matching import (
    FlowMatchingTrainState,
    fm_loss,
    make_fm_only_train_step,
    make_fm_reflow_train_step,
    make_train_step,
    make_train_step_persample,
    make_wph_features_fn,
    sample_euler_one_step,
    sample_heun,
    sample_heun_conditional,
    train_step,
    whitener_from_prior,
    wph_distribution_loss,
    wph_persample_loss,
)
from .unet import UNet
from .wph import (
    DEFAULT_CLASSES as DEFAULT_WPH_CLASSES,
    DEFAULT_SM_P_LIST as DEFAULT_WPH_SM_P_LIST,
    FilterBankConfig as WPHFilterBankConfig,
    MomentTable as WPHMomentTable,
    WPHConfig,
    WPHOp,
    WPHPriorStats,
    build_moment_table,
    compute_S,
    compute_S_batch,
    cosine_window,
    d4_orbit,
    estimate_wph_prior,
    random_patches,
    synthesize_from_prior,
    to_real_features,
)

try:
    __version__ = version("agorasynth")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "DEFAULT_WPH_CLASSES",
    "DEFAULT_WPH_SM_P_LIST",
    "FlowMatchingTrainState",
    "UNet",
    "WPHConfig",
    "WPHFilterBankConfig",
    "WPHMomentTable",
    "WPHOp",
    "WPHPriorStats",
    "__version__",
    "build_moment_table",
    "compute_S",
    "compute_S_batch",
    "cosine_window",
    "d4_orbit",
    "estimate_wph_prior",
    "extract_patches",
    "fm_loss",
    "gaussianize_patches",
    "gaussianized_to_physical",
    "make_fm_only_train_step",
    "make_fm_reflow_train_step",
    "make_train_step",
    "make_train_step_persample",
    "make_wph_features_fn",
    "project_healpix_to_zea",
    "random_patches",
    "random_sphere_directions",
    "sample_euler_one_step",
    "sample_heun",
    "sample_heun_conditional",
    "synthesize_from_prior",
    "to_real_features",
    "train_step",
    "whitener_from_prior",
    "wph_distribution_loss",
    "wph_persample_loss",
]
