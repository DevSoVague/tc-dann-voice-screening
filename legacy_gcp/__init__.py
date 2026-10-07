"""TC-DANN: Task-Conditioned, Domain-Adversarial Network for Bridge2AI-Voice."""

from .model import TCDANN
from .dataset import (
    BridgeVoiceDataset,
    FeatureStore,
    build_subgroup_sampler,
    cross_site_mixup,
    AGE_BUCKETS,
    AGE_BUCKET_NAMES,
    age_to_bucket,
    site_to_code,
    sex_to_code,
)
from .data_prep import (
    DISEASE_FILE_MAP,
    build_master_manifest,
    stratified_participant_split,
    load_diagnosis_matrix,
    load_demographics,
    load_recording_manifest,
    load_static_features,
    diagnose_schemas,
)
from .train import TrainConfig, fit
from .calibration import (
    GroupTemperatureScaler,
    SplitConformalPredictor,
    ensemble_mean_std,
    make_group_id,
)
from .confidence import AbstainConfig, AbstainPredictor
from .evaluate import run_four_criteria, fair_participant_aggregate

__all__ = [
    "TCDANN",
    "BridgeVoiceDataset", "FeatureStore",
    "build_subgroup_sampler", "cross_site_mixup",
    "AGE_BUCKETS", "AGE_BUCKET_NAMES", "age_to_bucket",
    "site_to_code", "sex_to_code",
    "DISEASE_FILE_MAP",
    "build_master_manifest", "stratified_participant_split",
    "load_diagnosis_matrix", "load_demographics",
    "load_recording_manifest", "load_static_features",
    "diagnose_schemas",
    "TrainConfig", "fit",
    "GroupTemperatureScaler", "SplitConformalPredictor",
    "ensemble_mean_std", "make_group_id",
    "AbstainConfig", "AbstainPredictor",
    "run_four_criteria", "fair_participant_aggregate",
]
