from pathlib import Path
import time
import pandas as pd

from pdmlabs.utils.dataset import Dataset
from pdmlabs.RunExperiment import run_experiment
from pdmlabs.experiment.batch.unsupervised_experiment import (
    UnsupervisedPdMExperiment,
)
from pdmlabs.method.isolation_forest_uns import (
    IsolationForestUnsupervised,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "ims.csv"


# --------------------------------------------------
# Dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

dataset_obj = Dataset(
    data=df,
    datetime_column="Artificial_timestamp",
    event_indicator="event",
    source_column="source",
    train_sources=["1"],
    val_sources=["2"],
    test_sources=["3"],
)

optimization_data, test_data = dataset_obj.get_unsupervised_dataset()


def evaluate_configuration(name, dataset, params):
    print("\n")
    print("=" * 60)
    print(name)
    print("=" * 60)

    start = time.perf_counter()

    result = run_experiment(
        dataset=dataset,

        methods=[IsolationForestUnsupervised],

        param_space_dict_per_method=[
            params
        ],

        method_names=[
            "IsolationForestUnsupervised"
        ],

        experiments=[
            UnsupervisedPdMExperiment
        ],

        experiment_names=[
            name
        ],

        # Only one fixed configuration
        MAX_RUNS=2,
        MAX_JOBS=1,
        INITIAL_RANDOM=1,

        optimization_param="AD1_AUC",
        maximize=True,

        optimizer="mango",

        random_state=42,
        deterministic=True,

        # No need to save model artifact here
        log_best_pipeline=False,
    )

    elapsed = time.perf_counter() - start

    print("\nResult:")
    print(result)

    print(f"\nRuntime: {elapsed:.2f} seconds")

    return result


# --------------------------------------------------
# Baseline configuration
# --------------------------------------------------

baseline_params = {
    "window": [40],
    "slide": [0.5],
    "policy": ["or"],

    "n_estimators": [10],
    "max_samples": ["auto"],
    "max_features": [1.0],
    "bootstrap": [False],

    "random_state": [42],
}


# --------------------------------------------------
# Tuned configuration from Experiment 01
# --------------------------------------------------

tuned_params = {
    "window": [160],
    "slide": [1.0],
    "policy": ["or"],

    "n_estimators": [100],
    "max_samples": [40],
    "max_features": [1.0],
    "bootstrap": [False],

    "random_state": [42],
}


# --------------------------------------------------
# Evaluate on held-out test source
# --------------------------------------------------

print("\nTesting on held-out source(s):")
print(test_data["target_sources"])


baseline_result = evaluate_configuration(
    "IMS_IF_BASELINE_TEST",
    test_data,
    baseline_params,
)


tuned_result = evaluate_configuration(
    "IMS_IF_TUNED_TEST",
    test_data,
    tuned_params,
)


print("\n")
print("=" * 60)
print("FINAL COMPARISON")
print("=" * 60)

print("\nBaseline:")
print(baseline_result)

print("\nTuned:")
print(tuned_result)