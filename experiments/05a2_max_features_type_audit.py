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


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "ims.csv"


# ==================================================
# Dataset
# ==================================================

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

optimization_data, _ = (
    dataset_obj.get_unsupervised_dataset()
)


# ==================================================
# Evaluator
# ==================================================

def evaluate(label, max_features_value):

    print("\n")
    print("=" * 70)
    print(label)
    print("=" * 70)

    print(
        "max_features value:",
        max_features_value
    )

    print(
        "Python type:",
        type(max_features_value).__name__
    )

    params = {
        "window": [160],
        "slide": [1.0],
        "policy": ["or"],

        "n_estimators": [100],
        "max_samples": [40],

        "max_features": [
            max_features_value
        ],

        "bootstrap": [False],
        "random_state": [42],
    }

    start = time.perf_counter()

    result = run_experiment(
        dataset=optimization_data,

        methods=[
            IsolationForestUnsupervised
        ],

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
            label
        ],

        MAX_RUNS=2,
        MAX_JOBS=1,
        INITIAL_RANDOM=1,

        optimization_param="AD1_AUC",
        maximize=True,

        optimizer="mango",

        random_state=42,
        deterministic=True,

        use_cache=False,
        log_best_pipeline=False,
    )

    runtime = (
        time.perf_counter()
        - start
    )

    best = result[0]

    print("\nReturned parameters:")
    print(best["best_params"])

    print("\nBest objective:")
    print(best["best_objective"])

    print("\nPipeline objective:")
    print(
        best["best_pipeline_objective"]
    )

    print(
        "\nRuntime:",
        runtime,
        "seconds"
    )

    return {
        "label": label,
        "max_features": max_features_value,
        "type": type(
            max_features_value
        ).__name__,
        "objective": float(
            best["best_objective"]
        ),
        "pipeline_objective": float(
            best[
                "best_pipeline_objective"
            ]
        ),
    }


# ==================================================
# A: INTEGER 1
# ==================================================

result_int = evaluate(
    "E05a2_MAX_FEATURES_INT",
    1,
)


# ==================================================
# B: FLOAT 1.0
# ==================================================

result_float = evaluate(
    "E05a2_MAX_FEATURES_FLOAT",
    1.0,
)


# ==================================================
# Comparison
# ==================================================

print("\n")
print("=" * 70)
print("FINAL TYPE COMPARISON")
print("=" * 70)

results = pd.DataFrame(
    [
        result_int,
        result_float,
    ]
)

print(
    results.to_string(
        index=False
    )
)

difference = (
    result_float["objective"]
    - result_int["objective"]
)

print(
    "\nFLOAT - INT objective difference:",
    difference,
)