from pathlib import Path
import json
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


# ==================================================
# 1. Paths
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "ims.csv"

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CSV_PATH = (
    RESULTS_DIR
    / "01b_if_tuning_corrected.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "01b_if_tuning_corrected_summary.txt"
)


# ==================================================
# 2. Experiment settings
# ==================================================

SEED = 42

# Same optimization budget as original E01
BUDGET = 8


# ==================================================
# 3. Load IMS
# ==================================================

print("\n=== Loading IMS dataset ===")

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

optimization_data, test_data = (
    dataset_obj.get_unsupervised_dataset()
)

print(
    "Optimization sources:",
    optimization_data["target_sources"]
)

print(
    "Held-out source(s):",
    test_data["target_sources"]
)


# ==================================================
# 4. CLEAN search space
# ==================================================

SEARCH_SPACE = {

    # Sliding-window parameters
    "window": [
        40,
        80,
        160,
    ],

    "slide": [
        0.33,
        0.5,
        1.0,
    ],

    # Isolation Forest parameters
    "n_estimators": [
        50,
        100,
        200,
    ],

    "max_samples": [
        20,
        40,
    ],

    # Keep model stochasticity reproducible
    "random_state": [
        SEED,
    ],
}


# IMPORTANT:
#
# These are intentionally NOT included:
#
# max_features
# policy
# bootstrap
#
# Therefore the library defaults apply:
#
# IsolationForest:
# max_features = 1.0
# bootstrap = False
#
# IsolationForestUnsupervised:
# policy = "or"


print("\n=== Corrected search space ===")
print(SEARCH_SPACE)

print(
    "\nNumber of variable configurations:",
    3 * 3 * 3 * 2
)


# ==================================================
# 5. Run corrected Mango tuning
# ==================================================

print("\n")
print("=" * 70)
print("E01b - CORRECTED ISOLATION FOREST TUNING")
print("=" * 70)

print(
    f"Optimizer: Mango"
)

print(
    f"Budget: {BUDGET}"
)

print(
    "Objective: AD1_AUC"
)

print(
    "Important: max_features is omitted "
    "and therefore remains float 1.0."
)


start = time.perf_counter()


result = run_experiment(

    dataset=optimization_data,

    methods=[
        IsolationForestUnsupervised
    ],

    param_space_dict_per_method=[
        SEARCH_SPACE
    ],

    method_names=[
        "IsolationForestUnsupervised"
    ],

    experiments=[
        UnsupervisedPdMExperiment
    ],

    experiment_names=[
        "E01b_IF_Tuning_Corrected"
    ],

    MAX_RUNS=BUDGET,

    MAX_JOBS=1,

    INITIAL_RANDOM=2,

    optimization_param="AD1_AUC",

    maximize=True,

    optimizer="mango",

    random_state=SEED,

    deterministic=True,

    use_cache=False,

    log_best_pipeline=False,
)


runtime = (
    time.perf_counter()
    - start
)


# ==================================================
# 6. Extract result
# ==================================================

best = result[0]

best_params = best[
    "best_params"
]

best_objective = float(
    best["best_objective"]
)

pipeline_objective = float(
    best[
        "best_pipeline_objective"
    ]
)

threshold = float(
    best["th"]
)


# ==================================================
# 7. Sanity checks
# ==================================================

print("\n")
print("=" * 70)
print("SANITY CHECK")
print("=" * 70)

print(
    "Returned best params:"
)

print(
    best_params
)


# These should NOT appear anymore.

for forbidden_param in [
    "method_max_features",
    "method_bootstrap",
    "method_policy",
]:

    if forbidden_param in best_params:

        print(
            f"WARNING: unexpected parameter found: "
            f"{forbidden_param}"
        )

    else:

        print(
            f"PASS: {forbidden_param} "
            f"not present in HPO result."
        )


# ==================================================
# 8. Print result
# ==================================================

print("\n")
print("=" * 70)
print("E01b RESULT")
print("=" * 70)

print(
    f"Best validation AD1_AUC: "
    f"{best_objective:.12f}"
)

print(
    f"Best pipeline AD1_AUC: "
    f"{pipeline_objective:.12f}"
)

print(
    f"Threshold: "
    f"{threshold:.12f}"
)

print(
    f"Runtime: "
    f"{runtime:.2f} seconds"
)

print(
    f"Runtime: "
    f"{runtime / 60:.2f} minutes"
)

print(
    "\nBest hyperparameters:"
)

for key, value in best_params.items():

    print(
        f"  {key} = {value} "
        f"({type(value).__name__})"
    )


# ==================================================
# 9. Save CSV
# ==================================================

row = {

    "experiment":
        "E01b",

    "optimizer":
        "mango",

    "budget":
        BUDGET,

    "seed":
        SEED,

    "best_validation_ad1_auc":
        best_objective,

    "best_pipeline_ad1_auc":
        pipeline_objective,

    "threshold":
        threshold,

    "runtime_seconds":
        runtime,

    "window":
        best_params.get(
            "method_window"
        ),

    "slide":
        best_params.get(
            "method_slide"
        ),

    "n_estimators":
        best_params.get(
            "method_n_estimators"
        ),

    "max_samples":
        best_params.get(
            "method_max_samples"
        ),

    # Explicitly record effective defaults
    "effective_max_features":
        1.0,

    "effective_policy":
        "or",

    "effective_bootstrap":
        False,

    "best_params_json":
        json.dumps(
            best_params,
            default=str,
        ),
}


results_df = pd.DataFrame(
    [row]
)

results_df.to_csv(
    CSV_PATH,
    index=False,
)


# ==================================================
# 10. Save text summary
# ==================================================

with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "Experiment: "
        "E01b - Corrected IF Tuning\n"
    )

    f.write(
        "Dataset: IMS\n"
    )

    f.write(
        "Optimization sources: 1 + 2\n"
    )

    f.write(
        "Method: "
        "IsolationForestUnsupervised\n"
    )

    f.write(
        "Optimizer: Mango\n"
    )

    f.write(
        "Objective: AD1_AUC\n"
    )

    f.write(
        f"Budget: {BUDGET}\n"
    )

    f.write(
        f"Seed: {SEED}\n\n"
    )

    f.write(
        "IMPORTANT CORRECTION\n"
    )

    f.write(
        "max_features was removed from "
        "the HPO search space to avoid "
        "int/float semantic coercion.\n"
    )

    f.write(
        "Effective max_features = 1.0 "
        "(sklearn default).\n"
    )

    f.write(
        "Effective policy = or.\n"
    )

    f.write(
        "Effective bootstrap = False.\n\n"
    )

    f.write(
        "SEARCH SPACE\n"
    )

    f.write(
        json.dumps(
            SEARCH_SPACE,
            indent=2,
        )
    )

    f.write(
        "\n\nRESULT\n"
    )

    f.write(
        f"Best validation AD1_AUC: "
        f"{best_objective}\n"
    )

    f.write(
        f"Best pipeline AD1_AUC: "
        f"{pipeline_objective}\n"
    )

    f.write(
        f"Threshold: "
        f"{threshold}\n"
    )

    f.write(
        f"Runtime_seconds: "
        f"{runtime}\n\n"
    )

    f.write(
        "Best parameters:\n"
    )

    f.write(
        json.dumps(
            best_params,
            indent=2,
            default=str,
        )
    )


# ==================================================
# 11. Done
# ==================================================

print("\n")
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print(CSV_PATH)
print(SUMMARY_PATH)