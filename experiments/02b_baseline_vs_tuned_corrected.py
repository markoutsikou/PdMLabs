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

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CSV_PATH = RESULTS_DIR / "02b_baseline_vs_tuned_corrected.csv"
SUMMARY_PATH = RESULTS_DIR / "02b_baseline_vs_tuned_corrected_summary.txt"

SEED = 42


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

_, test_data = dataset_obj.get_unsupervised_dataset()

print(
    "Generalization source(s):",
    test_data["target_sources"]
)


# ==================================================
# Fixed evaluator
# ==================================================

def evaluate(name, params):

    print("\n" + "=" * 65)
    print(name)
    print("=" * 65)

    start = time.perf_counter()

    result = run_experiment(
        dataset=test_data,

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
            name
        ],

        # Required by the current PdMLabs HPO wrapper
        # even though there is only one fixed config.
        MAX_RUNS=2,
        MAX_JOBS=1,
        INITIAL_RANDOM=1,

        optimization_param="AD1_AUC",
        maximize=True,

        optimizer="mango",

        random_state=SEED,
        deterministic=True,

        use_cache=False,
        log_best_pipeline=False,
    )

    runtime = time.perf_counter() - start

    best = result[0]

    return {
        "ad1_auc": float(
            best["best_objective"]
        ),
        "pipeline_ad1_auc": float(
            best["best_pipeline_objective"]
        ),
        "threshold": float(
            best["th"]
        ),
        "runtime_seconds": runtime,
    }


# ==================================================
# Baseline
# ==================================================
#
# IMPORTANT:
# max_features omitted -> sklearn default 1.0
# policy omitted -> "or"
# bootstrap omitted -> False
# ==================================================

baseline_params = {
    "window": [40],
    "slide": [0.5],

    "n_estimators": [10],
    "max_samples": ["auto"],

    "random_state": [SEED],
}


# ==================================================
# Corrected tuned configuration from E01b
# ==================================================
#
# Again, max_features is intentionally omitted.
# ==================================================

tuned_params = {
    "window": [160],
    "slide": [1.0],

    "n_estimators": [100],
    "max_samples": [40],

    "random_state": [SEED],
}


# ==================================================
# Evaluate
# ==================================================

baseline = evaluate(
    "E02b_BASELINE",
    baseline_params,
)

tuned = evaluate(
    "E02b_TUNED",
    tuned_params,
)


# ==================================================
# Results table
# ==================================================

results_df = pd.DataFrame([
    {
        "configuration": "baseline",
        "seed": SEED,
        "window": 40,
        "slide": 0.5,
        "n_estimators": 10,
        "max_samples": "auto",
        "effective_max_features": 1.0,
        "ad1_auc": baseline["ad1_auc"],
        "pipeline_ad1_auc": baseline["pipeline_ad1_auc"],
        "threshold": baseline["threshold"],
        "runtime_seconds": baseline["runtime_seconds"],
    },
    {
        "configuration": "tuned",
        "seed": SEED,
        "window": 160,
        "slide": 1.0,
        "n_estimators": 100,
        "max_samples": 40,
        "effective_max_features": 1.0,
        "ad1_auc": tuned["ad1_auc"],
        "pipeline_ad1_auc": tuned["pipeline_ad1_auc"],
        "threshold": tuned["threshold"],
        "runtime_seconds": tuned["runtime_seconds"],
    },
])

results_df.to_csv(
    CSV_PATH,
    index=False,
)


# ==================================================
# Comparison
# ==================================================

absolute_improvement = (
    tuned["ad1_auc"]
    - baseline["ad1_auc"]
)

if baseline["ad1_auc"] != 0:

    ratio = (
        tuned["ad1_auc"]
        / baseline["ad1_auc"]
    )

else:

    ratio = float("inf")


print("\n")
print("=" * 65)
print("E02b - CORRECTED BASELINE VS TUNED")
print("=" * 65)

print(
    results_df.to_string(
        index=False
    )
)

print(
    "\nAbsolute AD1_AUC improvement:",
    absolute_improvement,
)

print(
    "Tuned / Baseline ratio:",
    ratio,
)


# ==================================================
# Save summary
# ==================================================

with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "Experiment: "
        "E02b - Corrected Baseline vs Tuned\n"
    )

    f.write(
        "Dataset: IMS\n"
    )

    f.write(
        "Generalization source: 3\n"
    )

    f.write(
        "Metric: AD1_AUC\n"
    )

    f.write(
        f"Seed: {SEED}\n\n"
    )

    f.write(
        "IMPORTANT:\n"
    )

    f.write(
        "max_features is omitted from both "
        "configurations, so the sklearn "
        "default max_features=1.0 is used.\n\n"
    )

    f.write(
        results_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\nAbsolute improvement: "
        f"{absolute_improvement}\n"
    )

    f.write(
        "Tuned / Baseline ratio: "
        f"{ratio}\n"
    )


print("\nSaved:")
print(CSV_PATH)
print(SUMMARY_PATH)