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


# ==================================================
# 1. Paths
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "ims.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(exist_ok=True)

CSV_PATH = RESULTS_DIR / "03_seed_robustness.csv"
SUMMARY_PATH = RESULTS_DIR / "03_seed_robustness_summary.txt"


# ==================================================
# 2. Load IMS
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

optimization_data, test_data = dataset_obj.get_unsupervised_dataset()

print("Held-out test source(s):")
print(test_data["target_sources"])


# ==================================================
# 3. Seeds
# ==================================================

SEEDS = [
    1,
    10,
    42,
    100,
    2026,
]


# ==================================================
# 4. Fixed configuration evaluator
# ==================================================

def evaluate_configuration(
    experiment_name,
    params,
    seed,
):

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
            experiment_name
        ],

        # PdMLabs currently requires at least
        # this budget even for a fixed config.
        MAX_RUNS=2,
        MAX_JOBS=1,
        INITIAL_RANDOM=1,

        optimization_param="AD1_AUC",
        maximize=True,

        optimizer="mango",

        random_state=seed,
        deterministic=True,

        log_best_pipeline=False,
    )

    runtime = time.perf_counter() - start

    # run_experiment returns a list.
    first_result = result[0]

    score = float(
        first_result["best_objective"]
    )

    threshold = float(
        first_result["th"]
    )

    return {
        "ad1_auc": score,
        "threshold": threshold,
        "runtime_seconds": runtime,
    }


# ==================================================
# 5. Run experiment
# ==================================================

records = []

print("\n==========================================")
print("E03 - SEED ROBUSTNESS")
print("==========================================")

for seed in SEEDS:

    print("\n")
    print("=" * 60)
    print(f"SEED = {seed}")
    print("=" * 60)


    # ----------------------------------------------
    # Baseline
    # ----------------------------------------------

    baseline_params = {
        "window": [40],
        "slide": [0.5],
        "policy": ["or"],

        "n_estimators": [10],
        "max_samples": ["auto"],
        "max_features": [1.0],
        "bootstrap": [False],

        "random_state": [seed],
    }

    print("\nRunning BASELINE...")

    baseline = evaluate_configuration(
        experiment_name=f"E03_BASELINE_seed_{seed}",
        params=baseline_params,
        seed=seed,
    )

    print(
        f"Baseline AD1_AUC: "
        f"{baseline['ad1_auc']:.6f}"
    )

    print(
        f"Baseline runtime: "
        f"{baseline['runtime_seconds']:.2f} s"
    )


    # ----------------------------------------------
    # Tuned
    # ----------------------------------------------

    tuned_params = {
        "window": [160],
        "slide": [1.0],
        "policy": ["or"],

        "n_estimators": [100],
        "max_samples": [40],
        "max_features": [1.0],
        "bootstrap": [False],

        "random_state": [seed],
    }

    print("\nRunning TUNED...")

    tuned = evaluate_configuration(
        experiment_name=f"E03_TUNED_seed_{seed}",
        params=tuned_params,
        seed=seed,
    )

    print(
        f"Tuned AD1_AUC: "
        f"{tuned['ad1_auc']:.6f}"
    )

    print(
        f"Tuned runtime: "
        f"{tuned['runtime_seconds']:.2f} s"
    )


    # ----------------------------------------------
    # Store both
    # ----------------------------------------------

    records.append({
        "seed": seed,
        "configuration": "baseline",
        "window": 40,
        "slide": 0.5,
        "n_estimators": 10,
        "max_samples": "auto",
        "ad1_auc": baseline["ad1_auc"],
        "threshold": baseline["threshold"],
        "runtime_seconds": baseline[
            "runtime_seconds"
        ],
    })

    records.append({
        "seed": seed,
        "configuration": "tuned",
        "window": 160,
        "slide": 1.0,
        "n_estimators": 100,
        "max_samples": 40,
        "ad1_auc": tuned["ad1_auc"],
        "threshold": tuned["threshold"],
        "runtime_seconds": tuned[
            "runtime_seconds"
        ],
    })


# ==================================================
# 6. Create DataFrame
# ==================================================

results_df = pd.DataFrame(records)

results_df.to_csv(
    CSV_PATH,
    index=False,
)


# ==================================================
# 7. Summary statistics
# ==================================================

summary = (
    results_df
    .groupby("configuration")
    .agg(
        mean_ad1_auc=(
            "ad1_auc",
            "mean",
        ),
        std_ad1_auc=(
            "ad1_auc",
            "std",
        ),
        min_ad1_auc=(
            "ad1_auc",
            "min",
        ),
        max_ad1_auc=(
            "ad1_auc",
            "max",
        ),
        mean_runtime_seconds=(
            "runtime_seconds",
            "mean",
        ),
    )
)


# Pivot so baseline and tuned can be
# compared seed by seed.

comparison = results_df.pivot(
    index="seed",
    columns="configuration",
    values="ad1_auc",
)

comparison["improvement"] = (
    comparison["tuned"]
    - comparison["baseline"]
)

comparison["tuned_wins"] = (
    comparison["tuned"]
    > comparison["baseline"]
)

number_of_wins = int(
    comparison["tuned_wins"].sum()
)


# ==================================================
# 8. Print results
# ==================================================

print("\n")
print("=" * 60)
print("RESULTS BY SEED")
print("=" * 60)

print(
    results_df[
        [
            "seed",
            "configuration",
            "ad1_auc",
            "runtime_seconds",
        ]
    ].to_string(index=False)
)


print("\n")
print("=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)

print(summary)


print("\n")
print("=" * 60)
print("BASELINE VS TUNED BY SEED")
print("=" * 60)

print(comparison)


print(
    f"\nTuned configuration wins: "
    f"{number_of_wins}/{len(SEEDS)} seeds"
)


# ==================================================
# 9. Save human-readable summary
# ==================================================

with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "Experiment: E03 - Seed Robustness\n"
    )

    f.write(
        "Dataset: IMS\n"
    )

    f.write(
        "Held-out source: 3\n"
    )

    f.write(
        "Metric: AD1_AUC\n"
    )

    f.write(
        f"Seeds: {SEEDS}\n\n"
    )

    f.write(
        "SUMMARY STATISTICS\n"
    )

    f.write(
        summary.to_string()
    )

    f.write(
        "\n\nBASELINE VS TUNED BY SEED\n"
    )

    f.write(
        comparison.to_string()
    )

    f.write(
        "\n\n"
    )

    f.write(
        f"Tuned wins: "
        f"{number_of_wins}/{len(SEEDS)}\n"
    )


print("\n")
print("=" * 60)
print("FILES SAVED")
print("=" * 60)

print(CSV_PATH)
print(SUMMARY_PATH)