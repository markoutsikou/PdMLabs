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

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "ims.csv"

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CSV_PATH = (
    RESULTS_DIR
    / "03b_seed_robustness_corrected.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "03b_seed_robustness_corrected_summary.txt"
)


# ==================================================
# 2. Load IMS dataset
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

_, test_data = (
    dataset_obj.get_unsupervised_dataset()
)

print(
    "Generalization source(s):",
    test_data["target_sources"]
)


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
    name,
    params,
    seed,
):

    print("\n")
    print("-" * 60)
    print(name)
    print("-" * 60)

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

        # Current PdMLabs/Mango wrapper
        # requires this minimum budget
        # even for one fixed configuration.
        MAX_RUNS=2,
        MAX_JOBS=1,
        INITIAL_RANDOM=1,

        optimization_param="AD1_AUC",
        maximize=True,

        optimizer="mango",

        random_state=seed,
        deterministic=True,

        use_cache=False,

        # We only need the evaluation result.
        log_best_pipeline=False,
    )

    runtime = (
        time.perf_counter()
        - start
    )

    best = result[0]


    # ==================================================
    # Main metric
    # ==================================================
    #
    # This is the important value for E03b.
    #
    # Even when TrialSink fails to recover pipeline
    # metadata, best_objective can still be valid.
    # ==================================================

    objective_value = best.get(
        "best_objective"
    )

    if objective_value is None:

        raise RuntimeError(
            "best_objective is None. "
            "The actual evaluation failed."
        )

    ad1_auc = float(
        objective_value
    )


    # ==================================================
    # Pipeline objective
    # ==================================================
    #
    # This can occasionally be None if TrialSink did
    # not recover the materialized trial.
    # That should NOT terminate E03b.
    # ==================================================

    pipeline_value = best.get(
        "best_pipeline_objective"
    )

    if pipeline_value is None:

        pipeline_ad1_auc = None

        print(
            "\nWARNING:"
        )

        print(
            "best_pipeline_objective "
            "was not recovered."
        )

        print(
            "The optimizer best_objective "
            "will still be used:"
        )

        print(
            ad1_auc
        )

    else:

        pipeline_ad1_auc = float(
            pipeline_value
        )


    # ==================================================
    # Threshold
    # ==================================================
    #
    # Threshold may also be None when TrialSink
    # recovery fails.
    # It is not needed for our AD1_AUC comparison.
    # ==================================================

    threshold_value = best.get(
        "th"
    )

    if threshold_value is None:

        threshold = None

        print(
            "WARNING:"
        )

        print(
            "Threshold was not recovered "
            "for this run."
        )

    else:

        threshold = float(
            threshold_value
        )


    print(
        "\nAD1_AUC:",
        ad1_auc
    )

    print(
        "Pipeline AD1_AUC:",
        pipeline_ad1_auc
    )

    print(
        "Threshold:",
        threshold
    )

    print(
        "Runtime:",
        round(runtime, 2),
        "seconds"
    )


    return {
        "ad1_auc":
            ad1_auc,

        "pipeline_ad1_auc":
            pipeline_ad1_auc,

        "threshold":
            threshold,

        "runtime_seconds":
            runtime,
    }


# ==================================================
# 5. Run all seeds
# ==================================================

records = []


print("\n")
print("=" * 70)
print("E03b - CORRECTED SEED ROBUSTNESS")
print("=" * 70)

print(
    "\nIMPORTANT:"
)

print(
    "max_features is omitted everywhere."
)

print(
    "Effective sklearn max_features = 1.0"
)


for seed in SEEDS:

    print("\n")
    print("=" * 70)
    print(f"SEED = {seed}")
    print("=" * 70)


    # ==================================================
    # BASELINE
    # ==================================================
    #
    # max_features omitted -> 1.0
    # policy omitted -> "or"
    # bootstrap omitted -> False
    # ==================================================

    baseline_params = {

        "window": [
            40
        ],

        "slide": [
            0.5
        ],

        "n_estimators": [
            10
        ],

        "max_samples": [
            "auto"
        ],

        "random_state": [
            seed
        ],
    }


    print(
        "\n>>> Running BASELINE"
    )

    baseline = evaluate_configuration(

        name=(
            f"E03b_BASELINE_seed_{seed}"
        ),

        params=baseline_params,

        seed=seed,
    )


    # ==================================================
    # TUNED
    # ==================================================
    #
    # Corrected configuration from E01b:
    #
    # window       = 160
    # slide        = 1.0
    # n_estimators = 100
    # max_samples  = 40
    #
    # max_features omitted -> 1.0
    # ==================================================

    tuned_params = {

        "window": [
            160
        ],

        "slide": [
            1.0
        ],

        "n_estimators": [
            100
        ],

        "max_samples": [
            40
        ],

        "random_state": [
            seed
        ],
    }


    print(
        "\n>>> Running TUNED"
    )

    tuned = evaluate_configuration(

        name=(
            f"E03b_TUNED_seed_{seed}"
        ),

        params=tuned_params,

        seed=seed,
    )


    # ==================================================
    # 6. Store baseline result
    # ==================================================

    records.append({

        "seed":
            seed,

        "configuration":
            "baseline",

        "window":
            40,

        "slide":
            0.5,

        "n_estimators":
            10,

        "max_samples":
            "auto",

        "effective_max_features":
            1.0,

        "effective_policy":
            "or",

        "effective_bootstrap":
            False,

        "ad1_auc":
            baseline[
                "ad1_auc"
            ],

        "pipeline_ad1_auc":
            baseline[
                "pipeline_ad1_auc"
            ],

        "threshold":
            baseline[
                "threshold"
            ],

        "runtime_seconds":
            baseline[
                "runtime_seconds"
            ],
    })


    # ==================================================
    # 7. Store tuned result
    # ==================================================

    records.append({

        "seed":
            seed,

        "configuration":
            "tuned",

        "window":
            160,

        "slide":
            1.0,

        "n_estimators":
            100,

        "max_samples":
            40,

        "effective_max_features":
            1.0,

        "effective_policy":
            "or",

        "effective_bootstrap":
            False,

        "ad1_auc":
            tuned[
                "ad1_auc"
            ],

        "pipeline_ad1_auc":
            tuned[
                "pipeline_ad1_auc"
            ],

        "threshold":
            tuned[
                "threshold"
            ],

        "runtime_seconds":
            tuned[
                "runtime_seconds"
            ],
    })


    # ==================================================
    # Save intermediate results after every seed
    # ==================================================
    #
    # This way we do not lose completed runs if
    # something goes wrong later.
    # ==================================================

    intermediate_df = pd.DataFrame(
        records
    )

    intermediate_df.to_csv(
        CSV_PATH,
        index=False,
    )

    print(
        "\nIntermediate results saved."
    )


# ==================================================
# 8. Final raw results
# ==================================================

results_df = pd.DataFrame(
    records
)

results_df.to_csv(
    CSV_PATH,
    index=False,
)


# ==================================================
# 9. Summary statistics
# ==================================================

summary = (
    results_df
    .groupby(
        "configuration"
    )
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

        missing_pipeline_results=(
            "pipeline_ad1_auc",
            lambda x: x.isna().sum(),
        ),

        missing_thresholds=(
            "threshold",
            lambda x: x.isna().sum(),
        ),
    )
)


# ==================================================
# 10. Seed-by-seed comparison
# ==================================================

comparison = results_df.pivot(

    index="seed",

    columns="configuration",

    values="ad1_auc",
)


comparison[
    "improvement"
] = (
    comparison["tuned"]
    - comparison["baseline"]
)


comparison[
    "ratio"
] = (
    comparison["tuned"]
    / comparison["baseline"]
)


comparison[
    "tuned_wins"
] = (
    comparison["tuned"]
    > comparison["baseline"]
)


tuned_wins = int(

    comparison[
        "tuned_wins"
    ].sum()
)


# ==================================================
# 11. Print individual results
# ==================================================

print("\n")
print("=" * 70)
print("RESULTS BY SEED")
print("=" * 70)


print(

    results_df[
        [
            "seed",
            "configuration",
            "ad1_auc",
            "pipeline_ad1_auc",
            "threshold",
            "runtime_seconds",
        ]
    ]
    .to_string(
        index=False
    )
)


# ==================================================
# 12. Print summary
# ==================================================

print("\n")
print("=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

print(
    summary
)


# ==================================================
# 13. Print comparison
# ==================================================

print("\n")
print("=" * 70)
print("BASELINE VS TUNED")
print("=" * 70)

print(
    comparison
)


print("\n")
print(
    f"Tuned wins: "
    f"{tuned_wins}/{len(SEEDS)}"
)


# ==================================================
# 14. Overall relative improvement
# ==================================================

baseline_mean = (
    results_df[
        results_df[
            "configuration"
        ] == "baseline"
    ][
        "ad1_auc"
    ].mean()
)


tuned_mean = (
    results_df[
        results_df[
            "configuration"
        ] == "tuned"
    ][
        "ad1_auc"
    ].mean()
)


absolute_mean_improvement = (
    tuned_mean
    - baseline_mean
)


if baseline_mean != 0:

    mean_ratio = (
        tuned_mean
        / baseline_mean
    )

else:

    mean_ratio = float(
        "inf"
    )


print("\n")
print("=" * 70)
print("OVERALL IMPROVEMENT")
print("=" * 70)

print(
    "Baseline mean AD1_AUC:",
    baseline_mean
)

print(
    "Tuned mean AD1_AUC:",
    tuned_mean
)

print(
    "Absolute mean improvement:",
    absolute_mean_improvement
)

print(
    "Tuned / Baseline mean ratio:",
    mean_ratio
)


# ==================================================
# 15. Save summary text file
# ==================================================

with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "Experiment: "
        "E03b - Corrected Seed Robustness\n"
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
        f"Seeds: {SEEDS}\n\n"
    )


    f.write(
        "IMPORTANT CORRECTION\n"
    )

    f.write(
        "max_features is omitted from "
        "all parameter dictionaries.\n"
    )

    f.write(
        "Therefore sklearn default "
        "max_features=1.0 is used.\n"
    )

    f.write(
        "policy is omitted, therefore "
        "policy='or' is used.\n"
    )

    f.write(
        "bootstrap is omitted, therefore "
        "bootstrap=False is used.\n\n"
    )


    f.write(
        "BASELINE CONFIGURATION\n"
    )

    f.write(
        "window=40\n"
        "slide=0.5\n"
        "n_estimators=10\n"
        "max_samples=auto\n\n"
    )


    f.write(
        "TUNED CONFIGURATION\n"
    )

    f.write(
        "window=160\n"
        "slide=1.0\n"
        "n_estimators=100\n"
        "max_samples=40\n\n"
    )


    f.write(
        "SUMMARY STATISTICS\n"
    )

    f.write(
        summary.to_string()
    )


    f.write(
        "\n\nSEED-BY-SEED COMPARISON\n"
    )

    f.write(
        comparison.to_string()
    )


    f.write(
        "\n\nOVERALL RESULTS\n"
    )

    f.write(
        f"Baseline mean AD1_AUC: "
        f"{baseline_mean}\n"
    )

    f.write(
        f"Tuned mean AD1_AUC: "
        f"{tuned_mean}\n"
    )

    f.write(
        f"Absolute mean improvement: "
        f"{absolute_mean_improvement}\n"
    )

    f.write(
        f"Tuned / Baseline mean ratio: "
        f"{mean_ratio}\n"
    )

    f.write(
        f"Tuned wins: "
        f"{tuned_wins}/{len(SEEDS)}\n"
    )


# ==================================================
# 16. Done
# ==================================================

print("\n")
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print(
    CSV_PATH
)

print(
    SUMMARY_PATH
)