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
    / "04d_budget_generalization_corrected.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "04d_budget_generalization_corrected_summary.txt"
)


# ==================================================
# 2. Dataset
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

_, test_data = dataset_obj.get_unsupervised_dataset()

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
# 4. Fixed evaluator
# ==================================================

def evaluate_configuration(
    experiment_name,
    params,
    seed,
):

    print("\n")
    print("-" * 65)
    print(experiment_name)
    print("-" * 65)

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

        MAX_RUNS=2,
        MAX_JOBS=1,
        INITIAL_RANDOM=1,

        optimization_param="AD1_AUC",
        maximize=True,

        optimizer="mango",

        random_state=seed,
        deterministic=True,

        use_cache=False,
        log_best_pipeline=False,
    )

    runtime = time.perf_counter() - start

    best = result[0]

    objective_value = best.get(
        "best_objective"
    )

    if objective_value is None:
        raise RuntimeError(
            "best_objective is None."
        )

    ad1_auc = float(
        objective_value
    )

    pipeline_value = best.get(
        "best_pipeline_objective"
    )

    pipeline_ad1_auc = (
        None
        if pipeline_value is None
        else float(pipeline_value)
    )

    threshold_value = best.get(
        "th"
    )

    threshold = (
        None
        if threshold_value is None
        else float(threshold_value)
    )

    print(
        "AD1_AUC:",
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
# 5. Configurations
# ==================================================
#
# IMPORTANT:
# max_features omitted -> sklearn default 1.0
# policy omitted -> "or"
# bootstrap omitted -> False
# ==================================================

def budget4_params(seed):

    return {
        "window": [160],
        "slide": [0.5],

        "n_estimators": [100],
        "max_samples": [40],

        "random_state": [seed],
    }


def budget8_params(seed):

    return {
        "window": [160],
        "slide": [1.0],

        "n_estimators": [100],
        "max_samples": [40],

        "random_state": [seed],
    }


CONFIGURATIONS = {
    "budget_4": budget4_params,
    "budget_8": budget8_params,
}


# ==================================================
# 6. Run all seeds
# ==================================================

records = []

print("\n")
print("=" * 70)
print("E04d - BUDGET 4 VS BUDGET 8 GENERALIZATION")
print("=" * 70)

for seed in SEEDS:

    print("\n")
    print("=" * 70)
    print(f"SEED = {seed}")
    print("=" * 70)

    for config_name, param_function in CONFIGURATIONS.items():

        params = param_function(seed)

        result = evaluate_configuration(
            experiment_name=(
                f"E04d_{config_name}_seed_{seed}"
            ),
            params=params,
            seed=seed,
        )

        records.append({
            "seed":
                seed,

            "configuration":
                config_name,

            "window":
                params["window"][0],

            "slide":
                params["slide"][0],

            "n_estimators":
                params["n_estimators"][0],

            "max_samples":
                params["max_samples"][0],

            "effective_max_features":
                1.0,

            "effective_policy":
                "or",

            "effective_bootstrap":
                False,

            "ad1_auc":
                result["ad1_auc"],

            "pipeline_ad1_auc":
                result[
                    "pipeline_ad1_auc"
                ],

            "threshold":
                result["threshold"],

            "runtime_seconds":
                result["runtime_seconds"],
        })

    # Save after every seed
    pd.DataFrame(records).to_csv(
        CSV_PATH,
        index=False,
    )

    print(
        "\nIntermediate results saved."
    )


# ==================================================
# 7. Raw results
# ==================================================

results_df = pd.DataFrame(
    records
)

results_df.to_csv(
    CSV_PATH,
    index=False,
)


# ==================================================
# 8. Summary statistics
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
# 9. Seed-by-seed comparison
# ==================================================

comparison = results_df.pivot(
    index="seed",
    columns="configuration",
    values="ad1_auc",
)

comparison[
    "budget8_minus_budget4"
] = (
    comparison["budget_8"]
    - comparison["budget_4"]
)

comparison[
    "budget8_over_budget4"
] = (
    comparison["budget_8"]
    / comparison["budget_4"]
)

comparison[
    "budget8_wins"
] = (
    comparison["budget_8"]
    > comparison["budget_4"]
)

budget8_wins = int(
    comparison[
        "budget8_wins"
    ].sum()
)


# ==================================================
# 10. Overall means
# ==================================================

budget4_mean = (
    results_df[
        results_df[
            "configuration"
        ] == "budget_4"
    ][
        "ad1_auc"
    ].mean()
)

budget8_mean = (
    results_df[
        results_df[
            "configuration"
        ] == "budget_8"
    ][
        "ad1_auc"
    ].mean()
)

absolute_mean_improvement = (
    budget8_mean
    - budget4_mean
)

if budget4_mean != 0:

    mean_ratio = (
        budget8_mean
        / budget4_mean
    )

else:

    mean_ratio = float(
        "inf"
    )


# ==================================================
# 11. Print
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
            "runtime_seconds",
        ]
    ].to_string(
        index=False
    )
)


print("\n")
print("=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

print(
    summary
)


print("\n")
print("=" * 70)
print("BUDGET 4 VS BUDGET 8")
print("=" * 70)

print(
    comparison
)


print("\n")
print("=" * 70)
print("OVERALL RESULT")
print("=" * 70)

print(
    "Budget 4 mean AD1_AUC:",
    budget4_mean
)

print(
    "Budget 8 mean AD1_AUC:",
    budget8_mean
)

print(
    "Absolute mean improvement:",
    absolute_mean_improvement
)

print(
    "Budget 8 / Budget 4 mean ratio:",
    mean_ratio
)

print(
    f"Budget 8 wins: "
    f"{budget8_wins}/{len(SEEDS)}"
)


# ==================================================
# 12. Save summary
# ==================================================

with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "Experiment: "
        "E04d - Budget 4 vs Budget 8 Generalization\n"
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
        "max_features is omitted from all "
        "parameter dictionaries.\n"
    )

    f.write(
        "Effective max_features = 1.0.\n"
    )

    f.write(
        "Effective policy = or.\n"
    )

    f.write(
        "Effective bootstrap = False.\n\n"
    )


    f.write(
        "BUDGET 4 WINNER\n"
    )

    f.write(
        "window=160\n"
        "slide=0.5\n"
        "n_estimators=100\n"
        "max_samples=40\n\n"
    )


    f.write(
        "BUDGET 8 WINNER\n"
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
        "\n\nOVERALL RESULT\n"
    )

    f.write(
        f"Budget 4 mean AD1_AUC: "
        f"{budget4_mean}\n"
    )

    f.write(
        f"Budget 8 mean AD1_AUC: "
        f"{budget8_mean}\n"
    )

    f.write(
        f"Absolute mean improvement: "
        f"{absolute_mean_improvement}\n"
    )

    f.write(
        f"Budget 8 / Budget 4 mean ratio: "
        f"{mean_ratio}\n"
    )

    f.write(
        f"Budget 8 wins: "
        f"{budget8_wins}/{len(SEEDS)}\n"
    )


# ==================================================
# 13. Done
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