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

CSV_PATH = RESULTS_DIR / "04b_budget_generalization.csv"
SUMMARY_PATH = RESULTS_DIR / "04b_budget_generalization_summary.txt"


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

print("Generalization source(s):")
print(test_data["target_sources"])


# ==================================================
# 3. Random seeds
# ==================================================

SEEDS = [
    1,
    10,
    42,
    100,
    2026,
]


# ==================================================
# 4. Configurations
# ==================================================

def baseline_params(seed):
    return {
        "window": [40],
        "slide": [0.5],
        "policy": ["or"],

        "n_estimators": [10],
        "max_samples": ["auto"],
        "max_features": [1.0],
        "bootstrap": [False],

        "random_state": [seed],
    }


def budget8_params(seed):
    return {
        "window": [160],
        "slide": [1.0],
        "policy": ["or"],

        "n_estimators": [100],
        "max_samples": [40],
        "max_features": [1.0],
        "bootstrap": [False],

        "random_state": [seed],
    }


def budget32_params(seed):
    return {
        "window": [160],
        "slide": [1.0],
        "policy": ["or"],

        "n_estimators": [200],
        "max_samples": [20],
        "max_features": [1.0],
        "bootstrap": [False],

        "random_state": [seed],
    }


CONFIGURATIONS = {
    "baseline": baseline_params,
    "budget_8": budget8_params,
    "budget_32": budget32_params,
}


# ==================================================
# 5. Fixed configuration evaluator
# ==================================================

def evaluate_configuration(
    configuration_name,
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
            f"E04b_{configuration_name}_seed_{seed}"
        ],

        # Required by PdMLabs/Mango even though
        # the configuration itself is fixed.
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

    result = result[0]

    return {
        "ad1_auc": float(result["best_objective"]),
        "threshold": float(result["th"]),
        "runtime_seconds": runtime,
    }


# ==================================================
# 6. Run all configurations × seeds
# ==================================================

records = []

print("\n" + "=" * 65)
print("E04b - BUDGET GENERALIZATION CHECK")
print("=" * 65)

for seed in SEEDS:

    print("\n")
    print("=" * 65)
    print(f"SEED = {seed}")
    print("=" * 65)

    for config_name, param_function in CONFIGURATIONS.items():

        print(f"\nRunning {config_name.upper()}...")

        params = param_function(seed)

        result = evaluate_configuration(
            configuration_name=config_name,
            params=params,
            seed=seed,
        )

        print(
            f"{config_name} AD1_AUC: "
            f"{result['ad1_auc']:.6f}"
        )

        print(
            f"{config_name} runtime: "
            f"{result['runtime_seconds']:.2f} sec"
        )

        records.append({
            "seed": seed,
            "configuration": config_name,

            "window": params["window"][0],
            "slide": params["slide"][0],
            "n_estimators": params["n_estimators"][0],
            "max_samples": params["max_samples"][0],

            "ad1_auc": result["ad1_auc"],
            "threshold": result["threshold"],
            "runtime_seconds": result["runtime_seconds"],
        })


# ==================================================
# 7. Save raw results
# ==================================================

results_df = pd.DataFrame(records)

results_df.to_csv(
    CSV_PATH,
    index=False,
)


# ==================================================
# 8. Summary statistics
# ==================================================

summary = (
    results_df
    .groupby("configuration")
    .agg(
        mean_ad1_auc=("ad1_auc", "mean"),
        std_ad1_auc=("ad1_auc", "std"),
        min_ad1_auc=("ad1_auc", "min"),
        max_ad1_auc=("ad1_auc", "max"),
        mean_runtime_seconds=("runtime_seconds", "mean"),
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

comparison["budget32_minus_budget8"] = (
    comparison["budget_32"]
    - comparison["budget_8"]
)

comparison["budget8_minus_baseline"] = (
    comparison["budget_8"]
    - comparison["baseline"]
)

comparison["budget32_minus_baseline"] = (
    comparison["budget_32"]
    - comparison["baseline"]
)

comparison["budget32_beats_budget8"] = (
    comparison["budget_32"]
    > comparison["budget_8"]
)

budget32_wins = int(
    comparison["budget32_beats_budget8"].sum()
)


# ==================================================
# 10. Print results
# ==================================================

print("\n")
print("=" * 65)
print("RESULTS BY SEED")
print("=" * 65)

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
print("=" * 65)
print("SUMMARY STATISTICS")
print("=" * 65)

print(summary)


print("\n")
print("=" * 65)
print("BUDGET 8 VS BUDGET 32")
print("=" * 65)

print(comparison)


print(
    f"\nBudget-32 beats Budget-8: "
    f"{budget32_wins}/{len(SEEDS)} seeds"
)


# ==================================================
# 11. Save human-readable summary
# ==================================================

with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "Experiment: E04b - Budget Generalization Check\n"
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
        "Configurations:\n"
    )

    f.write(
        "Baseline: "
        "window=40, slide=0.5, "
        "n_estimators=10, max_samples=auto\n"
    )

    f.write(
        "Budget-8: "
        "window=160, slide=1.0, "
        "n_estimators=100, max_samples=40\n"
    )

    f.write(
        "Budget-32: "
        "window=160, slide=1.0, "
        "n_estimators=200, max_samples=20\n\n"
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
        "\n\n"
    )

    f.write(
        f"Budget-32 beats Budget-8: "
        f"{budget32_wins}/{len(SEEDS)} seeds\n"
    )


print("\n")
print("=" * 65)
print("FILES SAVED")
print("=" * 65)

print(CSV_PATH)
print(SUMMARY_PATH)