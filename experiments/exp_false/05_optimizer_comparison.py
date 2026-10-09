from pathlib import Path
import json
import time
import traceback

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
# 1. Experiment configuration
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "ims.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CSV_PATH = RESULTS_DIR / "05_optimizer_comparison.csv"
SUMMARY_PATH = RESULTS_DIR / "05_optimizer_comparison_summary.txt"

OPTIMIZERS = [
    "mango",
    "mango_random",
    "optuna",
    "smac",
]

BUDGET = 16
SEED = 42


# ==================================================
# 2. Load dataset
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

optimization_data, _ = dataset_obj.get_unsupervised_dataset()

print(
    "Optimization sources:",
    optimization_data["target_sources"]
)


# ==================================================
# 3. Same search space for every optimizer
# ==================================================

SEARCH_SPACE = {
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

    "policy": [
        "or",
    ],

    "n_estimators": [
        50,
        100,
        200,
    ],

    "max_samples": [
        20,
        40,
    ],

    "max_features": [
        1.0,
    ],

    "bootstrap": [
        False,
    ],

    # Keep model stochasticity fixed.
    # We want to compare optimizers here,
    # not different Isolation Forest seeds.
    "random_state": [
        SEED,
    ],
}


# ==================================================
# 4. Load existing results
# ==================================================

if CSV_PATH.exists():

    print("\nExisting E05 results found.")
    print("Loading:")
    print(CSV_PATH)

    results_df = pd.read_csv(CSV_PATH)

else:

    results_df = pd.DataFrame()


# ==================================================
# 5. Run optimizer comparison
# ==================================================

for optimizer_name in OPTIMIZERS:

    # ----------------------------------------------
    # Skip only SUCCESSFUL completed runs
    # ----------------------------------------------

    if not results_df.empty:

        already_done = (
            (
                results_df["optimizer"]
                == optimizer_name
            )
            &
            (
                results_df["budget"]
                == BUDGET
            )
            &
            (
                results_df["seed"]
                == SEED
            )
            &
            (
                results_df["status"]
                == "success"
            )
        ).any()

        if already_done:

            print(
                f"\n{optimizer_name} already "
                f"completed successfully. Skipping."
            )

            continue


    # ----------------------------------------------
    # Remove previous failed row for same setup
    # ----------------------------------------------

    if not results_df.empty:

        same_run_mask = (
            (
                results_df["optimizer"]
                == optimizer_name
            )
            &
            (
                results_df["budget"]
                == BUDGET
            )
            &
            (
                results_df["seed"]
                == SEED
            )
        )

        old_failed_rows = (
            same_run_mask
            &
            (
                results_df["status"]
                != "success"
            )
        )

        if old_failed_rows.any():

            print(
                f"\nRemoving previous failed "
                f"record for {optimizer_name}."
            )

            results_df = results_df[
                ~old_failed_rows
            ].copy()


    # ----------------------------------------------
    # Start optimizer
    # ----------------------------------------------

    print("\n")
    print("=" * 65)

    print(
        f"E05 - OPTIMIZER: "
        f"{optimizer_name.upper()}"
    )

    print(
        f"BUDGET: {BUDGET}"
    )

    print(
        f"SEED: {SEED}"
    )

    print("=" * 65)

    start_time = time.perf_counter()

    try:

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
                f"E05_{optimizer_name}_"
                f"budget_{BUDGET}_"
                f"seed_{SEED}"
            ],

            MAX_RUNS=BUDGET,

            # Sequential execution gives a
            # cleaner runtime comparison.
            MAX_JOBS=1,

            INITIAL_RANDOM=2,

            optimization_param="AD1_AUC",
            maximize=True,

            optimizer=optimizer_name,

            random_state=SEED,
            deterministic=True,

            use_cache=False,
            log_best_pipeline=False,
        )

        runtime = (
            time.perf_counter()
            - start_time
        )

        best = result[0]

        params = best["best_params"]

        row = {
            "optimizer":
                optimizer_name,

            "budget":
                BUDGET,

            "seed":
                SEED,

            "best_validation_ad1_auc":
                float(
                    best[
                        "best_objective"
                    ]
                ),

            "runtime_seconds":
                round(
                    runtime,
                    2
                ),

            "window":
                params.get(
                    "method_window"
                ),

            "slide":
                params.get(
                    "method_slide"
                ),

            "n_estimators":
                params.get(
                    "method_n_estimators"
                ),

            "max_samples":
                params.get(
                    "method_max_samples"
                ),

            "status":
                "success",

            "best_params_json":
                json.dumps(
                    params,
                    default=str,
                ),
        }

        print(
            "\nBest AD1_AUC:",
            row[
                "best_validation_ad1_auc"
            ]
        )

        print(
            "Runtime:",
            row[
                "runtime_seconds"
            ],
            "seconds",
        )

        print(
            "Best params:",
            params,
        )


    # ----------------------------------------------
    # Save failures too
    # ----------------------------------------------

    except Exception as exc:

        runtime = (
            time.perf_counter()
            - start_time
        )

        print(
            f"\nERROR in "
            f"{optimizer_name}:"
        )

        print(exc)

        traceback.print_exc()

        row = {
            "optimizer":
                optimizer_name,

            "budget":
                BUDGET,

            "seed":
                SEED,

            "best_validation_ad1_auc":
                None,

            "runtime_seconds":
                round(
                    runtime,
                    2
                ),

            "window":
                None,

            "slide":
                None,

            "n_estimators":
                None,

            "max_samples":
                None,

            "status":
                f"error: {exc}",

            "best_params_json":
                None,
        }


    # ----------------------------------------------
    # Append and save immediately
    # ----------------------------------------------

    results_df = pd.concat(
        [
            results_df,
            pd.DataFrame(
                [row]
            ),
        ],
        ignore_index=True,
    )

    results_df.to_csv(
        CSV_PATH,
        index=False,
    )

    print(
        "\nIntermediate results saved to:"
    )

    print(CSV_PATH)


# ==================================================
# 6. Final successful results
# ==================================================

print("\n")
print("=" * 70)
print("E05 - OPTIMIZER COMPARISON")
print("=" * 70)

success_df = results_df[
    results_df["status"]
    == "success"
].copy()

if not success_df.empty:

    success_df = success_df.sort_values(
        "best_validation_ad1_auc",
        ascending=False,
    )

    display_columns = [
        "optimizer",
        "budget",
        "best_validation_ad1_auc",
        "runtime_seconds",
        "window",
        "slide",
        "n_estimators",
        "max_samples",
    ]

    print(
        success_df[
            display_columns
        ].to_string(
            index=False
        )
    )

else:

    print(
        "No successful optimizer runs."
    )


# ==================================================
# 7. Failed results
# ==================================================

failed_df = results_df[
    results_df["status"]
    != "success"
].copy()

if not failed_df.empty:

    print("\n")
    print("=" * 70)
    print("FAILED OPTIMIZER RUNS")
    print("=" * 70)

    print(
        failed_df[
            [
                "optimizer",
                "budget",
                "seed",
                "status",
            ]
        ].to_string(
            index=False
        )
    )


# ==================================================
# 8. Save summary
# ==================================================

with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "Experiment: "
        "E05 - Optimizer Comparison\n"
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
        "Metric: AD1_AUC\n"
    )

    f.write(
        f"Budget: {BUDGET}\n"
    )

    f.write(
        f"Seed: {SEED}\n\n"
    )

    f.write(
        "SEARCH SPACE\n"
    )

    f.write(
        json.dumps(
            SEARCH_SPACE,
            indent=2,
            default=str,
        )
    )

    f.write(
        "\n\nSUCCESSFUL RUNS\n"
    )

    if not success_df.empty:

        f.write(
            success_df[
                [
                    "optimizer",
                    "best_validation_ad1_auc",
                    "runtime_seconds",
                    "window",
                    "slide",
                    "n_estimators",
                    "max_samples",
                ]
            ].to_string(
                index=False
            )
        )

    else:

        f.write(
            "None\n"
        )

    if not failed_df.empty:

        f.write(
            "\n\nFAILED RUNS\n"
        )

        f.write(
            failed_df[
                [
                    "optimizer",
                    "budget",
                    "seed",
                    "status",
                ]
            ].to_string(
                index=False
            )
        )


# ==================================================
# 9. Final paths
# ==================================================

print("\n")
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print(CSV_PATH)
print(SUMMARY_PATH)