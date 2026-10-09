from pathlib import Path
import json
import sys
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
    / "04c_budget_sensitivity_corrected.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "04c_budget_sensitivity_corrected_summary.txt"
)


# ==================================================
# 2. Experiment settings
# ==================================================

SEED = 42

# If budgets are given in the command line,
# use them.
#
# Example:
# python script.py 4 8
#
# Otherwise run all four.

if len(sys.argv) > 1:

    BUDGETS = [
        int(x)
        for x in sys.argv[1:]
    ]

else:

    BUDGETS = [
        4,
        8,
        16,
        32,
    ]


print(
    "Budgets to run:",
    BUDGETS
)


# ==================================================
# 3. Load IMS dataset
# ==================================================

print("\n=== Loading IMS dataset ===")

df = pd.read_csv(
    DATA_PATH
)

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


print(
    "Optimization sources:",
    optimization_data["target_sources"]
)


# ==================================================
# 4. Corrected search space
# ==================================================
#
# IMPORTANT:
#
# max_features is NOT included.
# -> sklearn default = 1.0
#
# policy is NOT included.
# -> PdMLabs default = "or"
#
# bootstrap is NOT included.
# -> sklearn default = False
#
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

    "n_estimators": [
        50,
        100,
        200,
    ],

    "max_samples": [
        20,
        40,
    ],

    "random_state": [
        SEED,
    ],
}


NUMBER_OF_CONFIGURATIONS = (
    3
    * 3
    * 3
    * 2
)


print(
    "\nCorrected search space:"
)

print(
    SEARCH_SPACE
)

print(
    "\nTotal possible configurations:",
    NUMBER_OF_CONFIGURATIONS
)


# ==================================================
# 5. Load existing results
# ==================================================

if CSV_PATH.exists():

    print(
        "\nExisting E04c results found."
    )

    results_df = pd.read_csv(
        CSV_PATH
    )

else:

    results_df = pd.DataFrame()


# ==================================================
# 6. Run budgets
# ==================================================

for budget in BUDGETS:

    # ----------------------------------------------
    # Check if already completed
    # ----------------------------------------------

    if not results_df.empty:

        completed = (
            (
                results_df["budget"]
                == budget
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

        if completed:

            print(
                f"\nBudget {budget} "
                f"already completed. Skipping."
            )

            continue


    # ----------------------------------------------
    # Remove old failed run for same budget
    # ----------------------------------------------

    if not results_df.empty:

        same_budget = (
            (
                results_df["budget"]
                == budget
            )
            &
            (
                results_df["seed"]
                == SEED
            )
        )

        failed_same_budget = (
            same_budget
            &
            (
                results_df["status"]
                != "success"
            )
        )

        if failed_same_budget.any():

            results_df = (
                results_df[
                    ~failed_same_budget
                ]
                .copy()
            )


    # ----------------------------------------------
    # Start
    # ----------------------------------------------

    print("\n")
    print("=" * 70)
    print(
        f"E04c - MANGO BUDGET = {budget}"
    )
    print("=" * 70)

    start = time.perf_counter()

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
                f"E04c_Mango_Budget_{budget}"
            ],

            MAX_RUNS=budget,

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

        best = result[0]

        params = best[
            "best_params"
        ]

        best_objective = float(
            best[
                "best_objective"
            ]
        )


        pipeline_value = best.get(
            "best_pipeline_objective"
        )

        if pipeline_value is None:

            pipeline_objective = None

        else:

            pipeline_objective = float(
                pipeline_value
            )


        threshold_value = best.get(
            "th"
        )

        if threshold_value is None:

            threshold = None

        else:

            threshold = float(
                threshold_value
            )


        row = {

            "budget":
                budget,

            "seed":
                SEED,

            "best_validation_ad1_auc":
                best_objective,

            "best_pipeline_ad1_auc":
                pipeline_objective,

            "threshold":
                threshold,

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

            # Explicitly record defaults
            "effective_max_features":
                1.0,

            "effective_policy":
                "or",

            "effective_bootstrap":
                False,

            "status":
                "success",

            "best_params_json":
                json.dumps(
                    params,
                    default=str,
                ),
        }


        print(
            "\nBest validation AD1_AUC:",
            best_objective
        )

        print(
            "Best parameters:",
            params
        )

        print(
            "Runtime:",
            round(
                runtime,
                2
            ),
            "seconds"
        )


    except Exception as exc:

        runtime = (
            time.perf_counter()
            - start
        )

        print(
            f"\nERROR for budget {budget}:"
        )

        print(
            exc
        )

        row = {

            "budget":
                budget,

            "seed":
                SEED,

            "best_validation_ad1_auc":
                None,

            "best_pipeline_ad1_auc":
                None,

            "threshold":
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

            "effective_max_features":
                1.0,

            "effective_policy":
                "or",

            "effective_bootstrap":
                False,

            "status":
                f"error: {exc}",

            "best_params_json":
                None,
        }


    # ----------------------------------------------
    # Save immediately
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


    results_df = (
        results_df
        .sort_values(
            "budget"
        )
        .reset_index(
            drop=True
        )
    )


    results_df.to_csv(
        CSV_PATH,
        index=False,
    )


    print(
        "\nIntermediate results saved:"
    )

    print(
        CSV_PATH
    )


# ==================================================
# 7. Successful results only
# ==================================================

success_df = results_df[
    results_df["status"]
    == "success"
].copy()


if not success_df.empty:

    success_df = (
        success_df
        .sort_values(
            "budget"
        )
    )


# ==================================================
# 8. Calculate budget-to-budget improvement
# ==================================================

if not success_df.empty:

    success_df[
        "objective_improvement"
    ] = (
        success_df[
            "best_validation_ad1_auc"
        ]
        .diff()
    )


    success_df[
        "runtime_increase_seconds"
    ] = (
        success_df[
            "runtime_seconds"
        ]
        .diff()
    )


# ==================================================
# 9. Print final comparison
# ==================================================

print("\n")
print("=" * 75)
print("E04c - CORRECTED BUDGET SENSITIVITY")
print("=" * 75)


if not success_df.empty:

    display_columns = [

        "budget",

        "best_validation_ad1_auc",

        "runtime_seconds",

        "objective_improvement",

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


# ==================================================
# 10. Best overall budget result
# ==================================================

if not success_df.empty:

    best_row = (
        success_df
        .sort_values(
            "best_validation_ad1_auc",
            ascending=False,
        )
        .iloc[0]
    )


    print("\n")
    print("=" * 75)
    print("BEST CONFIGURATION FOUND")
    print("=" * 75)


    print(
        "Budget:",
        int(
            best_row[
                "budget"
            ]
        )
    )

    print(
        "AD1_AUC:",
        best_row[
            "best_validation_ad1_auc"
        ]
    )

    print(
        "window:",
        best_row[
            "window"
        ]
    )

    print(
        "slide:",
        best_row[
            "slide"
        ]
    )

    print(
        "n_estimators:",
        best_row[
            "n_estimators"
        ]
    )

    print(
        "max_samples:",
        best_row[
            "max_samples"
        ]
    )


# ==================================================
# 11. Save summary
# ==================================================

with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "Experiment: "
        "E04c - Corrected Budget Sensitivity\n"
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
        "Metric: AD1_AUC\n"
    )

    f.write(
        f"Seed: {SEED}\n"
    )

    f.write(
        f"Budgets: {BUDGETS}\n\n"
    )


    f.write(
        "IMPORTANT CORRECTION\n"
    )

    f.write(
        "max_features is not included "
        "in the HPO search space.\n"
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
        "SEARCH SPACE\n"
    )

    f.write(
        json.dumps(
            SEARCH_SPACE,
            indent=2,
        )
    )


    f.write(
        "\n\nRESULTS\n"
    )


    if not success_df.empty:

        f.write(
            success_df[
                [
                    "budget",
                    "best_validation_ad1_auc",
                    "runtime_seconds",
                    "objective_improvement",
                    "window",
                    "slide",
                    "n_estimators",
                    "max_samples",
                ]
            ].to_string(
                index=False
            )
        )


    failed_df = results_df[
        results_df["status"]
        != "success"
    ]


    if not failed_df.empty:

        f.write(
            "\n\nFAILED RUNS\n"
        )

        f.write(
            failed_df[
                [
                    "budget",
                    "status",
                ]
            ].to_string(
                index=False
            )
        )


# ==================================================
# 12. Done
# ==================================================

print("\n")
print("=" * 75)
print("FILES SAVED")
print("=" * 75)

print(
    CSV_PATH
)

print(
    SUMMARY_PATH
)