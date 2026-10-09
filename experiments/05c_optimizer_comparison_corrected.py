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
    / "05c_optimizer_comparison_corrected.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "05c_optimizer_comparison_corrected_summary.txt"
)


# ==================================================
# 2. General settings
# ==================================================

SEED = 42
BUDGET = 16
INITIAL_RANDOM = 2


ALL_OPTIMIZERS = [
    "mango",
    "mango_random",
    "optuna",
    "smac",
]


# ==================================================
# 3. Optional command-line optimizer selection
# ==================================================
#
# Examples:
#
# python script.py mango
#
# python script.py mango mango_random
#
# python script.py optuna smac
#
# If no arguments are supplied, run all.
# ==================================================

if len(sys.argv) > 1:

    OPTIMIZERS = [
        x.strip()
        for x in sys.argv[1:]
    ]

else:

    OPTIMIZERS = (
        ALL_OPTIMIZERS.copy()
    )


unknown = [
    x
    for x in OPTIMIZERS
    if x not in ALL_OPTIMIZERS
]

if unknown:

    raise ValueError(
        f"Unknown optimizer(s): {unknown}. "
        f"Allowed: {ALL_OPTIMIZERS}"
    )


print(
    "Optimizers to run:",
    OPTIMIZERS
)


# ==================================================
# 4. Load IMS
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
# 5. Corrected search space
# ==================================================
#
# IMPORTANT:
#
# max_features is NOT optimized.
# -> effective sklearn default = 1.0
#
# policy is NOT optimized.
# -> effective PdMLabs default = "or"
#
# bootstrap is NOT optimized.
# -> effective sklearn default = False
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


TOTAL_CONFIGURATIONS = (
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
    TOTAL_CONFIGURATIONS
)


# ==================================================
# 6. Helpers
# ==================================================

def to_python_value(value):
    """
    Convert NumPy scalar values to regular
    Python scalars when possible.
    """

    if hasattr(value, "item"):

        try:

            return value.item()

        except Exception:

            pass

    return value


def normalize_params(params):

    return {
        key: to_python_value(value)
        for key, value in params.items()
    }


def param_types(params):

    return {
        key: type(value).__name__
        for key, value in params.items()
    }


# ==================================================
# 7. Load previous results if they exist
# ==================================================

if CSV_PATH.exists():

    print(
        "\nExisting E05c results found."
    )

    results_df = pd.read_csv(
        CSV_PATH
    )

else:

    results_df = pd.DataFrame()


# ==================================================
# 8. Run optimizer comparison
# ==================================================

for optimizer_name in OPTIMIZERS:

    # ----------------------------------------------
    # Skip successful completed run
    # ----------------------------------------------

    if not results_df.empty:

        successful_run = (
            (
                results_df[
                    "optimizer"
                ]
                == optimizer_name
            )
            &
            (
                results_df[
                    "budget"
                ]
                == BUDGET
            )
            &
            (
                results_df[
                    "seed"
                ]
                == SEED
            )
            &
            (
                results_df[
                    "status"
                ]
                == "success"
            )
        ).any()

        if successful_run:

            print(
                "\n"
                f"{optimizer_name} already "
                f"completed successfully. "
                f"Skipping."
            )

            continue


    # ----------------------------------------------
    # Remove previous failed result
    # ----------------------------------------------

    if not results_df.empty:

        same_setup = (
            (
                results_df[
                    "optimizer"
                ]
                == optimizer_name
            )
            &
            (
                results_df[
                    "budget"
                ]
                == BUDGET
            )
            &
            (
                results_df[
                    "seed"
                ]
                == SEED
            )
        )

        failed_same_setup = (
            same_setup
            &
            (
                results_df[
                    "status"
                ]
                != "success"
            )
        )

        if failed_same_setup.any():

            print(
                "\nRemoving previous failed "
                f"row for {optimizer_name}."
            )

            results_df = (
                results_df[
                    ~failed_same_setup
                ]
                .copy()
            )


    # ----------------------------------------------
    # Start optimizer
    # ----------------------------------------------

    print("\n")
    print("=" * 75)
    print(
        f"E05c - OPTIMIZER: "
        f"{optimizer_name.upper()}"
    )
    print("=" * 75)

    print(
        f"Budget = {BUDGET}"
    )

    print(
        f"Seed = {SEED}"
    )


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
                (
                    f"E05c_"
                    f"{optimizer_name}_"
                    f"budget_{BUDGET}"
                )
            ],

            MAX_RUNS=BUDGET,

            MAX_JOBS=1,

            INITIAL_RANDOM=INITIAL_RANDOM,

            optimization_param="AD1_AUC",

            maximize=True,

            optimizer=optimizer_name,

            random_state=SEED,

            deterministic=True,

            # Very important:
            # do not reuse older contaminated trials.
            use_cache=False,

            log_best_pipeline=False,
        )


        runtime = (
            time.perf_counter()
            - start
        )


        best = result[0]


        # ==========================================
        # Best objective
        # ==========================================

        objective_value = best.get(
            "best_objective"
        )

        if objective_value is None:

            raise RuntimeError(
                "Optimizer returned "
                "best_objective=None."
            )


        best_objective = float(
            objective_value
        )


        # ==========================================
        # Best parameters
        # ==========================================

        raw_params = best.get(
            "best_params"
        )

        if raw_params is None:

            raise RuntimeError(
                "Optimizer returned "
                "best_params=None."
            )


        # ==========================================
        # Safety audit
        # ==========================================
        #
        # These parameters MUST NOT appear in the
        # corrected HPO result.
        # ==========================================

        forbidden_parameters = [
            "method_max_features",
            "method_policy",
            "method_bootstrap",
        ]


        found_forbidden = [
            p
            for p in forbidden_parameters
            if p in raw_params
        ]


        if found_forbidden:

            raise RuntimeError(
                "Corrected HPO audit failed. "
                "Unexpected parameter(s) "
                f"returned: {found_forbidden}"
            )


        normalized = normalize_params(
            raw_params
        )

        types = param_types(
            raw_params
        )


        # ==========================================
        # Pipeline metadata
        # ==========================================

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


        # ==========================================
        # Print result
        # ==========================================

        print(
            "\nBest validation AD1_AUC:",
            best_objective
        )

        print(
            "\nBest parameters:"
        )

        for key, value in raw_params.items():

            print(
                f"  {key} = "
                f"{value!r} "
                f"({type(value).__name__})"
            )


        print(
            "\nRuntime:",
            round(
                runtime,
                2
            ),
            "seconds"
        )


        # ==========================================
        # Row
        # ==========================================

        row = {

            "optimizer":
                optimizer_name,

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
                round(
                    runtime,
                    2
                ),

            "window":
                normalized.get(
                    "method_window"
                ),

            "window_type":
                types.get(
                    "method_window"
                ),

            "slide":
                normalized.get(
                    "method_slide"
                ),

            "slide_type":
                types.get(
                    "method_slide"
                ),

            "n_estimators":
                normalized.get(
                    "method_n_estimators"
                ),

            "n_estimators_type":
                types.get(
                    "method_n_estimators"
                ),

            "max_samples":
                normalized.get(
                    "method_max_samples"
                ),

            "max_samples_type":
                types.get(
                    "method_max_samples"
                ),

            "random_state":
                normalized.get(
                    "method_random_state"
                ),

            "random_state_type":
                types.get(
                    "method_random_state"
                ),

            # Fixed effective defaults
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
                    normalized,
                    default=str,
                    sort_keys=True,
                ),

            "best_param_types_json":
                json.dumps(
                    types,
                    sort_keys=True,
                ),
        }


    except Exception as exc:

        runtime = (
            time.perf_counter()
            - start
        )

        print(
            "\nERROR:"
        )

        print(
            repr(exc)
        )


        row = {

            "optimizer":
                optimizer_name,

            "budget":
                BUDGET,

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

            "window_type":
                None,

            "slide":
                None,

            "slide_type":
                None,

            "n_estimators":
                None,

            "n_estimators_type":
                None,

            "max_samples":
                None,

            "max_samples_type":
                None,

            "random_state":
                None,

            "random_state_type":
                None,

            "effective_max_features":
                1.0,

            "effective_policy":
                "or",

            "effective_bootstrap":
                False,

            "status":
                (
                    f"error: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),

            "best_params_json":
                None,

            "best_param_types_json":
                None,
        }


    # ----------------------------------------------
    # Save immediately after every optimizer
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
        "\nIntermediate results saved:"
    )

    print(
        CSV_PATH
    )


# ==================================================
# 9. Successful runs
# ==================================================

success_df = results_df[
    results_df[
        "status"
    ]
    == "success"
].copy()


if not success_df.empty:

    success_df = (
        success_df
        .sort_values(
            [
                "best_validation_ad1_auc",
                "runtime_seconds",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )


# ==================================================
# 10. Print comparison
# ==================================================

print("\n")
print("=" * 80)
print("E05c - CORRECTED OPTIMIZER COMPARISON")
print("=" * 80)


if success_df.empty:

    print(
        "No successful optimizer runs."
    )

else:

    columns = [
        "optimizer",
        "best_validation_ad1_auc",
        "runtime_seconds",
        "window",
        "slide",
        "n_estimators",
        "max_samples",
    ]


    print(
        success_df[
            columns
        ].to_string(
            index=False
        )
    )


# ==================================================
# 11. Parameter type audit
# ==================================================

if not success_df.empty:

    print("\n")
    print("=" * 80)
    print("PARAMETER TYPE AUDIT")
    print("=" * 80)


    type_columns = [
        "optimizer",
        "window_type",
        "slide_type",
        "n_estimators_type",
        "max_samples_type",
        "random_state_type",
    ]


    print(
        success_df[
            type_columns
        ].to_string(
            index=False
        )
    )


# ==================================================
# 12. Best optimizer on validation
# ==================================================

if not success_df.empty:

    best_row = success_df.iloc[0]


    print("\n")
    print("=" * 80)
    print("BEST VALIDATION RESULT")
    print("=" * 80)


    print(
        "Optimizer:",
        best_row[
            "optimizer"
        ]
    )

    print(
        "AD1_AUC:",
        best_row[
            "best_validation_ad1_auc"
        ]
    )

    print(
        "Configuration:"
    )

    print(
        "  window =",
        best_row[
            "window"
        ]
    )

    print(
        "  slide =",
        best_row[
            "slide"
        ]
    )

    print(
        "  n_estimators =",
        best_row[
            "n_estimators"
        ]
    )

    print(
        "  max_samples =",
        best_row[
            "max_samples"
        ]
    )


# ==================================================
# 13. Save summary
# ==================================================

with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "Experiment: "
        "E05c - Corrected Optimizer Comparison\n"
    )

    f.write(
        "Dataset: IMS\n"
    )

    f.write(
        "Optimization sources: 1 + 2\n"
    )

    f.write(
        "Method: IsolationForestUnsupervised\n"
    )

    f.write(
        "Metric: AD1_AUC\n"
    )

    f.write(
        f"Budget: {BUDGET}\n"
    )

    f.write(
        f"Seed: {SEED}\n"
    )

    f.write(
        f"Initial random trials: "
        f"{INITIAL_RANDOM}\n\n"
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
        "Effective bootstrap = False.\n"
    )

    f.write(
        "use_cache=False to avoid reuse "
        "of older contaminated trials.\n\n"
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
        "\n\nSUCCESSFUL RESULTS\n"
    )


    if success_df.empty:

        f.write(
            "None\n"
        )

    else:

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


        f.write(
            "\n\nPARAMETER TYPE AUDIT\n"
        )

        f.write(
            success_df[
                [
                    "optimizer",
                    "window_type",
                    "slide_type",
                    "n_estimators_type",
                    "max_samples_type",
                    "random_state_type",
                ]
            ].to_string(
                index=False
            )
        )


    failed_df = results_df[
        results_df[
            "status"
        ]
        != "success"
    ]


    if not failed_df.empty:

        f.write(
            "\n\nFAILED RUNS\n"
        )

        f.write(
            failed_df[
                [
                    "optimizer",
                    "status",
                ]
            ].to_string(
                index=False
            )
        )


# ==================================================
# 14. Done
# ==================================================

print("\n")
print("=" * 80)
print("FILES SAVED")
print("=" * 80)

print(
    CSV_PATH
)

print(
    SUMMARY_PATH
)