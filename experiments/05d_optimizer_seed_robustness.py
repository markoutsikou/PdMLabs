from pathlib import Path
import argparse
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
    / "05d_optimizer_seed_robustness.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "05d_optimizer_seed_robustness_summary.txt"
)


# ==================================================
# 2. General experiment settings
# ==================================================

ALL_OPTIMIZERS = [
    "mango",
    "mango_random",
    "optuna",
    "smac",
]

DEFAULT_HPO_SEEDS = [
    1,
    10,
    42,
    100,
    2026,
]

DEFAULT_BUDGET = 16

INITIAL_RANDOM = 2


# IMPORTANT:
#
# This is the MODEL seed.
#
# It stays fixed across ALL experiments.
#
# Only the HPO seed changes.
#
# This isolates optimizer stochasticity from
# Isolation Forest stochasticity.

MODEL_RANDOM_STATE = 42


# ==================================================
# 3. Reference result from corrected E05c
# ==================================================
#
# All four optimizers obtained this score/configuration
# in corrected E05c with HPO seed 42.
#
# We use it only as a REFERENCE, not as a claim that
# it is necessarily the theoretical global optimum.
# ==================================================

REFERENCE_SCORE = (
    0.8333333333333334
)

REFERENCE_CONFIG = {
    "method_window": 160,
    "method_slide": 1.0,
    "method_n_estimators": 100,
    "method_max_samples": 40,
    "method_random_state": 42,
}

REFERENCE_TOLERANCE = 1e-10


# ==================================================
# 4. Command-line arguments
# ==================================================
#
# Examples:
#
# One smoke-test run:
#
# python script.py --optimizers mango --seeds 1
#
# All Mango-related:
#
# python script.py --optimizers mango mango_random
#
# Optuna + SMAC:
#
# python script.py --optimizers optuna smac
#
# Everything:
#
# python script.py
#
# ==================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--optimizers",
    nargs="+",
    default=ALL_OPTIMIZERS,
)

parser.add_argument(
    "--seeds",
    nargs="+",
    type=int,
    default=DEFAULT_HPO_SEEDS,
)

parser.add_argument(
    "--budget",
    type=int,
    default=DEFAULT_BUDGET,
)

args = parser.parse_args()

OPTIMIZERS = args.optimizers
HPO_SEEDS = args.seeds
BUDGET = args.budget


# ==================================================
# 5. Validate command-line arguments
# ==================================================

unknown_optimizers = [
    optimizer
    for optimizer in OPTIMIZERS
    if optimizer not in ALL_OPTIMIZERS
]

if unknown_optimizers:

    raise ValueError(
        "Unknown optimizer(s): "
        f"{unknown_optimizers}. "
        f"Allowed: {ALL_OPTIMIZERS}"
    )


print("\n")
print("=" * 80)
print("E05d - OPTIMIZER SEED ROBUSTNESS")
print("=" * 80)

print(
    "Optimizers:",
    OPTIMIZERS
)

print(
    "HPO seeds:",
    HPO_SEEDS
)

print(
    "Budget:",
    BUDGET
)

print(
    "Fixed model random_state:",
    MODEL_RANDOM_STATE
)


# ==================================================
# 6. Load IMS
# ==================================================

print(
    "\n=== Loading IMS dataset ==="
)

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
# 7. Corrected search space
# ==================================================
#
# IMPORTANT:
#
# random_state here is MODEL random_state.
# It is always fixed at 42.
#
# run_experiment(random_state=...)
# will receive the HPO seed separately.
#
# max_features is omitted -> 1.0
# policy is omitted -> "or"
# bootstrap is omitted -> False
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
        MODEL_RANDOM_STATE,
    ],
}


print(
    "\nCorrected search space:"
)

print(
    SEARCH_SPACE
)

print(
    "\nTotal candidate configurations:",
    3 * 3 * 3 * 2
)


# ==================================================
# 8. Helper functions
# ==================================================

def to_python_value(value):

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


def get_param_types(params):

    return {
        key: type(value).__name__
        for key, value in params.items()
    }


def configuration_signature(params):

    return (
        f"w={params.get('method_window')};"
        f"s={params.get('method_slide')};"
        f"n={params.get('method_n_estimators')};"
        f"m={params.get('method_max_samples')}"
    )


def matches_reference_config(params):

    for key, reference_value in (
        REFERENCE_CONFIG.items()
    ):

        if params.get(key) != reference_value:

            return False

    return True


# ==================================================
# 9. Load existing results
# ==================================================

if CSV_PATH.exists():

    print(
        "\nExisting E05d CSV found."
    )

    results_df = pd.read_csv(
        CSV_PATH
    )

else:

    results_df = pd.DataFrame()


# ==================================================
# 10. Summary writer
# ==================================================

def write_summary(df):

    successful = df[
        df["status"]
        == "success"
    ].copy()

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "Experiment: "
            "E05d - Optimizer Seed Robustness\n"
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
            "HPO seeds: "
            f"{DEFAULT_HPO_SEEDS}\n"
        )

        f.write(
            "Fixed model random_state: "
            f"{MODEL_RANDOM_STATE}\n\n"
        )


        f.write(
            "IMPORTANT METHODOLOGY\n"
        )

        f.write(
            "The HPO seed changes between runs, "
            "while the Isolation Forest "
            "random_state remains fixed at 42.\n"
        )

        f.write(
            "This isolates optimizer stochasticity "
            "from model stochasticity.\n"
        )

        f.write(
            "max_features is omitted from the "
            "search space; effective "
            "max_features=1.0.\n"
        )

        f.write(
            "Effective policy=or.\n"
        )

        f.write(
            "Effective bootstrap=False.\n"
        )

        f.write(
            "use_cache=False.\n\n"
        )


        f.write(
            "REFERENCE RESULT FROM E05c\n"
        )

        f.write(
            f"Reference AD1_AUC: "
            f"{REFERENCE_SCORE}\n"
        )

        f.write(
            "Reference configuration:\n"
        )

        f.write(
            json.dumps(
                REFERENCE_CONFIG,
                indent=2,
            )
        )

        f.write(
            "\n\n"
        )


        if successful.empty:

            f.write(
                "No successful runs yet.\n"
            )

            return


        successful = (
            successful
            .sort_values(
                [
                    "optimizer",
                    "hpo_seed",
                ]
            )
        )


        f.write(
            "PER-RUN RESULTS\n"
        )

        f.write(
            successful[
                [
                    "optimizer",
                    "hpo_seed",
                    "best_validation_ad1_auc",
                    "runtime_seconds",
                    "window",
                    "slide",
                    "n_estimators",
                    "max_samples",
                    "reached_reference_score",
                    "matches_reference_config",
                ]
            ].to_string(
                index=False
            )
        )


        grouped = (
            successful
            .groupby(
                "optimizer"
            )
            .agg(
                successful_runs=(
                    "hpo_seed",
                    "count",
                ),

                mean_best_ad1_auc=(
                    "best_validation_ad1_auc",
                    "mean",
                ),

                std_best_ad1_auc=(
                    "best_validation_ad1_auc",
                    "std",
                ),

                min_best_ad1_auc=(
                    "best_validation_ad1_auc",
                    "min",
                ),

                max_best_ad1_auc=(
                    "best_validation_ad1_auc",
                    "max",
                ),

                reference_score_hits=(
                    "reached_reference_score",
                    "sum",
                ),

                reference_config_hits=(
                    "matches_reference_config",
                    "sum",
                ),

                unique_best_configs=(
                    "configuration_signature",
                    "nunique",
                ),

                mean_runtime_seconds=(
                    "runtime_seconds",
                    "mean",
                ),

                median_runtime_seconds=(
                    "runtime_seconds",
                    "median",
                ),
            )
        )


        f.write(
            "\n\nSUMMARY BY OPTIMIZER\n"
        )

        f.write(
            grouped.to_string()
        )


        failed = df[
            df["status"]
            != "success"
        ]


        if not failed.empty:

            f.write(
                "\n\nFAILED RUNS\n"
            )

            f.write(
                failed[
                    [
                        "optimizer",
                        "hpo_seed",
                        "status",
                    ]
                ].to_string(
                    index=False
                )
            )


# ==================================================
# 11. Main experiment loop
# ==================================================

for optimizer_name in OPTIMIZERS:

    for hpo_seed in HPO_SEEDS:

        # ------------------------------------------
        # Skip already-successful run
        # ------------------------------------------

        if not results_df.empty:

            already_successful = (
                (
                    results_df["optimizer"]
                    == optimizer_name
                )
                &
                (
                    results_df["hpo_seed"]
                    == hpo_seed
                )
                &
                (
                    results_df["budget"]
                    == BUDGET
                )
                &
                (
                    results_df["status"]
                    == "success"
                )
            ).any()

            if already_successful:

                print(
                    "\nSKIP:"
                    f" {optimizer_name}, "
                    f"HPO seed={hpo_seed} "
                    "already successful."
                )

                continue


        # ------------------------------------------
        # Remove previous failed row for this run
        # ------------------------------------------

        if not results_df.empty:

            same_run = (
                (
                    results_df["optimizer"]
                    == optimizer_name
                )
                &
                (
                    results_df["hpo_seed"]
                    == hpo_seed
                )
                &
                (
                    results_df["budget"]
                    == BUDGET
                )
            )

            failed_same_run = (
                same_run
                &
                (
                    results_df["status"]
                    != "success"
                )
            )

            if failed_same_run.any():

                results_df = (
                    results_df[
                        ~failed_same_run
                    ]
                    .copy()
                )


        # ------------------------------------------
        # Start run
        # ------------------------------------------

        print("\n")
        print("=" * 80)

        print(
            f"OPTIMIZER = "
            f"{optimizer_name.upper()}"
        )

        print(
            f"HPO SEED = {hpo_seed}"
        )

        print(
            f"MODEL SEED = "
            f"{MODEL_RANDOM_STATE}"
        )

        print("=" * 80)


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
                        "E05d_"
                        f"{optimizer_name}_"
                        f"hposeed_{hpo_seed}"
                    )
                ],

                MAX_RUNS=BUDGET,

                MAX_JOBS=1,

                INITIAL_RANDOM=INITIAL_RANDOM,

                optimization_param="AD1_AUC",

                maximize=True,

                optimizer=optimizer_name,

                # THIS is the HPO seed.
                random_state=hpo_seed,

                deterministic=True,

                use_cache=False,

                log_best_pipeline=False,
            )


            runtime = (
                time.perf_counter()
                - start
            )


            best = result[0]


            # ======================================
            # Objective
            # ======================================

            objective_value = best.get(
                "best_objective"
            )

            if objective_value is None:

                raise RuntimeError(
                    "best_objective is None."
                )


            best_objective = float(
                objective_value
            )


            # ======================================
            # Parameters
            # ======================================

            raw_params = best.get(
                "best_params"
            )

            if raw_params is None:

                raise RuntimeError(
                    "best_params is None."
                )


            normalized_params = (
                normalize_params(
                    raw_params
                )
            )

            types = get_param_types(
                raw_params
            )


            # ======================================
            # Corrected-space safety audit
            # ======================================

            forbidden = [
                "method_max_features",
                "method_policy",
                "method_bootstrap",
            ]

            unexpected = [
                parameter
                for parameter in forbidden
                if parameter in raw_params
            ]

            if unexpected:

                raise RuntimeError(
                    "Corrected search-space audit "
                    "failed. Unexpected params: "
                    f"{unexpected}"
                )


            # ======================================
            # Verify model seed did not change
            # ======================================

            returned_model_seed = (
                normalized_params.get(
                    "method_random_state"
                )
            )

            if (
                returned_model_seed
                != MODEL_RANDOM_STATE
            ):

                raise RuntimeError(
                    "Model random_state changed! "
                    f"Expected "
                    f"{MODEL_RANDOM_STATE}, "
                    f"got "
                    f"{returned_model_seed}."
                )


            # ======================================
            # Pipeline metadata
            # ======================================

            pipeline_value = best.get(
                "best_pipeline_objective"
            )

            pipeline_objective = (
                None
                if pipeline_value is None
                else float(
                    pipeline_value
                )
            )


            threshold_value = best.get(
                "th"
            )

            threshold = (
                None
                if threshold_value is None
                else float(
                    threshold_value
                )
            )


            # ======================================
            # Reference comparison
            # ======================================

            reached_reference_score = (
                best_objective
                >= (
                    REFERENCE_SCORE
                    - REFERENCE_TOLERANCE
                )
            )

            reference_config_match = (
                matches_reference_config(
                    normalized_params
                )
            )


            config_signature = (
                configuration_signature(
                    normalized_params
                )
            )


            # ======================================
            # Output
            # ======================================

            print(
                "\nBest validation AD1_AUC:",
                best_objective
            )

            print(
                "Reached/exceeded reference:",
                reached_reference_score
            )

            print(
                "Matches reference config:",
                reference_config_match
            )

            print(
                "\nBest parameters:"
            )


            for key, value in (
                raw_params.items()
            ):

                print(
                    f"  {key} = "
                    f"{value!r} "
                    f"({type(value).__name__})"
                )


            print(
                "\nRuntime:",
                round(runtime, 2),
                "seconds"
            )


            # ======================================
            # Successful row
            # ======================================

            row = {

                "optimizer":
                    optimizer_name,

                "hpo_seed":
                    hpo_seed,

                "model_random_state":
                    MODEL_RANDOM_STATE,

                "budget":
                    BUDGET,

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
                    normalized_params.get(
                        "method_window"
                    ),

                "slide":
                    normalized_params.get(
                        "method_slide"
                    ),

                "n_estimators":
                    normalized_params.get(
                        "method_n_estimators"
                    ),

                "max_samples":
                    normalized_params.get(
                        "method_max_samples"
                    ),

                "returned_model_random_state":
                    returned_model_seed,

                "effective_max_features":
                    1.0,

                "effective_policy":
                    "or",

                "effective_bootstrap":
                    False,

                "reached_reference_score":
                    reached_reference_score,

                "matches_reference_config":
                    reference_config_match,

                "configuration_signature":
                    config_signature,

                "status":
                    "success",

                "best_params_json":
                    json.dumps(
                        normalized_params,
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

                "hpo_seed":
                    hpo_seed,

                "model_random_state":
                    MODEL_RANDOM_STATE,

                "budget":
                    BUDGET,

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

                "returned_model_random_state":
                    None,

                "effective_max_features":
                    1.0,

                "effective_policy":
                    "or",

                "effective_bootstrap":
                    False,

                "reached_reference_score":
                    False,

                "matches_reference_config":
                    False,

                "configuration_signature":
                    None,

                "status":
                    (
                        "error: "
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),

                "best_params_json":
                    None,

                "best_param_types_json":
                    None,
            }


        # ==========================================
        # Save immediately after every run
        # ==========================================

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
                [
                    "optimizer",
                    "hpo_seed",
                ]
            )
            .reset_index(
                drop=True
            )
        )


        results_df.to_csv(
            CSV_PATH,
            index=False,
        )


        write_summary(
            results_df
        )


        print(
            "\nIntermediate CSV saved:"
        )

        print(
            CSV_PATH
        )

        print(
            "Intermediate summary saved:"
        )

        print(
            SUMMARY_PATH
        )


# ==================================================
# 12. Final successful results
# ==================================================

successful_df = results_df[
    results_df[
        "status"
    ]
    == "success"
].copy()


if not successful_df.empty:

    successful_df = (
        successful_df
        .sort_values(
            [
                "optimizer",
                "hpo_seed",
            ]
        )
    )


# ==================================================
# 13. Per-run results
# ==================================================

print("\n")
print("=" * 90)
print("PER-RUN RESULTS")
print("=" * 90)


if successful_df.empty:

    print(
        "No successful runs."
    )

else:

    print(
        successful_df[
            [
                "optimizer",
                "hpo_seed",
                "best_validation_ad1_auc",
                "runtime_seconds",
                "window",
                "slide",
                "n_estimators",
                "max_samples",
                "reached_reference_score",
                "matches_reference_config",
            ]
        ].to_string(
            index=False
        )
    )


# ==================================================
# 14. Optimizer robustness summary
# ==================================================

if not successful_df.empty:

    optimizer_summary = (
        successful_df
        .groupby(
            "optimizer"
        )
        .agg(

            successful_runs=(
                "hpo_seed",
                "count",
            ),

            mean_best_ad1_auc=(
                "best_validation_ad1_auc",
                "mean",
            ),

            std_best_ad1_auc=(
                "best_validation_ad1_auc",
                "std",
            ),

            min_best_ad1_auc=(
                "best_validation_ad1_auc",
                "min",
            ),

            max_best_ad1_auc=(
                "best_validation_ad1_auc",
                "max",
            ),

            reference_score_hits=(
                "reached_reference_score",
                "sum",
            ),

            reference_config_hits=(
                "matches_reference_config",
                "sum",
            ),

            unique_best_configs=(
                "configuration_signature",
                "nunique",
            ),

            mean_runtime_seconds=(
                "runtime_seconds",
                "mean",
            ),

            median_runtime_seconds=(
                "runtime_seconds",
                "median",
            ),
        )
    )


    optimizer_summary = (
        optimizer_summary
        .sort_values(
            [
                "mean_best_ad1_auc",
                "reference_score_hits",
                "mean_runtime_seconds",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
    )


    print("\n")
    print("=" * 90)
    print("OPTIMIZER ROBUSTNESS SUMMARY")
    print("=" * 90)

    print(
        optimizer_summary
    )


# ==================================================
# 15. Final write
# ==================================================

write_summary(
    results_df
)


print("\n")
print("=" * 90)
print("E05d FINISHED")
print("=" * 90)

print(
    CSV_PATH
)

print(
    SUMMARY_PATH
)