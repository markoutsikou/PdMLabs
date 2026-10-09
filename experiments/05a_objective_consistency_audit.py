from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from pdmlabs.utils.dataset import Dataset
from pdmlabs.RunExperiment import run_experiment
from pdmlabs.experiment.batch.unsupervised_experiment import (
    UnsupervisedPdMExperiment,
)
from pdmlabs.method.isolation_forest_uns import (
    IsolationForestUnsupervised,
)
from pdmlabs.utils.seeding import seed_for_params


# ==========================================================
# 1. Paths / configuration
# ==========================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "ims.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CSV_PATH = RESULTS_DIR / "05a_objective_consistency_audit.csv"
SUMMARY_PATH = RESULTS_DIR / "05a_objective_consistency_audit.txt"

OPTIMIZERS = [
    "mango",
    "mango_random",
    "optuna",
    "smac",
]

EXPERIMENT_SEED = 42


# ==========================================================
# 2. Dataset
# ==========================================================

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


# ==========================================================
# 3. ONE AND ONLY ONE configuration
# ==========================================================

FIXED_SPACE = {
    "window": [160],
    "slide": [1.0],
    "policy": ["or"],
    "n_estimators": [100],
    "max_samples": [40],
    "max_features": [1.0],
    "bootstrap": [False],
    "random_state": [42],
}


# ==========================================================
# 4. Helpers
# ==========================================================

def type_dict(d):
    if d is None:
        return None

    return {
        str(k): type(v).__name__
        for k, v in d.items()
    }


def normalize_for_display(value):

    if isinstance(value, np.generic):
        return value.item()

    return value


def normalized_dict(d):

    if d is None:
        return None

    return {
        str(k): normalize_for_display(v)
        for k, v in d.items()
    }


# ==========================================================
# 5. Run same configuration through every optimizer
# ==========================================================

records = []

print("\n")
print("=" * 75)
print("E05a - OBJECTIVE CONSISTENCY AUDIT")
print("=" * 75)

for optimizer_name in OPTIMIZERS:

    print("\n")
    print("=" * 75)
    print(f"OPTIMIZER = {optimizer_name.upper()}")
    print("=" * 75)

    start = time.perf_counter()

    try:

        result = run_experiment(
            dataset=optimization_data,

            methods=[
                IsolationForestUnsupervised
            ],

            param_space_dict_per_method=[
                FIXED_SPACE
            ],

            method_names=[
                "IsolationForestUnsupervised"
            ],

            experiments=[
                UnsupervisedPdMExperiment
            ],

            experiment_names=[
                f"E05a_fixed_{optimizer_name}"
            ],

            # Smallest practical budget.
            # There is only ONE possible configuration.
            MAX_RUNS=2,
            MAX_JOBS=1,
            INITIAL_RANDOM=1,

            optimization_param="AD1_AUC",
            maximize=True,

            optimizer=optimizer_name,

            random_state=EXPERIMENT_SEED,
            deterministic=True,

            use_cache=False,
            log_best_pipeline=False,
        )

        runtime = time.perf_counter() - start

        result = result[0]

        optimizer_params = result.get(
            "best_params"
        )

        pipeline_params = result.get(
            "best_pipeline_params"
        )

        optimizer_objective = result.get(
            "best_objective"
        )

        pipeline_objective = result.get(
            "best_pipeline_objective"
        )

        threshold = result.get("th")


        # --------------------------------------------------
        # Trial seeds derived from returned parameter dicts
        # --------------------------------------------------

        optimizer_trial_seed = None
        pipeline_trial_seed = None

        if optimizer_params is not None:

            optimizer_trial_seed = seed_for_params(
                EXPERIMENT_SEED,
                optimizer_params,
            )

        if pipeline_params is not None:

            pipeline_trial_seed = seed_for_params(
                EXPERIMENT_SEED,
                pipeline_params,
            )


        print("\nOptimizer-reported:")
        print("params =", optimizer_params)
        print("objective =", optimizer_objective)

        print("\nTrial-sink / pipeline:")
        print("params =", pipeline_params)
        print("objective =", pipeline_objective)

        print("\nParameter Python types:")
        print(
            "optimizer:",
            type_dict(optimizer_params)
        )

        print(
            "pipeline:",
            type_dict(pipeline_params)
        )

        print("\nDerived trial seeds:")
        print(
            "optimizer params seed =",
            optimizer_trial_seed,
        )

        print(
            "pipeline params seed =",
            pipeline_trial_seed,
        )

        if (
            optimizer_objective is not None
            and pipeline_objective is not None
        ):

            objective_difference = (
                float(optimizer_objective)
                - float(pipeline_objective)
            )

        else:

            objective_difference = None


        records.append({
            "optimizer":
                optimizer_name,

            "optimizer_best_objective":
                float(optimizer_objective)
                if optimizer_objective is not None
                else None,

            "pipeline_best_objective":
                float(pipeline_objective)
                if pipeline_objective is not None
                else None,

            "objective_difference":
                objective_difference,

            "threshold":
                float(threshold)
                if threshold is not None
                else None,

            "runtime_seconds":
                runtime,

            "optimizer_trial_seed":
                optimizer_trial_seed,

            "pipeline_trial_seed":
                pipeline_trial_seed,

            "optimizer_param_types":
                json.dumps(
                    type_dict(optimizer_params),
                    default=str,
                ),

            "pipeline_param_types":
                json.dumps(
                    type_dict(pipeline_params),
                    default=str,
                ),

            "optimizer_params":
                json.dumps(
                    normalized_dict(
                        optimizer_params
                    ),
                    default=str,
                ),

            "pipeline_params":
                json.dumps(
                    normalized_dict(
                        pipeline_params
                    ),
                    default=str,
                ),

            "status":
                "success",
        })


    except Exception as exc:

        runtime = time.perf_counter() - start

        print(
            f"\nERROR for {optimizer_name}:"
        )

        print(exc)

        records.append({
            "optimizer":
                optimizer_name,

            "optimizer_best_objective":
                None,

            "pipeline_best_objective":
                None,

            "objective_difference":
                None,

            "threshold":
                None,

            "runtime_seconds":
                runtime,

            "optimizer_trial_seed":
                None,

            "pipeline_trial_seed":
                None,

            "optimizer_param_types":
                None,

            "pipeline_param_types":
                None,

            "optimizer_params":
                None,

            "pipeline_params":
                None,

            "status":
                f"error: {exc}",
        })


# ==========================================================
# 6. Save results
# ==========================================================

results_df = pd.DataFrame(records)

results_df.to_csv(
    CSV_PATH,
    index=False,
)


# ==========================================================
# 7. Human-readable comparison
# ==========================================================

print("\n")
print("=" * 75)
print("AUDIT RESULTS")
print("=" * 75)

display_columns = [
    "optimizer",
    "optimizer_best_objective",
    "pipeline_best_objective",
    "objective_difference",
    "optimizer_trial_seed",
    "pipeline_trial_seed",
    "runtime_seconds",
    "status",
]

print(
    results_df[
        display_columns
    ].to_string(index=False)
)


# ==========================================================
# 8. Cross-backend consistency checks
# ==========================================================

successful = results_df[
    results_df["status"] == "success"
].copy()

print("\n")
print("=" * 75)
print("CONSISTENCY CHECKS")
print("=" * 75)


if not successful.empty:

    print("\nOptimizer-reported objectives:")
    print(
        successful[
            [
                "optimizer",
                "optimizer_best_objective",
            ]
        ].to_string(index=False)
    )

    print("\nPipeline objectives:")
    print(
        successful[
            [
                "optimizer",
                "pipeline_best_objective",
            ]
        ].to_string(index=False)
    )


    optimizer_unique = (
        successful[
            "optimizer_best_objective"
        ]
        .round(12)
        .nunique()
    )

    pipeline_unique = (
        successful[
            "pipeline_best_objective"
        ]
        .round(12)
        .nunique()
    )


    print(
        "\nNumber of unique "
        "optimizer objectives:",
        optimizer_unique,
    )

    print(
        "Number of unique "
        "pipeline objectives:",
        pipeline_unique,
    )


    if optimizer_unique == 1:

        print(
            "\nPASS: all optimizers report "
            "the same objective for the "
            "fixed configuration."
        )

    else:

        print(
            "\nWARNING: optimizer-reported "
            "objectives differ even though "
            "the configuration is fixed."
        )


    if pipeline_unique == 1:

        print(
            "PASS: all materialized pipelines "
            "have the same objective."
        )

    else:

        print(
            "WARNING: materialized pipeline "
            "objectives differ across backends."
        )


# ==========================================================
# 9. Save text summary
# ==========================================================

with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "Experiment: "
        "E05a - Objective Consistency Audit\n"
    )

    f.write(
        "Dataset: IMS\n"
    )

    f.write(
        "Optimization sources: 1 + 2\n"
    )

    f.write(
        "Metric: AD1_AUC\n"
    )

    f.write(
        "Fixed configuration:\n"
    )

    f.write(
        json.dumps(
            FIXED_SPACE,
            indent=2,
        )
    )

    f.write(
        "\n\nRESULTS\n"
    )

    f.write(
        results_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    f.write(
        "\n\nPARAMETER TYPES\n"
    )

    f.write(
        results_df[
            [
                "optimizer",
                "optimizer_param_types",
                "pipeline_param_types",
            ]
        ].to_string(
            index=False
        )
    )


print("\n")
print("=" * 75)
print("FILES SAVED")
print("=" * 75)

print(CSV_PATH)
print(SUMMARY_PATH)