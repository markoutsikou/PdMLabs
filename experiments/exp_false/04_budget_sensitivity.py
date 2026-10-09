
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


# ==========================================
# 1. Experiment configuration
# ==========================================

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "ims.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_CSV = RESULTS_DIR / "04_budget_sensitivity.csv"

# Initially run budgets 4 and 8.
# Larger budgets can be run later.
BUDGETS = (
    [int(x) for x in sys.argv[1:]]
    if len(sys.argv) > 1
    else [4, 8]
)

SEED = 42

if any(budget < 3 or budget > 54 for budget in BUDGETS):
    raise ValueError("Budgets must be between 3 and 54.")


# ==========================================
# 2. Load IMS dataset
# ==========================================

print("Loading IMS dataset...")

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


# ==========================================
# 3. Hyperparameter search space
# Same as E01
# ==========================================

SEARCH_SPACE = {
    "window": [40, 80, 160],
    "slide": [0.33, 0.5, 1.0],
    "policy": ["or"],
    "n_estimators": [50, 100, 200],
    "max_samples": [20, 40],
    "max_features": [1.0],
    "bootstrap": [False],
    "random_state": [SEED],
}


# ==========================================
# 4. Load existing results, if any
# ==========================================

if OUTPUT_CSV.exists():
    results_df = pd.read_csv(OUTPUT_CSV)
else:
    results_df = pd.DataFrame()


# ==========================================
# 5. Run different optimization budgets
# ==========================================

for budget in BUDGETS:

    if not results_df.empty:
        already_done = (
            (results_df["budget"] == budget)
            & (results_df["seed"] == SEED)
        ).any()

        if already_done:
            print(f"Budget {budget} already completed. Skipping.")
            continue

    print("\n" + "=" * 55)
    print(f"E04 - MANGO BUDGET = {budget}")
    print("=" * 55)

    start_time = time.perf_counter()

    result = run_experiment(
        dataset=optimization_data,
        methods=[IsolationForestUnsupervised],
        param_space_dict_per_method=[SEARCH_SPACE],
        method_names=["IsolationForestUnsupervised"],
        experiments=[UnsupervisedPdMExperiment],
        experiment_names=[f"E04_Mango_Budget_{budget}"],

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

    runtime = time.perf_counter() - start_time

    best = result[0]
    params = best["best_params"]

    row = {
        "budget": budget,
        "seed": SEED,
        "best_validation_ad1_auc": float(
            best["best_objective"]
        ),
        "runtime_seconds": round(runtime, 2),
        "window": int(params["method_window"]),
        "slide": float(params["method_slide"]),
        "n_estimators": int(params["method_n_estimators"]),
        "max_samples": int(params["method_max_samples"]),
        "best_params_json": json.dumps(
            params,
            default=str,
        ),
    }

    results_df = pd.concat(
        [results_df, pd.DataFrame([row])],
        ignore_index=True,
    )

    results_df = results_df.sort_values("budget")

    # Save immediately after each budget
    results_df.to_csv(OUTPUT_CSV, index=False)

    print("\nBest validation AD1_AUC:",
          row["best_validation_ad1_auc"])
    print("Runtime:", row["runtime_seconds"], "seconds")
    print("Best parameters:", params)
    print("Results saved to:", OUTPUT_CSV)


# ==========================================
# 6. Show comparison
# ==========================================

print("\n" + "=" * 60)
print("E04 - FINAL BUDGET COMPARISON")
print("=" * 60)

if not results_df.empty:
    columns = [
        "budget",
        "best_validation_ad1_auc",
        "runtime_seconds",
        "window",
        "slide",
        "n_estimators",
        "max_samples",
    ]

    print(results_df[columns].to_string(index=False))

print("\nExperiment completed.")
