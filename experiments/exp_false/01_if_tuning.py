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


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "ims.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. Load IMS dataset
# --------------------------------------------------

print("\n=== Loading IMS dataset ===")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# --------------------------------------------------
# 3. Create PdMLabs Dataset
# --------------------------------------------------

print("\n=== Creating PdMLabs Dataset ===")

dataset_obj = Dataset(
    data=df,
    datetime_column="Artificial_timestamp",
    event_indicator="event",
    source_column="source",

    # Same split as the official demo
    train_sources=["1"],
    val_sources=["2"],
    test_sources=["3"],
)

train_data, test_data = dataset_obj.get_unsupervised_dataset()

print(f"Predictive horizon: {dataset_obj.predictive_horizon}")
print(f"Max wait time: {dataset_obj.max_wait_time}")

print(f"Optimization sources: {train_data['target_sources']}")
print(f"Test sources: {test_data['target_sources']}")


# --------------------------------------------------
# 4. Method
# --------------------------------------------------

methods = [
    IsolationForestUnsupervised
]

method_names = [
    "IsolationForestUnsupervised"
]


# --------------------------------------------------
# 5. Hyperparameter search space
# --------------------------------------------------

param_spaces = [
    {
        # Sliding-window parameters
        "window": [40, 80, 160],
        "slide": [0.33, 0.5, 1.0],

        # Keep score aggregation fixed for this experiment
        "policy": ["or"],

        # Isolation Forest parameters
        "n_estimators": [50, 100, 200],

        # <= smallest window so every configuration is meaningful
        "max_samples": [20, 40],

        # Keep these fixed for the first controlled experiment
        "max_features": [1.0],
        "bootstrap": [False],

        # Reproducible Isolation Forest
        "random_state": [42],
    }
]


# --------------------------------------------------
# 6. Run tuning
# --------------------------------------------------

print("\n=== Starting Isolation Forest tuning ===")
print("Optimizer: Mango")
print("Objective: AD1_AUC")
print("Maximum configurations evaluated: 8")

start_time = time.perf_counter()

best_params = run_experiment(
    dataset=train_data,

    methods=methods,
    param_space_dict_per_method=param_spaces,
    method_names=method_names,

    experiments=[UnsupervisedPdMExperiment],
    experiment_names=["Thesis_IF_Tuning_01"],

    MAX_RUNS=8,
    MAX_JOBS=1,
    INITIAL_RANDOM=2,

    optimization_param="AD1_AUC",
    maximize=True,

    optimizer="mango",

    random_state=42,
    deterministic=True,

    log_best_pipeline=True,
)

elapsed = time.perf_counter() - start_time


# --------------------------------------------------
# 7. Results
# --------------------------------------------------

print("\n====================================")
print("       TUNING FINISHED")
print("====================================")

print("\nBest result:")
print(best_params)

print(f"\nTotal runtime: {elapsed:.2f} seconds")
print(f"Total runtime: {elapsed / 60:.2f} minutes")


# --------------------------------------------------
# 8. Save small experiment summary
# --------------------------------------------------

summary_path = RESULTS_DIR / "01_if_tuning_summary.txt"

with open(summary_path, "w", encoding="utf-8") as f:
    f.write("Experiment: Thesis_IF_Tuning_01\n")
    f.write("Dataset: IMS\n")
    f.write("Method: IsolationForestUnsupervised\n")
    f.write("Optimizer: Mango\n")
    f.write("Objective: AD1_AUC\n")
    f.write("MAX_RUNS: 8\n")
    f.write(f"Runtime_seconds: {elapsed:.2f}\n")
    f.write("\nBest result:\n")
    f.write(str(best_params))

print(f"\nSummary saved to:")
print(summary_path)