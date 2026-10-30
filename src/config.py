"""Project-wide settings. Change values here, not inside the other modules."""
from pathlib import Path

# ---- folders ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

# ---- files -----------------------------------------------------------------
DATASET_PATH = DATA_DIR / "dataset_5000.csv"      # 500 original + 4,500 synthetic rows
ORIGINAL_PATH = DATA_DIR / "dataset.csv"          # the 500 original rows
DICTIONARY_PATH = DATA_DIR / "data_dictionary.csv"
METRICS_PATH = REPORTS_DIR / "metrics.json"
COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"

# ---- columns ---------------------------------------------------------------
ID_COL = "patient_id"
TARGET = "label"                                   # 1 = disease, 0 = no disease
FEATURES = ["age", "bmi", "blood_pressure", "glucose", "cholesterol", "smoking"]
NUMERIC_FEATURES = ["age", "bmi", "blood_pressure", "glucose", "cholesterol"]
BINARY_FEATURES = ["smoking"]
N_ORIGINAL_ROWS = 500                              # P0001-P0500 are the original rows

FEATURE_LABELS = {
    "age": "Age (years)",
    "bmi": "BMI (kg/m²)",
    "blood_pressure": "Blood pressure (mmHg)",
    "glucose": "Glucose (mg/dL)",
    "cholesterol": "Cholesterol (mg/dL)",
    "smoking": "Smoker (1 = yes)",
}

# Allowed input ranges (same as the data the models were trained on)
VALID_RANGES = {
    "age": (25, 75),
    "bmi": (18.0, 40.0),
    "blood_pressure": (60, 140),
    "glucose": (70, 200),
    "cholesterol": (150, 280),
    "smoking": (0, 1),
}

# ---- experiment settings ---------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
SEARCH_ITERATIONS = 20          # random-search draws for Random Forest and XGBoost
TARGET_RECALL = 0.85            # screening threshold keeps at least this recall

DISCLAIMER = (
    "This tool is a student research project built on partly synthetic data. "
    "It must NOT be used for clinical diagnosis or medical decisions without "
    "supervision by qualified medical professionals."
)
