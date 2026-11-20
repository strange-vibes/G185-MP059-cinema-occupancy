
from pathlib import Path
import pandas as pd


# ==========================================
# 1. DATASET PATH
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = PROJECT_ROOT / "Data" / "raw" / "train.csv"


# ==========================================
# 2. LOAD DATASET
# ==========================================

def load_data():
    """Load the original training dataset."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    return pd.read_csv(DATASET_PATH)


# ==========================================
# 3. DATA ANALYSIS
# ==========================================

def analyze_data():

    df = load_data()

    print("\n========== DATASET OVERVIEW ==========")
    print("Shape:", df.shape)
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    # First five rows
    print("\n========== FIRST FIVE ROWS ==========")
    print(df.head())

    # Data types
    print("\n========== DATA TYPES ==========")
    print(df.dtypes)

    # Dataset information
    print("\n========== DATASET INFORMATION ==========")
    df.info()

    # Statistical summary
    print("\n========== STATISTICAL SUMMARY ==========")
    print(df.describe(include="all"))

    # Missing values
    print("\n========== MISSING VALUES ==========")
    print(df.isnull().sum())

    print("Total missing values:", df.isnull().sum().sum())

    # Duplicate records
    print("\n========== DUPLICATE RECORDS ==========")
    print("Duplicate rows:", df.duplicated().sum())

    # Unique categorical values
    print("\n========== UNIQUE CATEGORICAL VALUES ==========")

    for column in df.select_dtypes(include="object").columns:
        print(f"\n{column}:")
        print(df[column].value_counts().head(10))

    # Numerical columns
    numerical_columns = df.select_dtypes(
        include="number"
    ).columns

    # Outlier detection using IQR
    print("\n========== OUTLIER DETECTION ==========")

    for column in numerical_columns:

        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = df[
            (df[column] < lower) |
            (df[column] > upper)
        ]

        print(f"{column}: {len(outliers)} potential outliers")

    # Negative values
    print("\n========== NEGATIVE VALUE CHECK ==========")

    for column in numerical_columns:

        negative_count = (df[column] < 0).sum()

        if negative_count > 0:
            print(f"{column}: {negative_count} negative values")

    # Occupancy validation
    print("\n========== OCCUPANCY VALIDATION ==========")

    if "occupancy_pct" in df.columns:

        print(
            "Occupancy below 0:",
            (df["occupancy_pct"] < 0).sum()
        )

        print(
            "Occupancy above 100:",
            (df["occupancy_pct"] > 100).sum()
        )

    # Constant columns
    print("\n========== CONSTANT COLUMNS ==========")

    constant_columns = df.columns[df.nunique(dropna=False) == 1]

    print(list(constant_columns))

    # Maximum numerical values
    print("\n========== MAXIMUM NUMERICAL VALUES ==========")
    print(df[numerical_columns].max())

    # Minimum numerical values
    print("\n========== MINIMUM NUMERICAL VALUES ==========")
    print(df[numerical_columns].min())

    print("\n========== ANALYSIS COMPLETED ==========")


# ==========================================
# 4. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    analyze_data()
