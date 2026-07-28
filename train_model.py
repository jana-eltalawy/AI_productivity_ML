"""
train_model.py
---------------
Reproduces the preprocessing + modeling pipeline from `User_AI_prediction.ipynb`
and saves everything the FastAPI service needs to make predictions:

    - model.joblib             -> trained RandomForestRegressor
    - scaler.joblib            -> fitted StandardScaler
    - feature_columns.joblib   -> exact column order used at training time
                                   (after one-hot encoding) so new requests
                                   can be aligned to the same shape.

Run this once (locally) whenever you retrain:

    python train_model.py --csv user_level_ai_adoption.csv

It will create an `artifacts/` folder containing the three files above.
Copy that folder next to `main.py` before starting the API.
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---- these must stay in sync with main.py -----------------------------
CAT_FEATURES = [
    "Industry",
    "Job_Role",
    "Location",
    "Primary_AI_Tool",
    "Experience_Level",
]
NUM_FEATURES = [
    "Daily_Token_Usage",
    "Tasks_Automated_Per_Week",
    "Tokens_per_Task",
]
TARGET = "Productivity_Gain_Percent"
EXPERIENCE_BINS = [0, 5, 10, 20, 25]
EXPERIENCE_LABELS = ["Junior", "Mid", "Senior", "Expert"]
# -------------------------------------------------------------------------


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Same feature engineering steps as the notebook."""
    df = df.copy()

    if "User_ID" in df.columns:
        df = df.drop(columns=["User_ID"])

    df["Experience_Level"] = pd.cut(
        df["Experience_Years"],
        bins=EXPERIENCE_BINS,
        labels=EXPERIENCE_LABELS,
    )

    df["Tokens_per_Task"] = df["Daily_Token_Usage"] / df["Tasks_Automated_Per_Week"]

    # Columns explored in the notebook but not used by the final model
    drop_cols = [c for c in ["Experience_Years", "Adoption_Date", "Productivity_class"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    return df


def main(csv_path: str, out_dir: str):
    df = pd.read_csv(csv_path)
    df = engineer_features(df)

    x = df[CAT_FEATURES + NUM_FEATURES]
    y = df[TARGET]

    x = pd.get_dummies(x, columns=CAT_FEATURES, drop_first=True)
    feature_columns = x.columns.tolist()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=7,
        min_samples_leaf=2,
        min_samples_split=2,
        random_state=42,
    )
    model.fit(x_train_scaled, y_train)

    y_pred = model.predict(x_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("Random Forest Regressor")
    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")

    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(out_dir, "model.joblib"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler.joblib"))
    joblib.dump(feature_columns, os.path.join(out_dir, "feature_columns.joblib"))
    print(f"\nSaved model, scaler and feature_columns to '{out_dir}/'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="user_level_ai_adoption.csv", help="Path to training CSV")
    parser.add_argument("--out", default="artifacts", help="Output directory for saved artifacts")
    args = parser.parse_args()
    main(args.csv, args.out)
