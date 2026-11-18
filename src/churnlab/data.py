import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

CSV_PATH = Path("data/raw/telco.csv")
SPLIT_PATH = Path("evidence/tp02/split_ids.json")

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = ["Contract", "InternetService", "PaymentMethod"]
FEATURES = NUMERIC + CATEGORICAL
SEED = 42


def load_clean() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    # TotalCharges contient des blancs : conversion en NaN (imputé plus tard
    # dans le pipeline, donc appris sur le train uniquement)
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype("string").str.strip(), errors="coerce"
    )
    df["target"] = (df["Churn"] == "Yes").astype(int)
    return df


def make_split(df: pd.DataFrame):
    # 60 / 20 / 20 stratifié sur la cible
    train_df, rest_df = train_test_split(
        df, test_size=0.4, stratify=df["target"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        rest_df, test_size=0.5, stratify=rest_df["target"], random_state=SEED
    )
    return train_df, val_df, test_df


def save_split(train_df, val_df, test_df) -> None:
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_PATH.write_text(json.dumps({
        "seed": SEED,
        "train": sorted(train_df["customerID"]),
        "validation": sorted(val_df["customerID"]),
        "test": sorted(test_df["customerID"]),
    }, indent=1), encoding="utf-8")