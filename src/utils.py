# src/utils.py

import pandas as pd
from sklearn.model_selection import train_test_split


def load_feature_dataset(path: str = "data/features/feature_dataset.csv"):
    """
    Load the feature dataset produced in Phase 2.
    Returns the full DataFrame.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    print(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    return df


# These are the 9 features the model will actually see.
# Meta columns (date, home_team, away_team, tournament) are excluded.
FEATURE_COLS = [
    "home_fifa_points",
    "away_fifa_points",
    "fifa_points_diff",
    "home_form",
    "away_form",
    "h2h_home_win_rate",
    "tournament_weight",
    "home_avg_goal_diff",
    "away_avg_goal_diff",
]
LABEL_COL = "label"


def make_splits(df: pd.DataFrame):
    """
    Split the dataset into train, validation, and test sets.

    Test set  → 2018 and 2022 FIFA World Cup matches only.
                These are real high-stakes matches — a perfect held-out test.
    Train/Val → Everything else, split 85/15 with stratification so each
                split keeps the same class ratio (29% away / 23% draw / 48% home).

    Why stratified split?
        Our labels are imbalanced (home wins = 48%). Without stratification,
        random splits could accidentally put more draws in validation, making
        metrics misleading. Stratification guarantees each split mirrors
        the full dataset's distribution.

    Returns:
        X_train, X_val, X_test  — NumPy arrays of shape (N, 9)
        y_train, y_val, y_test  — NumPy arrays of shape (N,)
        df_test                 — The raw test rows (useful for inspection)
    """
    # --- Test set: 2018 & 2022 World Cup matches ---
    wc_mask = (
        df["tournament"].str.contains("FIFA World Cup", case=False, na=False)
        & df["date"].dt.year.isin([2018, 2022])
    )
    df_test = df[wc_mask].copy()
    df_trainval = df[~wc_mask].copy()

    print(f"\nTest set  (WC 2018+2022): {len(df_test):,} matches")
    print(f"Train+Val (everything else): {len(df_trainval):,} matches")

    X = df_trainval[FEATURE_COLS].values
    y = df_trainval[LABEL_COL].values

    # 85% train, 15% validation — stratified on label
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.15,
        random_state=42,
        stratify=y          # ← keeps class ratios equal in both splits
    )

    X_test = df_test[FEATURE_COLS].values
    y_test = df_test[LABEL_COL].values

    print(f"\nSplit sizes:")
    print(f"  Train : {X_train.shape[0]:,} rows")
    print(f"  Val   : {X_val.shape[0]:,} rows")
    print(f"  Test  : {X_test.shape[0]:,} rows")

    # Verify class distribution is preserved in train
    import numpy as np
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"\nTrain label distribution: { {int(k): f'{v/len(y_train):.1%}' for k, v in zip(unique, counts)} }")

    return X_train, X_val, X_test, y_train, y_val, y_test, df_test