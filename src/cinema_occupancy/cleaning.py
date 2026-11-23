import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the cinema occupancy dataset by fixing identified data issues.
    """

    # Create a copy so the original dataset remains unchanged
    df = df.copy()

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove data leakage column
    if "tickets_scanned_pct" in df.columns:
        df = df.drop(columns=["tickets_scanned_pct"])

    # Fix distance unit mix-up for Mall location
    if "location" in df.columns and "distance_to_metro_km" in df.columns:
        mall_mask = df["location"].astype(str).str.strip().str.lower() == "mall"

        df.loc[mall_mask, "distance_to_metro_km"] = (
            df.loc[mall_mask, "distance_to_metro_km"] / 1000
        )

    # Replace invalid ticket price sentinel value 999 with missing value
    if "ticket_price_rs" in df.columns:
        df.loc[df["ticket_price_rs"] == 999, "ticket_price_rs"] = pd.NA

        # Impute missing ticket prices using median
        median_price = df["ticket_price_rs"].median()

        df["ticket_price_rs"] = df["ticket_price_rs"].fillna(median_price)

    # Fix occupancy target values accidentally multiplied by 10
    if "occupancy_pct" in df.columns:
        mask = df["occupancy_pct"] > 100

        df.loc[mask, "occupancy_pct"] = (
            df.loc[mask, "occupancy_pct"] / 10
        )

    # Remove unnecessary spaces from categorical columns
    categorical_columns = ["language", "location"]

    for column in categorical_columns:
        if column in df.columns:
            df[column] = df[column].astype(str).str.strip()

    return df
    