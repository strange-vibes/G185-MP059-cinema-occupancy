import pandas as pd

from src.cinema_occupancy.cleaning import clean_data


def sample_data():
    return pd.DataFrame({
        "record_id": [1, 2, 3],
        "ticket_price_rs": [200, 999, 300],
        "days_since_release": [10, 20, 30],
        "star_rating": [4.0, 3.5, 4.5],
        "show_hour": [12, 15, 18],
        "screen_seats": [100, 150, 200],
        "distance_to_metro_km": [2.0, 5000.0, 3.0],
        "language": [" Hindi ", "English", "Hindi"],
        "location": ["Mall", "Mall", "City"],
        "tickets_scanned_pct": [80, 90, 70],
        "occupancy_pct": [50, 600, 75],
    })


def test_leakage_column_removed():
    df = sample_data()
    cleaned = clean_data(df)

    assert "tickets_scanned_pct" not in cleaned.columns


def test_sentinel_price_replaced():
    df = sample_data()
    cleaned = clean_data(df)

    assert 999 not in cleaned["ticket_price_rs"].values
    assert cleaned["ticket_price_rs"].isna().sum() == 0


def test_mall_distance_corrected():
    df = sample_data()
    cleaned = clean_data(df)

    # Mall distance 5000 should become 5
    assert cleaned.loc[1, "distance_to_metro_km"] == 5.0


def test_occupancy_typo_corrected():
    df = sample_data()
    cleaned = clean_data(df)

    # 600 was accidentally multiplied by 10
    assert cleaned.loc[1, "occupancy_pct"] == 60.0


def test_duplicate_rows_removed():
    df = sample_data()

    # Add one duplicate row
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    cleaned = clean_data(df)

    assert len(cleaned) == 3
    