# src/data/preprocess.py
import polars as pl

def preprocess_dataset(df: pl.DataFrame) -> pl.DataFrame:
    """
    Clean & cast columns. Polars is extremely fast here.
    """

    numeric_cols = [
        "wages_monthly_usd",
        "wages_hourly_usd",
        "emp",
        "unem_people",
        "whours",
        "pop",
    ]

    # Cast numeric columns
    for col in numeric_cols:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))

    # Fix year
    if "year" in df.columns:
        df = df.with_columns(pl.col("year").cast(pl.Int32, strict=False))

    # Drop rows missing essential dimensions
    df = df.drop_nulls(subset=["countryname", "year", "gender"], maintain_order=True)

    return df


def save_to_parquet(df: pl.DataFrame, path: str):
    """
    Save cleaned df to Parquet (fast binary format).
    """
    df.write_parquet(path)
