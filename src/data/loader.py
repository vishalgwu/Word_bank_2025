# src/data/loader.py
import polars as pl

def load_dataset(path: str) -> pl.DataFrame:
    """
    Load CSV using Polars for extreme speed.
    """
    return pl.read_csv(path, infer_schema_length=50000)
