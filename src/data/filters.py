# src/data/filters.py
import polars as pl

def get_filter_options(df: pl.DataFrame) -> dict:
    """
    Extract unique values for sidebar filters.
    Polars unique() is extremely fast.
    """
    def unique_list(col):
        return df[col].drop_nulls().unique().sort().to_list() if col in df.columns else []

    return {
        "countries": unique_list("countryname"),
        "years": unique_list("year"),
        "genders": unique_list("gender"),
        "age_groups": unique_list("age_group"),
        "urban_rural": unique_list("urb"),
        "industries": unique_list("industry"),
        "occupations": unique_list("occup"),
        "education": unique_list("edulevelsel"),
        "regions": unique_list("regionname"),
    }
