# src/rag/summarize.py
import polars as pl


def generate_summaries(df: pl.DataFrame) -> list[dict]:
    """
    Create text summaries for (country, year, gender) groups.
    Adjust as you like – this is just a starting point.
    """

    groups = ["countryname", "year", "gender"]

    grouped = df.group_by(groups).agg([
        pl.col("wages_monthly_usd").mean().alias("avg_wage"),
        pl.col("emp").sum().alias("total_employment"),
        pl.col("industry").mode().alias("top_industry"),
        pl.col("edulevelsel").mode().alias("top_education"),
        pl.col("regionname").mode().alias("region"),
    ])

    summaries: list[dict] = []

    for row in grouped.iter_rows(named=True):
        text = (
            f"In {row['countryname']} in {row['year']} for {row['gender']} workers, "
            f"the average monthly wage was {row['avg_wage']:.2f} USD. "
            f"Total employment in the sample was {row['total_employment']}. "
            f"The most common industry was {row['top_industry']}, "
            f"with typical education level {row['top_education']}. "
            f"Region: {row['region']}."
        )

        summaries.append(
            {
                "id": f"{row['countryname']}_{row['year']}_{row['gender']}",
                "text": text,
            }
        )

    return summaries
