import pandas as pd
import plotly.express as px


def _safe_subset(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Return subset where column is non-null and DataFrame has > 0 rows."""
    if col not in df.columns:
        return df.iloc[0:0]  # empty
    return df[df[col].notna()]


# ---------------------------
# WAGE ANALYSIS
# ---------------------------
def plot_wage_distribution(df: pd.DataFrame):
    df_wage = _safe_subset(df, "wages_monthly_usd")
    if df_wage.empty:
        return None

    fig = px.histogram(
        df_wage,
        x="wages_monthly_usd",
        color="gender" if "gender" in df_wage.columns else None,
        nbins=40,
        marginal="box",
        title="Wage Distribution (Monthly USD)",
    )
    fig.update_layout(xaxis_title="Monthly Wages (USD)", yaxis_title="Count")
    return fig


def plot_wage_by_education(df: pd.DataFrame):
    if "edulevelsel" not in df.columns or "wages_monthly_usd" not in df.columns:
        return None

    grouped = (
        df.dropna(subset=["edulevelsel", "wages_monthly_usd"])
        .groupby("edulevelsel", as_index=False)["wages_monthly_usd"]
        .mean()
        .sort_values("wages_monthly_usd", ascending=False)
    )

    if grouped.empty:
        return None

    fig = px.bar(
        grouped,
        x="edulevelsel",
        y="wages_monthly_usd",
        title="Average Monthly Wage by Education Level",
    )
    fig.update_layout(
        xaxis_title="Education Level",
        yaxis_title="Avg Monthly Wage (USD)",
        xaxis_tickangle=-30,
    )
    return fig


# ---------------------------
# GENDER COMPARISON
# ---------------------------
def plot_gender_wage_box(df: pd.DataFrame):
    if "gender" not in df.columns or "wages_monthly_usd" not in df.columns:
        return None

    df_wage = df.dropna(subset=["gender", "wages_monthly_usd"])
    if df_wage.empty:
        return None

    fig = px.box(
        df_wage,
        x="gender",
        y="wages_monthly_usd",
        title="Monthly Wages by Gender",
    )
    fig.update_layout(
        xaxis_title="Gender",
        yaxis_title="Monthly Wages (USD)",
    )
    return fig


def plot_gender_industry_share(df: pd.DataFrame):
    if "gender" not in df.columns or "industry" not in df.columns:
        return None

    grouped = (
        df.dropna(subset=["gender", "industry"])
        .groupby(["industry", "gender"], as_index=False)
        .size()
    )
    if grouped.empty:
        return None

    fig = px.bar(
        grouped,
        x="industry",
        y="size",
        color="gender",
        barmode="group",
        title="Gender Distribution across Industries",
    )
    fig.update_layout(
        xaxis_title="Industry",
        yaxis_title="Number of Records",
        xaxis_tickangle=-30,
    )
    return fig


# ---------------------------
# INDUSTRY INSIGHTS
# ---------------------------
def plot_industry_employment(df: pd.DataFrame):
    # Prefer 'emp' column to represent employment if present
    value_col = "emp" if "emp" in df.columns else None
    if "industry" not in df.columns:
        return None

    data = df.dropna(subset=["industry"])
    if data.empty:
        return None

    if value_col and value_col in df.columns:
        grouped = (
            data.groupby("industry", as_index=False)[value_col]
            .sum()
            .sort_values(value_col, ascending=False)
        )
        y_col = value_col
        y_title = "Employment (sum of emp)"
    else:
        grouped = (
            data.groupby("industry", as_index=False)
            .size()
            .sort_values("size", ascending=False)
        )
        y_col = "size"
        y_title = "Number of Records"

    fig = px.bar(
        grouped.head(15),
        x="industry",
        y=y_col,
        title="Top Industries by Employment",
    )
    fig.update_layout(
        xaxis_title="Industry",
        yaxis_title=y_title,
        xaxis_tickangle=-30,
    )
    return fig


# ---------------------------
# AGE GROUP ANALYSIS
# ---------------------------
def plot_age_group_employment(df: pd.DataFrame):
    if "age_group" not in df.columns:
        return None

    value_col = "emp" if "emp" in df.columns else None
    data = df.dropna(subset=["age_group"])
    if data.empty:
        return None

    if value_col and value_col in df.columns:
        grouped = (
            data.groupby("age_group", as_index=False)[value_col]
            .sum()
            .sort_values(value_col, ascending=False)
        )
        y_col = value_col
        y_title = "Employment (sum of emp)"
    else:
        grouped = (
            data.groupby("age_group", as_index=False)
            .size()
            .sort_values("size", ascending=False)
        )
        y_col = "size"
        y_title = "Number of Records"

    fig = px.bar(
        grouped,
        x="age_group",
        y=y_col,
        title="Employment by Age Group",
    )
    fig.update_layout(
        xaxis_title="Age Group",
        yaxis_title=y_title,
        xaxis_tickangle=-20,
    )
    return fig


# ---------------------------
# REGIONAL ANALYSIS
# ---------------------------
def plot_region_wage_comparison(df: pd.DataFrame):
    if "regionname" not in df.columns or "wages_monthly_usd" not in df.columns:
        return None

    grouped = (
        df.dropna(subset=["regionname", "wages_monthly_usd"])
        .groupby("regionname", as_index=False)["wages_monthly_usd"]
        .mean()
        .sort_values("wages_monthly_usd", ascending=False)
    )

    if grouped.empty:
        return None

    fig = px.bar(
        grouped,
        x="regionname",
        y="wages_monthly_usd",
        title="Average Monthly Wage by Region",
    )
    fig.update_layout(
        xaxis_title="Region",
        yaxis_title="Avg Monthly Wage (USD)",
        xaxis_tickangle=-20,
    )
    return fig
