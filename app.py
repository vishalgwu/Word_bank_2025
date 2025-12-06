import os
import sys
import streamlit as st
import polars as pl
from dotenv import load_dotenv
from loguru import logger

load_dotenv()
from src.rag.llm import call_llm

# ---------------------------------
# FIX IMPORT PATH
# ---------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ---------------------------------
# IMPORT MODULES
# ---------------------------------
from src.data.loader import load_dataset
from src.data.preprocess import preprocess_dataset, save_to_parquet
from src.data.filters import get_filter_options
from src.viz.charts import (
    plot_wage_distribution,
    plot_wage_by_education,
    plot_gender_wage_box,
    plot_gender_industry_share,
    plot_industry_employment,
    plot_age_group_employment,
    plot_region_wage_comparison,
)

# RAG Modules
from src.rag.summarize import generate_summaries
from src.rag.ingest import build_qdrant_index
from src.rag.retrieve import retrieve_relevant
from src.rag.llm import generate_answer


# ---------------------------------
# CONFIG
# ---------------------------------
CSV_PATH = "JOIN_Benchmarking_Data_2023_10_04.csv"
PARQUET_PATH = "cleaned_data.parquet"


# ---------------------------------
# LOAD DATA WITH POLARS
# ---------------------------------
@st.cache_data(show_spinner=True)
def load_or_create_clean_data():
    if os.path.exists(PARQUET_PATH):
        return pl.read_parquet(PARQUET_PATH)

    df = load_dataset(CSV_PATH)
    clean_df = preprocess_dataset(df)
    save_to_parquet(clean_df, PARQUET_PATH)
    return clean_df


# ---------------------------------
# MAIN APP
# ---------------------------------
st.set_page_config(page_title="Word Bank Dashboard", layout="wide")

st.title("Word Bank Analytics Dashboard")

df = load_or_create_clean_data()
filters = get_filter_options(df)


# ----------- Helper: add "All" option ----------
def add_all(values):
    return ["All"] + values


countries   = add_all(filters["countries"])
years       = add_all(filters["years"])
genders     = add_all(filters["genders"])
age_groups  = add_all(filters["age_groups"])
urbs        = add_all(filters["urban_rural"])
industries  = add_all(filters["industries"])
occupations = add_all(filters["occupations"])
educations  = add_all(filters["education"])
regions     = add_all(filters["regions"])


# ---------------------------------
# SIDEBAR FILTER UI
# ---------------------------------
st.sidebar.header("🔍 Filter Data")

country    = st.sidebar.selectbox("Country", countries)
year       = st.sidebar.selectbox("Year", years)
gender     = st.sidebar.selectbox("Gender", genders)
age_group  = st.sidebar.selectbox("Age Group", age_groups)
urb        = st.sidebar.selectbox("Urban / Rural", urbs)
industry   = st.sidebar.selectbox("Industry", industries)
occupation = st.sidebar.selectbox("Occupation", occupations)
education  = st.sidebar.selectbox("Education", educations)
region     = st.sidebar.selectbox("Region", regions)


# ---------------------------------
# APPLY FILTERS
# ---------------------------------
filtered = df

if country != "All": filtered = filtered.filter(pl.col("countryname") == country)
if year != "All": filtered = filtered.filter(pl.col("year") == year)
if gender != "All": filtered = filtered.filter(pl.col("gender") == gender)
if age_group != "All": filtered = filtered.filter(pl.col("age_group") == age_group)
if urb != "All": filtered = filtered.filter(pl.col("urb") == urb)
if industry != "All": filtered = filtered.filter(pl.col("industry") == industry)
if occupation != "All": filtered = filtered.filter(pl.col("occup") == occupation)
if education != "All": filtered = filtered.filter(pl.col("edulevelsel") == education)
if region != "All": filtered = filtered.filter(pl.col("regionname") == region)

st.success("Filters applied successfully!")

filtered_pdf = filtered.to_pandas()


# ---------------------------------
# DATA PREVIEW
# ---------------------------------
st.subheader("📊 Filtered Data Preview")

if filtered_pdf.empty:
    st.info("No data available for these filters.")
else:
    st.dataframe(filtered_pdf.head(50))


# ----------------------------------------------------
#                TABS
# ----------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Wage Analysis",
    "🧑‍🤝‍🧑 Gender",
    "🎓 Education",
    "🏭 Industry",
    "👥 Age Groups",
    "🌍 Regions",
    "🤖 Ask the Dashboard"
])


# ---------------- TAB 1: WAGES ---------------------
with tab1:
    st.header("📈 Wage Analysis")

    if not filtered_pdf.empty:
        fig1 = plot_wage_distribution(filtered_pdf)
        if fig1: st.plotly_chart(fig1, width="stretch", key="wage_dist")

        fig2 = plot_wage_by_education(filtered_pdf)
        if fig2: st.plotly_chart(fig2, width="stretch", key="wage_edu")
    else:
        st.info("No wage data available.")


# ---------------- TAB 2: GENDER ---------------------
with tab2:
    st.header("🧑‍🤝‍🧑 Gender Comparison")

    if not filtered_pdf.empty:
        fig3 = plot_gender_wage_box(filtered_pdf)
        if fig3: st.plotly_chart(fig3, width="stretch", key="gender_box")

        fig4 = plot_gender_industry_share(filtered_pdf)
        if fig4: st.plotly_chart(fig4, width="stretch", key="gender_industry")
    else:
        st.info("No gender data available.")


# ---------------- TAB 3: EDUCATION ---------------------
with tab3:
    st.header("🎓 Education Insights")

    if not filtered_pdf.empty:
        fig_edu = plot_wage_by_education(filtered_pdf)
        if fig_edu: st.plotly_chart(fig_edu, width="stretch", key="edu_chart")
    else:
        st.info("No education insights available.")


# ---------------- TAB 4: INDUSTRY ---------------------
with tab4:
    st.header("🏭 Industry Insights")

    if not filtered_pdf.empty:
        fig5 = plot_industry_employment(filtered_pdf)
        if fig5: st.plotly_chart(fig5, width="stretch", key="industry_chart")
    else:
        st.info("No industry data available.")


# ---------------- TAB 5: AGE GROUPS ---------------------
with tab5:
    st.header("👥 Age Group Analysis")

    if not filtered_pdf.empty:
        fig6 = plot_age_group_employment(filtered_pdf)
        if fig6: st.plotly_chart(fig6, width="stretch", key="age_chart")
    else:
        st.info("No age group data available.")


# ---------------- TAB 6: REGIONS ---------------------
with tab6:
    st.header("🌍 Regional Comparison")

    if not filtered_pdf.empty:
        fig7 = plot_region_wage_comparison(filtered_pdf)
        if fig7: st.plotly_chart(fig7, width="stretch", key="region_chart")
    else:
        st.info("No regional insights available.")


