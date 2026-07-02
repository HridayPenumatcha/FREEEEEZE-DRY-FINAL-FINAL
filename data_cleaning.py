"""
data_cleaning.py
=================
Cleans and transforms the raw primary-research questionnaire export
(Questionnaire_Responses sheet) into an analysis-ready dataset for the
Freeze-Dried Home-Cooked Meal Service - Hyderabad Market Validation.

This is the "Data Preparation" step of the individual assignment:
  1. Bracketed survey answers (age group, income, price, quantity, spend)
     are converted to numeric midpoint estimates so they can be used in
     correlation/descriptive statistics.
  2. Multi-select questions (B5, C1, C2, C3) - stored as semicolon-joined
     text in the raw export - are parsed into individual binary indicator
     columns.
  3. Missing values are identified and handled column-by-column with a
     documented, defensible decision (not blanket imputation).
  4. Outliers on the derived numeric fields are detected via the IQR rule
     and flagged (capped version also produced, raw version preserved).

Every transformation is logged in `cleaning_report()` so the steps are
visible and explainable in the report.
"""

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Bracket -> numeric midpoint lookup tables
# --------------------------------------------------------------------------
AGE_MIDPOINT = {
    "Under 25": 22, "25-34": 29.5, "35-44": 39.5, "45-54": 49.5, "55 and above": 60,
}
INCOME_MIDPOINT = {
    "Below Rs 50,000": 40000, "Rs 50,000 - 1,00,000": 75000,
    "Rs 1,00,001 - 2,00,000": 150000, "Rs 2,00,001 - 3,50,000": 275000,
    "Above Rs 3,50,000": 400000, "Prefer not to say": np.nan,
}
HOUSEHOLD_MIDPOINT = {"1-2": 1.5, "3-4": 3.5, "5-6": 5.5, "7 or more": 8}
PRICE_MIDPOINT = {
    "Below Rs 300": 250, "Rs 300-400": 350, "Rs 401-500": 450,
    "Rs 501-600": 550, "Above Rs 600": 650,
}
QTY_MIDPOINT = {"1-2 kg": 1.5, "3-5 kg": 4, "6-10 kg": 8, "More than 10 kg": 12}
SPEND_MIDPOINT = {
    "Below Rs 500": 350, "Rs 500-1,500": 1000, "Rs 1,501-3,000": 2250,
    "Rs 3,001-5,000": 4000, "Above Rs 5,000": 6000,
}

MULTISELECT_COLS = [
    "B5_Methods_Currently_Used", "C1_Dishes_Preferred",
    "C2_Occasions", "C3_Trusted_Channels",
]

NUMERIC_DERIVED_COLS = ["age_numeric", "income_numeric", "household_numeric",
                          "price_numeric", "qty_numeric", "spend_numeric"]


def load_raw(file):
    """Load the raw questionnaire export. Accepts an Excel file (reads the
    Questionnaire_Responses sheet) or a CSV with the same column schema."""
    name = getattr(file, "name", str(file))
    if str(name).lower().endswith(".csv"):
        return pd.read_csv(file)
    try:
        return pd.read_excel(file, sheet_name="Questionnaire_Responses")
    except Exception:
        return pd.read_excel(file)


def _parse_multiselect(series, sep="; "):
    """Turn a semicolon-joined multi-select text column into a dict of
    {option_label: binary_array}."""
    all_options = set()
    parsed = series.fillna("").apply(lambda x: [s.strip() for s in str(x).split(sep) if s.strip()])
    for opts in parsed:
        for o in opts:
            if o and o != "None selected":
                all_options.add(o)
    cols = {}
    for opt in sorted(all_options):
        cols[opt] = parsed.apply(lambda lst: int(opt in lst))
    return cols


def clean_and_transform(df_raw: pd.DataFrame):
    """Returns (cleaned_df, report) where report is a dict logging every
    transformation decision made, for transparency in the write-up."""
    df = df_raw.copy()
    report = {"missing_handled": {}, "brackets_converted": [], "multiselect_parsed": [],
              "outliers_flagged": {}}

    # ---- 1. Standardize text formatting ----
    obj_cols = df.select_dtypes(include="object").columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan})

    # ---- 2. Missing value handling (documented per column) ----
    if "A5_Locality" in df.columns:
        n_missing = df["A5_Locality"].isna().sum()
        df["A5_Locality"] = df["A5_Locality"].fillna("Unknown")
        report["missing_handled"]["A5_Locality"] = f"{n_missing} blank -> filled 'Unknown' (non-analytical field)"

    if "A8_Household_Size" in df.columns:
        n_missing = df["A8_Household_Size"].isna().sum()
        mode_val = df["A8_Household_Size"].mode().iloc[0] if not df["A8_Household_Size"].mode().empty else "3-4"
        df["A8_Household_Size"] = df["A8_Household_Size"].fillna(mode_val)
        report["missing_handled"]["A8_Household_Size"] = f"{n_missing} blank -> filled with mode ('{mode_val}')"

    if "B3_Brand_Used" in df.columns:
        n_missing = df["B3_Brand_Used"].isna().sum()
        df["B3_Brand_Used"] = df["B3_Brand_Used"].fillna("Not Applicable (never used)")
        report["missing_handled"]["B3_Brand_Used"] = f"{n_missing} blank -> 'Not Applicable' (only filled for prior users by design)"

    if "E5_Willing_Pay_Extra_For_Speed" in df.columns:
        n_missing = df["E5_Willing_Pay_Extra_For_Speed"].isna().sum()
        df["E5_Willing_Pay_Extra_For_Speed"] = df["E5_Willing_Pay_Extra_For_Speed"].fillna("Maybe")
        report["missing_handled"]["E5_Willing_Pay_Extra_For_Speed"] = f"{n_missing} blank -> filled 'Maybe' (neutral default)"

    # ---- 3. Bracket -> numeric midpoint conversion ----
    bracket_map = [
        ("A3_Age_Group", "age_numeric", AGE_MIDPOINT),
        ("A7_Income_Bracket", "income_numeric", INCOME_MIDPOINT),
        ("A8_Household_Size", "household_numeric", HOUSEHOLD_MIDPOINT),
        ("E2_Max_Price_Per_Kg_Bracket", "price_numeric", PRICE_MIDPOINT),
        ("E3_Min_Order_Qty_Bracket", "qty_numeric", QTY_MIDPOINT),
        ("E4_Monthly_Spend_Bracket", "spend_numeric", SPEND_MIDPOINT),
    ]
    for src_col, new_col, mapping in bracket_map:
        if src_col in df.columns:
            df[new_col] = df[src_col].map(mapping)
            report["brackets_converted"].append(f"{src_col} -> {new_col} (bracket midpoint)")

    # ---- 4. Classification target ----
    df["likely_to_use_binary"] = (
        df["E1_Likelihood_To_Use_Label"].isin(["Likely", "Very likely"])
    ).astype(int)
    df["likely_to_use_label2"] = np.where(df["likely_to_use_binary"] == 1, "Yes", "No")

    # ---- 5. Parse multi-select columns into binary indicator columns ----
    for col in MULTISELECT_COLS:
        if col in df.columns:
            indicator_cols = _parse_multiselect(df[col])
            prefix = col.split("_")[0]  # B5, C1, C2, C3
            for label, vals in indicator_cols.items():
                safe_name = f"{prefix}__{label}"[:60]
                df[safe_name] = vals
            report["multiselect_parsed"].append(f"{col} -> {len(indicator_cols)} binary columns (prefix '{prefix}__')")

    # ---- 6. Outlier detection (IQR rule) on derived numeric fields ----
    for col in NUMERIC_DERIVED_COLS:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        is_outlier = (df[col] < lower) | (df[col] > upper)
        df[f"{col}_outlier"] = is_outlier.astype(int)
        df[f"{col}_capped"] = df[col].clip(lower, upper)
        n_out = int(is_outlier.sum())
        report["outliers_flagged"][col] = {
            "n_outliers": n_out, "pct": round(n_out / len(df) * 100, 1),
            "lower_bound": round(lower, 1), "upper_bound": round(upper, 1),
        }

    return df, report


def cleaning_summary_table(report):
    """Flatten the outlier section of the report into a display-ready table."""
    rows = []
    for col, info in report["outliers_flagged"].items():
        rows.append({
            "Column": col, "Outliers detected": info["n_outliers"],
            "% of respondents": info["pct"], "IQR lower bound": info["lower_bound"],
            "IQR upper bound": info["upper_bound"],
        })
    return pd.DataFrame(rows)
