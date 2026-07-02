# 🥘 Freeze-Dry Hyderabad - Market Validation Dashboard

A lean Streamlit dashboard answering one question: **is there a real market
in Hyderabad for a freeze-dried home-cooked meal service?** Built entirely
from the primary research questionnaire data (520 respondents, NRI/Family
and Traveler/Student segments) - no predictive modeling, just descriptive,
diagnostic, and correlation analytics.

## 📂 Project Structure

```
freeze_dry_app/
├── app.py              ← Main Streamlit dashboard (4 pages)
├── data_cleaning.py     ← Raw questionnaire -> analysis-ready data (standalone)
├── analytics.py         ← Descriptive / diagnostic / correlation functions (standalone)
├── requirements.txt     ← Python dependencies (lightweight - no sklearn/ML libs)
├── README.md            ← This file
├── .gitignore
└── data/
    └── Freeze_Dry_Survey_Data.xlsx   ← Bundled questionnaire export
```

## 🚀 Quick Start (Local)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. Use the sidebar to upload your own
questionnaire export (same column schema), or leave blank to explore the
bundled sample.

## ☁️ Deploy on Streamlit Community Cloud

1. Push this folder's contents to a GitHub repo (root = this folder's contents).
2. [share.streamlit.io](https://share.streamlit.io) → **New app** → select repo,
   branch `main`, main file `app.py`.
3. Click **Deploy** - Streamlit Cloud installs `requirements.txt` automatically.

> If you ever hit `ModuleNotFoundError`, it means something imported in the
> code is missing from `requirements.txt`. Add it, push, then **Manage app
> → Reboot app**.

## 📊 Dashboard Pages

| # | Page | What's inside |
|---|------|----------------|
| 1 | 🏠 **Home & Data Overview** | KPI strip, segment mix, price distribution, raw data preview |
| 2 | 📊 **Descriptive Analysis** | Summary statistics (mean/median/std/skew), full correlation heatmap, auto-generated plain-language insights, interactive scatter explorer with OLS trendline, dish/occasion/channel preference frequency charts |
| 3 | 🔍 **Diagnostic Analysis** | Chi-square + Cramer's V for any attribute vs purchase intent, standardized residuals (which specific categories drive the effect), likely-to-use rate tables, Mann-Whitney comparison of continuous variables |
| 4 | 💡 **Business Insights** | Auto-generated narrative tying the strongest segment, price sensitivity, and demand signal back to the Hyderabad go/no-go decision |

## 🧪 Data Pipeline

`data_cleaning.py` takes the **raw questionnaire export** (the human-readable
`Questionnaire_Responses` sheet - bracketed answers, semicolon-joined
multi-select text, occasional blanks) and produces an analysis-ready
dataframe:

1. **Missing values** - handled per-column with a documented, defensible
   decision (not blanket imputation) - e.g. blank locality → "Unknown",
   blank brand-used → "Not Applicable" (only ever filled for prior users).
2. **Bracket → numeric** - age group, income, max price/kg, order quantity,
   and monthly spend are converted to numeric midpoint estimates so they can
   be correlated and plotted.
3. **Multi-select parsing** - semicolon-joined answers (dishes, occasions,
   channels, current methods) are exploded into individual binary columns.
4. **Outlier detection** - IQR rule on every derived numeric field; outliers
   are flagged (not silently dropped), and a capped version is kept
   alongside the raw value.

Every step is logged in a `report` dict produced by `data_cleaning.clean_and_transform()`
(missing-value handling, bracket→numeric conversions, multi-select parsing, IQR outlier
flags) - it runs automatically in the background so every other page works with clean,
numeric, analysis-ready data. It is not shown as a dedicated UI page.

## Notes

- This app intentionally does **not** include classification, regression,
  clustering, or association-rule mining - those belong to a separate
  predictive-modeling deliverable on the same underlying dataset. This
  dashboard is scoped purely to descriptive + diagnostic + correlation
  analytics for market validation.
- `likely_to_use_binary` (the purchase-intent flag used throughout) is
  derived from the survey's E1 likelihood question (`Likely` / `Very likely`
  = 1, else 0).
