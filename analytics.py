"""
analytics.py
============
Descriptive, diagnostic, and correlation analytics for the cleaned
Freeze-Dry Meal Service survey dataset.
"""

import numpy as np
import pandas as pd
from scipy import stats

NUMERIC_COLS = [
    "age_numeric", "income_numeric", "household_numeric", "price_numeric",
    "qty_numeric", "spend_numeric", "E1_Likelihood_To_Use_Score",
    "D1_Taste_Retention_Importance", "D2_Price_Affordability_Importance",
    "D3_Shelf_Life_Importance", "D4_Hygiene_Certification_Importance",
    "D5_Convenience_Importance", "D6_Brand_Trust_Importance",
    "D7_Variety_Importance", "D8_Speed_Turnaround_Importance",
]

CATEGORICAL_COLS = [
    "A2_Segment", "A4_Gender", "A6_Occupation", "C4_Packaging_Preference",
    "B4_Frequency_Sending_Food", "A3_Age_Group", "A7_Income_Bracket",
]


# --------------------------------------------------------------------------
# DESCRIPTIVE
# --------------------------------------------------------------------------

def descriptive_stats(df, cols=None):
    cols = [c for c in (cols or NUMERIC_COLS) if c in df.columns]
    desc = df[cols].describe().T
    desc["skew"] = df[cols].skew()
    return desc.round(2)


def correlation_matrix(df, cols=None):
    cols = [c for c in (cols or NUMERIC_COLS) if c in df.columns]
    return df[cols].corr()


def top_correlations(corr_df, top_n=8, min_abs=0.05):
    pairs = []
    cols = corr_df.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr_df.iloc[i, j]
            if pd.notna(r) and abs(r) >= min_abs:
                pairs.append((cols[i], cols[j], r))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return pairs[:top_n]


def _strength_label(r):
    a = abs(r)
    if a >= 0.5:
        return "strong"
    if a >= 0.3:
        return "moderate"
    if a >= 0.1:
        return "weak"
    return "negligible"


def correlation_insight_sentences(pairs):
    sentences = []
    for v1, v2, r in pairs:
        direction = "positive" if r > 0 else "negative"
        strength = _strength_label(r)
        sentences.append(f"**{v1}** and **{v2}** show a {strength} {direction} correlation (r = {r:.2f}).")
    return sentences


def preference_frequency(df, prefix):
    """Count frequency of each option for a parsed multi-select group
    (columns named '{prefix}__{label}')."""
    cols = [c for c in df.columns if c.startswith(f"{prefix}__")]
    counts = df[cols].sum().sort_values(ascending=False)
    counts.index = [c.split("__", 1)[1] for c in counts.index]
    out = counts.reset_index()
    out.columns = ["Option", "Count"]
    out["% of respondents"] = (out["Count"] / len(df) * 100).round(1)
    return out


# --------------------------------------------------------------------------
# DIAGNOSTIC
# --------------------------------------------------------------------------

def chi_square_test(df, cat_col, target_col="likely_to_use_label2"):
    ct = pd.crosstab(df[cat_col], df[target_col])
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    n = ct.values.sum()
    r, k = ct.shape
    denom = min(r - 1, k - 1) if min(r - 1, k - 1) > 0 else 1
    cramers_v = np.sqrt((chi2 / n) / denom)
    resid = ((ct - expected) / np.sqrt(expected)).round(2)
    return {"crosstab": ct, "chi2": chi2, "p_value": p, "dof": dof,
            "cramers_v": cramers_v, "std_residuals": resid}


def rate_table(df, group_col, target_col="likely_to_use_binary", min_count=5):
    overall_rate = df[target_col].mean()
    g = df.groupby(group_col).agg(n=(target_col, "count"), positive=(target_col, "sum"))
    g["rate_%"] = (g["positive"] / g["n"] * 100).round(1)
    g["overall_rate_%"] = round(overall_rate * 100, 1)
    g["deviation_pts"] = (g["rate_%"] - g["overall_rate_%"]).round(1)
    g = g[g["n"] >= min_count].sort_values("deviation_pts", ascending=False)
    return g.reset_index()


def continuous_group_diff(df, numeric_col, target_col="likely_to_use_label2"):
    a = df.loc[df[target_col] == "Yes", numeric_col].dropna()
    b = df.loc[df[target_col] == "No", numeric_col].dropna()
    stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"yes_mean": a.mean(), "yes_median": a.median(),
            "no_mean": b.mean(), "no_median": b.median(),
            "u_stat": stat, "p_value": p}


# --------------------------------------------------------------------------
# CLUSTERING / SEGMENTATION
# --------------------------------------------------------------------------

CLUSTER_FEATURES = [
    "D1_Taste_Retention_Importance", "D2_Price_Affordability_Importance",
    "D3_Shelf_Life_Importance", "D4_Hygiene_Certification_Importance",
    "D5_Convenience_Importance", "D6_Brand_Trust_Importance",
    "D7_Variety_Importance", "D8_Speed_Turnaround_Importance",
]

CLUSTER_LABELS = {
    0: "Quality-Conscious NRI",
    1: "Budget Traveller",
    2: "Convenience Seeker",
    3: "Low-Intent / Skeptical",
}


def run_clustering(df, k=4):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    feats = [c for c in CLUSTER_FEATURES if c in df.columns]
    X = df[feats].fillna(df[feats].median())
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(Xs)

    from sklearn.metrics import silhouette_score
    sil = silhouette_score(Xs, labels)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(Xs)

    centers_df = pd.DataFrame(km.cluster_centers_, columns=feats)
    centers_df.index = [f"Cluster {i}" for i in range(k)]

    return {
        "labels": labels, "silhouette": sil,
        "coords": coords, "explained_var": pca.explained_variance_ratio_,
        "centers": centers_df, "features": feats,
    }


# --------------------------------------------------------------------------
# CLASSIFICATION
# --------------------------------------------------------------------------

CLF_FEATURES = [
    "age_numeric", "income_numeric", "price_numeric", "qty_numeric", "spend_numeric",
    "D1_Taste_Retention_Importance", "D2_Price_Affordability_Importance",
    "D3_Shelf_Life_Importance", "D4_Hygiene_Certification_Importance",
    "D5_Convenience_Importance", "D6_Brand_Trust_Importance",
    "D7_Variety_Importance", "D8_Speed_Turnaround_Importance",
]


def run_classification(df):
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                  f1_score, roc_auc_score, roc_curve, confusion_matrix)
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    feats = [c for c in CLF_FEATURES if c in df.columns]
    X = df[feats]
    y = df["likely_to_use_binary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    models = {
        "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42),
        "KNN":                 KNeighborsClassifier(n_neighbors=9),
    }

    results = {}
    for name, model in models.items():
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", model),
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        results[name] = {
            "accuracy":  round(accuracy_score(y_test, y_pred), 3),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 3),
            "recall":    round(recall_score(y_test, y_pred, zero_division=0), 3),
            "f1":        round(f1_score(y_test, y_pred, zero_division=0), 3),
            "roc_auc":   round(roc_auc_score(y_test, y_prob), 3),
            "confusion_matrix": cm,
            "fpr": fpr, "tpr": tpr,
        }

    # Feature importance from Random Forest
    rf_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)),
    ])
    rf_pipe.fit(X_train, y_train)
    importance_df = pd.DataFrame({
        "Feature": feats,
        "Importance": rf_pipe.named_steps["model"].feature_importances_,
    }).sort_values("Importance", ascending=True)

    return results, importance_df, y_test


# --------------------------------------------------------------------------
# REGRESSION
# --------------------------------------------------------------------------

REG_FEATURES = [
    "age_numeric", "income_numeric", "qty_numeric",
    "D1_Taste_Retention_Importance", "D2_Price_Affordability_Importance",
    "D3_Shelf_Life_Importance", "D4_Hygiene_Certification_Importance",
    "D5_Convenience_Importance", "D6_Brand_Trust_Importance",
    "D7_Variety_Importance", "D8_Speed_Turnaround_Importance",
]


def run_regression(df, target_col="price_numeric"):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    feats = [c for c in REG_FEATURES if c in df.columns]
    X = df[feats]
    y = df[target_col].fillna(df[target_col].median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    models = {
        "Random Forest":  RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42),
        "Ridge (Linear)": Ridge(alpha=1.0),
        "Decision Tree":  DecisionTreeRegressor(max_depth=5, random_state=42),
        "KNN":            KNeighborsRegressor(n_neighbors=9),
    }

    results = {}
    for name, model in models.items():
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", model),
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        results[name] = {
            "r2":  round(r2_score(y_test, y_pred), 3),
            "mae": round(mean_absolute_error(y_test, y_pred), 1),
            "y_test": y_test.values,
            "y_pred": y_pred,
        }

    # Best model actual vs predicted
    best_name = max(results, key=lambda n: results[n]["r2"])
    return results, best_name


# --------------------------------------------------------------------------
# ASSOCIATION RULE MINING
# --------------------------------------------------------------------------

def run_association_rules(df, min_support=0.08, min_lift=1.2):
    from mlxtend.frequent_patterns import apriori, association_rules

    basket_cols = [c for c in df.columns
                   if c.startswith(("C1__", "C2__", "C3__", "B5__"))]
    if not basket_cols:
        return pd.DataFrame(), pd.DataFrame()

    basket = df[basket_cols].astype(bool)
    freq_items = apriori(basket, min_support=min_support,
                          use_colnames=True, max_len=3)
    if freq_items.empty:
        return pd.DataFrame(), freq_items

    rules = association_rules(freq_items, metric="lift", min_threshold=min_lift)
    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)

    # Clean up frozenset labels
    rules["antecedents"] = rules["antecedents"].apply(
        lambda x: ", ".join(s.split("__", 1)[1] if "__" in s else s for s in x)
    )
    rules["consequents"] = rules["consequents"].apply(
        lambda x: ", ".join(s.split("__", 1)[1] if "__" in s else s for s in x)
    )
    return rules[["antecedents", "consequents", "support", "confidence", "lift"]].round(3), freq_items
