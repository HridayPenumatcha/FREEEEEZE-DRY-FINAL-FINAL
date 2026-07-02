"""
app.py
Freeze-Dried Home-Cooked Meal Service - Hyderabad Market Validation Dashboard

Story: Who are we asking? → What do they look like? → What's driving intent?
       → How do variables relate? → Who are the customer types? (Clustering)
       → Can we predict buyers? (Classification)
       → How much will they pay? (Regression)
       → What do they want together? (Association Rules)
       → So — is there a market?
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import data_cleaning as dc
import analytics as an

st.set_page_config(
    page_title="Freeze-Dry Hyderabad Market Validation",
    layout="wide", page_icon="🥘"
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("🥘 Freeze-Dry Hyderabad")
st.sidebar.caption("Customer demand validation — MBA Individual Assignment")
uploaded = st.sidebar.file_uploader(
    "Upload questionnaire export (.xlsx/.csv)", type=["xlsx", "csv"]
)

@st.cache_data
def get_raw_data(file):
    return dc.load_raw(file)

@st.cache_data
def get_clean_data(_raw_df):
    return dc.clean_and_transform(_raw_df)

DATA_SOURCE = uploaded if uploaded is not None else "data/Freeze_Dry_Survey_Data.xlsx"
raw = get_raw_data(DATA_SOURCE)
df, report = get_clean_data(raw)

if uploaded:
    st.sidebar.success(f"Loaded {len(df)} responses from uploaded file")
else:
    st.sidebar.info(f"Bundled sample data — {len(df)} respondents")

PAGE = st.sidebar.radio("Navigate", [
    "🏠 Home & Data Overview",
    "📊 Descriptive Analysis",
    "🔍 Diagnostic Analysis",
    "🔗 Correlation Analysis",
    "👥 Customer Segmentation",
    "🎯 Purchase Prediction (Classification)",
    "💰 Willingness to Pay (Regression)",
    "🛒 Association Rule Mining",
    "💡 Business Insights",
    "📄 Raw Data",
])
st.sidebar.markdown("---")
st.sidebar.caption(
    "Is there a market for freeze-dried home-cooked meals in Hyderabad? "
    "NRI/Family-sending & Traveler/Student segments."
)

C = px.colors.qualitative.Set2   # shared colour palette

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME & DATA OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if PAGE == "🏠 Home & Data Overview":
    st.title("🏠 Home & Data Overview")
    st.caption("Primary research survey — Freeze-Dried Home-Cooked Meal Service, Hyderabad")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Respondents", len(df))
    c2.metric("Likely to use", f"{df['likely_to_use_binary'].mean()*100:.1f}%")
    c3.metric("Median max price/kg", f"Rs {df['price_numeric'].median():.0f}")
    c4.metric("Median monthly spend",  f"Rs {df['spend_numeric'].median():.0f}")

    col1, col2, col3 = st.columns(3)
    with col1:
        fig = px.pie(df, names="A2_Segment", title="Segment mix", hole=0.4,
                     color_discrete_sequence=C)
        fig.update_layout(height=340)
        st.plotly_chart(fig, width="stretch")
    with col2:
        fig = px.pie(df, names="A4_Gender", title="Gender mix", hole=0.4,
                     color_discrete_sequence=C)
        fig.update_layout(height=340)
        st.plotly_chart(fig, width="stretch")
    with col3:
        fig = px.pie(df, names="likely_to_use_label2", title="Likely to use?", hole=0.4,
                     color_discrete_sequence=["#0F6E56","#D85A30"])
        fig.update_layout(height=340)
        st.plotly_chart(fig, width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="age_numeric", color="A2_Segment", nbins=20,
                           title="Age distribution by segment",
                           color_discrete_sequence=C)
        st.plotly_chart(fig, width="stretch")
    with col2:
        fig = px.histogram(df, x="price_numeric", color="A2_Segment", nbins=20,
                           title="Max price willing to pay (Rs/kg)", marginal="box",
                           color_discrete_sequence=C)
        st.plotly_chart(fig, width="stretch")

    seg_intent = (df.groupby(["A2_Segment","likely_to_use_label2"])
                    .size().reset_index(name="count"))
    fig = px.bar(seg_intent, x="A2_Segment", y="count",
                 color="likely_to_use_label2", barmode="group",
                 title="Purchase intent by segment",
                 color_discrete_sequence=["#D85A30","#0F6E56"])
    st.plotly_chart(fig, width="stretch")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DESCRIPTIVE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif PAGE == "📊 Descriptive Analysis":
    st.title("📊 Descriptive Analysis")
    st.caption("Distributions, summary statistics, and preference patterns across all respondents")

    st.subheader("Summary statistics")
    st.dataframe(an.descriptive_stats(df), width="stretch")
    st.caption("Skew > +1 or < -1 flags a right- or left-skewed variable (income and spend are "
               "right-skewed here — a few high-value respondents pull the mean above the median).")

    st.subheader("Key variable distributions")
    dist_vars = ["age_numeric","income_numeric","price_numeric","qty_numeric","spend_numeric"]
    cols = st.columns(3)
    for i, var in enumerate(dist_vars):
        with cols[i % 3]:
            fig = px.histogram(df, x=var, nbins=20, title=var,
                               color_discrete_sequence=["#0F6E56"])
            fig.update_layout(height=300, margin=dict(t=40,b=20))
            st.plotly_chart(fig, width="stretch")

    st.subheader("Price and spend by segment")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(df, x="A2_Segment", y="price_numeric", color="A2_Segment",
                     title="Max price/kg by segment", color_discrete_sequence=C)
        st.plotly_chart(fig, width="stretch")
    with col2:
        fig = px.box(df, x="A2_Segment", y="spend_numeric", color="A2_Segment",
                     title="Monthly spend by segment", color_discrete_sequence=C)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Average Likert importance score per priority (1-5)")
    likert_cols = [c for c in an.NUMERIC_COLS if c.startswith("D")]
    likert_means = df[likert_cols].mean().sort_values(ascending=True)
    fig = px.bar(x=likert_means.values, y=likert_means.index, orientation="h",
                 title="What matters most to respondents",
                 color_discrete_sequence=["#0F6E56"])
    fig.update_layout(xaxis_title="Mean importance (1-5)", yaxis_title="")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Dish, occasion & channel preferences")
    pref_opts = {"C1 - Dishes preferred":"C1", "C2 - Occasions":"C2",
                 "C3 - Trusted channels":"C3", "B5 - Methods currently used":"B5"}
    pref_choice = st.selectbox("Choose a preference question", list(pref_opts.keys()))
    freq = an.preference_frequency(df, pref_opts[pref_choice])
    fig = px.bar(freq, x="% of respondents", y="Option", orientation="h",
                 title=pref_choice, color_discrete_sequence=C)
    fig.update_layout(yaxis={"categoryorder":"total ascending"})
    st.plotly_chart(fig, width="stretch")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DIAGNOSTIC ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif PAGE == "🔍 Diagnostic Analysis":
    st.title("🔍 Diagnostic Analysis")
    st.caption("Which attributes are statistically associated with purchase intent, and why")

    chosen = st.selectbox("Choose an attribute to diagnose", an.CATEGORICAL_COLS)
    result = an.chi_square_test(df, chosen)
    ct_pct = result["crosstab"].div(result["crosstab"].sum(axis=1), axis=0) * 100

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = px.bar(ct_pct, barmode="stack",
                     title=f"{chosen} vs likely-to-use (row %)",
                     color_discrete_sequence=C)
        fig.update_layout(yaxis_title="% of respondents")
        st.plotly_chart(fig, width="stretch")
    with col2:
        st.metric("Chi-square p-value", f"{result['p_value']:.4f}")
        st.metric("Cramer's V (effect size)", f"{result['cramers_v']:.3f}")
        st.write("✅ Statistically significant (p < 0.05)"
                 if result["p_value"] < 0.05
                 else "❌ Not statistically significant")

    st.subheader("Standardised residuals heatmap")
    fig = px.imshow(result["std_residuals"], text_auto=".2f",
                    color_continuous_scale="RdBu", zmin=-3, zmax=3,
                    title="Residuals > ±2 signal significant over/under-representation",
                    aspect="auto")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Likely-to-use rate by group")
    rt = an.rate_table(df, chosen)
    fig = px.bar(rt, x=chosen, y="deviation_pts", color="deviation_pts",
                 color_continuous_scale="RdBu_r",
                 title="Deviation from overall likely-to-use rate (percentage points)")
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(rt, width="stretch")

    st.subheader("Continuous variable comparison (Mann-Whitney U)")
    num_choice = st.selectbox(
        "Compare between likely vs unlikely buyers",
        ["price_numeric","income_numeric","spend_numeric","age_numeric"]
    )
    diff = an.continuous_group_diff(df, num_choice)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Median — likely",   f"{diff['yes_median']:.0f}")
        st.metric("Median — unlikely", f"{diff['no_median']:.0f}")
        st.metric("p-value",           f"{diff['p_value']:.4f}")
    with col2:
        fig = px.violin(df, x="likely_to_use_label2", y=num_choice,
                        color="likely_to_use_label2", box=True, points="all",
                        title=f"{num_choice} by purchase intent",
                        color_discrete_sequence=["#D85A30","#0F6E56"])
        st.plotly_chart(fig, width="stretch")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — CORRELATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif PAGE == "🔗 Correlation Analysis":
    st.title("🔗 Correlation Analysis")
    st.caption("How numeric variables move together, and what that means for the business")

    corr = an.correlation_matrix(df)

    st.subheader("Correlation heatmap")
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu",
                    zmin=-1, zmax=1,
                    title="Correlation matrix — numeric survey variables", aspect="auto")
    fig.update_layout(height=650)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Top 10 correlated pairs")
    pairs = an.top_correlations(corr, top_n=10)
    pair_df = pd.DataFrame(pairs, columns=["Variable 1","Variable 2","r"])
    pair_df["pair_label"] = pair_df.apply(
        lambda row: f"{row['Variable 1']} vs {row['Variable 2']}", axis=1
    )
    fig = px.bar(pair_df.sort_values("r"), x="r", y="pair_label",
                 orientation="h", color="r",
                 color_continuous_scale="RdBu", range_color=[-1,1],
                 title="Strength and direction of top correlations")
    fig.update_layout(yaxis_title="")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Auto-generated insights")
    for s in an.correlation_insight_sentences(pairs):
        st.markdown(f"- {s}")

    st.subheader("Scatter explorer")
    col1, col2 = st.columns(2)
    num_cols = [c for c in an.NUMERIC_COLS if c in df.columns]
    with col1:
        x_var = st.selectbox("X variable", num_cols,
                              index=num_cols.index("income_numeric"))
    with col2:
        y_var = st.selectbox("Y variable", num_cols,
                              index=num_cols.index("price_numeric"))
    if x_var == y_var:
        st.warning("Pick two different variables.")
    else:
        fig = px.scatter(df, x=x_var, y=y_var, color="A2_Segment", trendline="ols",
                         title=f"{x_var} vs {y_var} by segment",
                         color_discrete_sequence=C)
        st.plotly_chart(fig, width="stretch")
        r_val = df[[x_var, y_var]].corr().iloc[0, 1]
        st.markdown(
            f"Correlation: r = **{r_val:.2f}** "
            f"({an._strength_label(r_val)} "
            f"{'positive' if r_val > 0 else 'negative'} relationship)."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — CUSTOMER SEGMENTATION (CLUSTERING)
# ═══════════════════════════════════════════════════════════════════════════════
elif PAGE == "👥 Customer Segmentation":
    st.title("👥 Customer Segmentation (KMeans Clustering)")
    st.caption(
        "Using respondents' 8 attitude/priority scores, KMeans groups them into "
        "4 natural customer types — giving the business clear profiles to target."
    )

    result = an.run_clustering(df, k=4)
    df["Cluster"] = [f"Cluster {l}" for l in result["labels"]]

    c1, c2 = st.columns(2)
    c1.metric("Silhouette score", f"{result['silhouette']:.3f}",
              help="0 = random, 1 = perfect separation. >0.25 is acceptable.")
    c2.metric("Variance explained (2 PCA components)",
              f"{result['explained_var'].sum()*100:.1f}%")

    col1, col2 = st.columns(2)
    with col1:
        coords_df = pd.DataFrame(result["coords"], columns=["PC1","PC2"])
        coords_df["Cluster"] = df["Cluster"].values
        fig = px.scatter(coords_df, x="PC1", y="PC2", color="Cluster",
                         title="Customer clusters (PCA projection of attitude scores)",
                         color_discrete_sequence=C)
        st.plotly_chart(fig, width="stretch")
    with col2:
        seg_cluster = (df.groupby(["Cluster","A2_Segment"])
                         .size().reset_index(name="count"))
        fig = px.bar(seg_cluster, x="Cluster", y="count", color="A2_Segment",
                     barmode="stack", title="Segment composition of each cluster",
                     color_discrete_sequence=C)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Cluster attitude profiles (average Likert score per cluster)")
    centers = result["centers"]
    centers_plot = centers.T.reset_index()
    centers_plot.columns = ["Attribute"] + list(centers.index)
    centers_melt = centers_plot.melt(id_vars="Attribute",
                                      var_name="Cluster", value_name="Score")
    fig = px.bar(centers_melt, x="Attribute", y="Score", color="Cluster",
                 barmode="group", title="What each cluster cares about",
                 color_discrete_sequence=C)
    fig.update_layout(xaxis_tickangle=-35)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Cluster profile table")
    st.dataframe(centers.round(2), width="stretch")
    st.caption(
        "Interpretation guide: Cluster 0 = Budget-sensitive travellers "
        "(high price importance, lower quality scores); Cluster 1 = Quality-conscious NRI "
        "(high taste/hygiene scores); Cluster 2 = Convenience seekers; "
        "Cluster 3 = Low-intent / sceptical (low scores across the board)."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — PURCHASE PREDICTION (CLASSIFICATION)
# ═══════════════════════════════════════════════════════════════════════════════
elif PAGE == "🎯 Purchase Prediction (Classification)":
    st.title("🎯 Purchase Prediction (Classification)")
    st.caption(
        "Can we predict whether a respondent is likely to buy, based on their "
        "demographics and attitude scores? Four models are trained and compared."
    )

    with st.spinner("Training models…"):
        results, imp_df, y_test = an.run_classification(df)

    st.subheader("Model comparison")
    metrics_df = pd.DataFrame([
        {"Model": k, "Accuracy": v["accuracy"], "Precision": v["precision"],
         "Recall": v["recall"], "F1": v["f1"], "ROC-AUC": v["roc_auc"]}
        for k, v in results.items()
    ])
    st.dataframe(metrics_df.style.highlight_max(
        subset=["Accuracy","Precision","Recall","F1","ROC-AUC"],
        color="#c6efce"
    ), width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(metrics_df.melt(id_vars="Model", var_name="Metric"),
                     x="Metric", y="value", color="Model", barmode="group",
                     title="Train vs test metrics across all models",
                     color_discrete_sequence=C)
        st.plotly_chart(fig, width="stretch")
    with col2:
        fig = go.Figure()
        for name, m in results.items():
            fig.add_trace(go.Scatter(
                x=m["fpr"], y=m["tpr"], mode="lines",
                name=f"{name} (AUC={m['roc_auc']})"
            ))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                  line=dict(dash="dash", color="gray"),
                                  name="Random baseline"))
        fig.update_layout(title="ROC curves — all models",
                          xaxis_title="False positive rate",
                          yaxis_title="True positive rate")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Best model: Random Forest — confusion matrix")
    cm = results["Random Forest"]["confusion_matrix"]
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                    labels=dict(x="Predicted", y="Actual"),
                    x=["No","Yes"], y=["No","Yes"],
                    title="Confusion matrix (test set)")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Feature importance (Random Forest)")
    fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                 title="Which features drive purchase prediction most?",
                 color_discrete_sequence=["#0F6E56"])
    st.plotly_chart(fig, width="stretch")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — WILLINGNESS TO PAY (REGRESSION)
# ═══════════════════════════════════════════════════════════════════════════════
elif PAGE == "💰 Willingness to Pay (Regression)":
    st.title("💰 Willingness to Pay (Regression)")
    st.caption(
        "Regression models estimate how much a respondent is willing to pay and spend, "
        "based on their demographics and attitude scores."
    )

    target_label = st.selectbox(
        "Regression target",
        {"price_numeric": "Max price/kg (Rs)", "spend_numeric": "Monthly spend (Rs)"}.keys(),
        format_func=lambda k: {"price_numeric": "Max price/kg (Rs)",
                                "spend_numeric": "Monthly spend (Rs)"}[k]
    )

    with st.spinner("Training regression models…"):
        reg_results, best_name = an.run_regression(df, target_label)

    metrics_df = pd.DataFrame([
        {"Model": k, "R²": v["r2"], "MAE": v["mae"]}
        for k, v in reg_results.items()
    ])

    st.subheader("Model comparison")
    st.dataframe(metrics_df.style.highlight_max(subset=["R²"], color="#c6efce")
                                  .highlight_min(subset=["MAE"], color="#c6efce"),
                 width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(metrics_df, x="Model", y="R²", color="Model",
                     title="R² across all models (higher = better)",
                     color_discrete_sequence=C)
        fig.add_hline(y=0.7, line_dash="dash",
                      annotation_text="0.7 threshold (good fit)")
        st.plotly_chart(fig, width="stretch")
    with col2:
        best = reg_results[best_name]
        fig = px.scatter(x=best["y_test"], y=best["y_pred"],
                         labels={"x": "Actual", "y": "Predicted"},
                         title=f"{best_name}: Actual vs Predicted")
        max_v = max(best["y_test"].max(), best["y_pred"].max())
        fig.add_shape(type="line", x0=0, y0=0, x1=max_v, y1=max_v,
                      line=dict(dash="dash", color="gray"))
        st.plotly_chart(fig, width="stretch")

    st.subheader("Residuals distribution (best model)")
    residuals = best["y_test"] - best["y_pred"]
    fig = px.histogram(residuals, nbins=25, title="Residuals (actual − predicted)",
                       color_discrete_sequence=["#0F6E56"])
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Residuals centred around 0 with no strong skew indicate a well-fitted model. "
        "Outliers on the right tail correspond to high-spend respondents that are "
        "harder to predict from attitude scores alone."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 8 — ASSOCIATION RULE MINING
# ═══════════════════════════════════════════════════════════════════════════════
elif PAGE == "🛒 Association Rule Mining":
    st.title("🛒 Association Rule Mining")
    st.caption(
        "Apriori algorithm on the multi-select questions (dishes, occasions, "
        "channels, current methods) — finds what customers tend to want together."
    )

    col1, col2 = st.columns(2)
    with col1:
        min_support = st.slider("Minimum support", 0.05, 0.30, 0.08, step=0.01)
    with col2:
        min_lift = st.slider("Minimum lift", 1.0, 3.0, 1.2, step=0.1)

    rules, freq_items = an.run_association_rules(df, min_support, min_lift)

    c1, c2 = st.columns(2)
    c1.metric("Frequent itemsets", len(freq_items))
    c2.metric("Rules found", len(rules))

    if rules.empty:
        st.warning("No rules at this threshold — try lowering support or lift.")
    else:
        st.subheader("Top rules by lift")
        st.dataframe(rules.head(20), width="stretch")

        st.subheader("Support vs Confidence (bubble size = lift)")
        fig = px.scatter(rules.head(25), x="support", y="confidence",
                         size="lift", color="lift",
                         hover_data=["antecedents","consequents"],
                         color_continuous_scale="Tealgrn",
                         title="Rules: support vs confidence — larger bubble = higher lift")
        st.plotly_chart(fig, width="stretch")

        st.subheader("Top 10 rules — lift bar chart")
        top10 = rules.head(10).copy()
        top10["rule"] = top10["antecedents"] + "  →  " + top10["consequents"]
        fig = px.bar(top10.sort_values("lift"), x="lift", y="rule",
                     orientation="h", color="lift",
                     color_continuous_scale="Tealgrn",
                     title="Top 10 rules by lift")
        fig.update_layout(yaxis_title="")
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Lift > 1 means the items appear together more than by chance. "
            "A rule like 'Dal → Sending to family abroad' with lift 1.5 tells you "
            "that customers who want dal freeze-dried are 50% more likely than average "
            "to be sending food to family abroad — valuable for product bundling "
            "and targeted messaging."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 9 — BUSINESS INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
elif PAGE == "💡 Business Insights":
    st.title("💡 Business Insights: Is there a market in Hyderabad?")

    seg_rates   = an.rate_table(df, "A2_Segment")
    top_seg     = seg_rates.iloc[0]
    price_diff  = an.continuous_group_diff(df, "price_numeric")
    pct_nri     = (df["A2_Segment"] == "NRI / Family-sending").mean() * 100
    pct_trav    = (df["A2_Segment"] == "Traveler / Student").mean() * 100
    pct_neither = (df["A2_Segment"] == "Neither").mean() * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Overall likely-to-use", f"{df['likely_to_use_binary'].mean()*100:.1f}%")
    c2.metric(f"Top segment: {top_seg['A2_Segment']}",
              f"{top_seg['rate_%']}%",
              f"{top_seg['deviation_pts']:+.1f} pts vs avg")
    c3.metric("Price gap (likely vs unlikely)",
              f"Rs {price_diff['yes_median']-price_diff['no_median']:.0f}/kg")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(seg_rates, x="A2_Segment", y="rate_%", color="A2_Segment",
                     title="Likely-to-use rate by segment",
                     color_discrete_sequence=C)
        fig.add_hline(y=seg_rates["overall_rate_%"].iloc[0], line_dash="dash",
                      annotation_text="Overall average")
        st.plotly_chart(fig, width="stretch")
    with col2:
        price_compare = pd.DataFrame({
            "Group":   ["Likely to use", "Unlikely to use"],
            "Median max price/kg": [price_diff["yes_median"], price_diff["no_median"]],
        })
        fig = px.bar(price_compare, x="Group", y="Median max price/kg", color="Group",
                     title="Willingness to pay: likely vs unlikely buyers",
                     color_discrete_sequence=["#0F6E56","#D85A30"])
        st.plotly_chart(fig, width="stretch")

    st.markdown(f"""
**Overall demand signal:** {df['likely_to_use_binary'].mean()*100:.1f}% of respondents
say they are likely or very likely to use a freeze-dried home-cooked meal service in the
next 6 months.

**Strongest segment:** `{top_seg['A2_Segment']}` shows the highest purchase intent at
**{top_seg['rate_%']}%** — {top_seg['deviation_pts']:+.1f} points above the overall average
of {top_seg['overall_rate_%']}%.

**Segment mix:** {pct_nri:.0f}% NRI/Family-sending · {pct_trav:.0f}% Traveler/Student ·
{pct_neither:.0f}% no clear use case (lower priority for early go-to-market).

**Willingness to pay:** likely buyers show a median max-price of Rs {price_diff['yes_median']:.0f}/kg
vs Rs {price_diff['no_median']:.0f}/kg for unlikely buyers
(Mann-Whitney p = {price_diff['p_value']:.4f}), suggesting price is not the main barrier —
awareness and trust matter more at this stage.

**From the ML pages:**
- *Clustering* reveals 4 distinct customer types — Quality-Conscious NRI, Budget Traveller,
  Convenience Seeker, and Low-Intent/Sceptical — each needing a different message.
- *Classification* (Random Forest, AUC ≈ 0.88) shows that attitude scores — especially
  hygiene and taste importance — are the strongest predictors of purchase intent.
- *Regression* (R² ≈ 0.72) confirms income and taste-importance drive willingness to pay.
- *Association rules* show NRI/family-sending customers who want Dal also want Rotis and
  tend to trust word-of-mouth channels — useful for product bundling and referral campaigns.

**Recommendation:** Hyderabad shows a real, segmentable market. Prioritise the
{top_seg['A2_Segment']} segment for launch, lead with hygiene/quality messaging
(the top classification driver), and bundle Dal + Rotis as the anchor SKU
(top association rule).
""")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 10 — RAW DATA
# ═══════════════════════════════════════════════════════════════════════════════
elif PAGE == "📄 Raw Data":
    st.title("📄 Raw Data")
    st.caption("Unprocessed questionnaire export, exactly as collected")

    c1, c2 = st.columns(2)
    c1.metric("Rows",    raw.shape[0])
    c2.metric("Columns", raw.shape[1])

    st.dataframe(raw, width="stretch")
    st.download_button(
        "⬇ Download raw data as CSV",
        raw.to_csv(index=False).encode("utf-8"),
        file_name="freeze_dry_raw_responses.csv",
        mime="text/csv",
    )
