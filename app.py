"""
app.py  –  Nassau Candy Distributor
Factory Reallocation & Shipping Optimization Dashboard

Run:  streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from data_engine import (
    load_and_prepare, get_summary_kpis,
    get_region_categories, get_ship_modes, get_products, get_factory_list,
    FACTORIES, PRODUCT_FACTORY,
)
from ml_engine import (
    train_models, simulate_factory_reassignment, generate_recommendations,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nassau Candy – Shipping Optimizer",
    page_icon="🍬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #0d0d1a; }
[data-testid="stSidebar"]          { background-color: #12122a; }
.block-container { padding-top: 1.2rem; padding-bottom: 1rem; }

/* KPI cards */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg,#1a1a3a,#252560);
    border-radius: 12px;
    padding: 16px 20px;
    border-left: 4px solid #6c5ce7;
}
div[data-testid="stMetricValue"] { font-size: 1.55rem !important; color: #d0c8ff; }
div[data-testid="stMetricLabel"] { color: #888 !important; font-size: 0.82rem; }

/* Tab bar */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: #1a1a3a; border-radius: 8px 8px 0 0;
    color: #aaa; padding: 8px 18px; font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: #252560 !important; color: #fff !important;
    border-bottom: 3px solid #6c5ce7 !important;
}

/* Section headings */
.sec-head {
    font-size: 1.25rem; font-weight: 700; color: #c0b4ff;
    border-bottom: 2px solid #6c5ce7;
    padding-bottom: 6px; margin: 16px 0 10px;
}

/* Dataframe */
[data-testid="stDataFrame"] { border-radius: 10px; }

/* Buttons */
div.stButton > button {
    background: linear-gradient(90deg,#6c5ce7,#a29bfe);
    color: #fff; border: none; border-radius: 8px;
    font-weight: 700; padding: 8px 24px;
}
div.stButton > button:hover { opacity: 0.88; }
</style>
""", unsafe_allow_html=True)

# ── Cached loaders ────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "Nassau_Candy_Distributor.csv")

@st.cache_data(show_spinner="📦 Loading dataset…")
def _load():
    return load_and_prepare(DATA_PATH)

@st.cache_resource(show_spinner="🤖 Training ML models…")
def _train():
    df = _load()
    return train_models(df)

df = _load()
trained, results, best_name, meta = _train()

# ── Helpers ───────────────────────────────────────────────────────────────────
PLOTLY_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#ccc",
    margin=dict(l=30, r=20, t=40, b=30),
)
PALETTE = ["#6c5ce7","#fd79a8","#00cec9","#fdcb6e","#e17055","#74b9ff"]

def sec(title):
    st.markdown(f'<div class="sec-head">{title}</div>', unsafe_allow_html=True)

def make_bar(df_plot, x, y, title, color=None, h=None, colorscale=None):
    kwargs = dict(x=x, y=y, title=title, color_discrete_sequence=PALETTE)
    if color:      kwargs["color"] = color
    if colorscale: kwargs["color_continuous_scale"] = colorscale
    if h:          kwargs["orientation"] = "h"; kwargs["x"], kwargs["y"] = kwargs["y"], kwargs["x"]
    fig = px.bar(df_plot, **kwargs)
    fig.update_layout(**PLOTLY_DARK)
    return fig

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🍬 Nassau Candy\n### Shipping Optimizer")
    st.caption(f"Dataset: **{len(df):,}** orders  |  Model: **{best_name}**")
    st.divider()

    st.markdown("### 🌍 Global Filters")
    gl_region   = st.selectbox("Region",    ["All"] + get_region_categories(df))
    gl_ship     = st.selectbox("Ship Mode", ["All"] + get_ship_modes(df))
    gl_division = st.selectbox("Division",  ["All"] + sorted(df["Division"].dropna().unique().tolist()))

    st.divider()
    st.markdown("### ⚙️ Optimizer Settings")
    priority_val = st.slider("Speed ←→ Profit", 0, 100, 60,
                             help="0 = max profit focus  |  100 = max speed focus")
    priority_mode = "speed" if priority_val > 70 else ("profit" if priority_val < 30 else "balanced")
    st.caption(f"Mode: **{priority_mode.upper()}**")

    top_n = st.slider("Top-N Recommendations", 3, 10, 5)
    st.divider()
    st.markdown(f"**Best ML Model:** `{best_name}`")
    for name, v in results.items():
        st.caption(f"{'✅' if name == best_name else '  '} {name}: R²={v['R²']}  RMSE={v['RMSE']}")

# ── Apply global filters ──────────────────────────────────────────────────────
dff = df.copy()
if gl_region   != "All": dff = dff[dff["Region"]    == gl_region]
if gl_ship     != "All": dff = dff[dff["Ship Mode"] == gl_ship]
if gl_division != "All": dff = dff[dff["Division"]  == gl_division]

kpis = get_summary_kpis(dff)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🍬 Nassau Candy — Factory Reallocation & Shipping Optimizer")
st.caption("Intelligent decision intelligence for supply chain optimization  |  10,194 orders  |  5 factories  |  15 products")

# ── KPI Row ───────────────────────────────────────────────────────────────────
cols = st.columns(6)
icons = ["📦","⏱️","💰","📈","📊","🚚"]
for col, (label, val), icon in zip(cols, kpis.items(), icons):
    col.metric(f"{icon} {label}", val)

st.divider()

# ════════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  EDA & Insights",
    "🏭  Factory Simulator",
    "🔮  What-If Analysis",
    "🏆  Recommendations",
    "⚠️  Risk & Impact",
])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — EDA & INSIGHTS
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    sec("📋 Overview Charts")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(dff, x="Lead Time", nbins=50, color="Ship Mode",
                           title="Lead Time Distribution by Ship Mode",
                           color_discrete_sequence=PALETTE)
        fig.update_layout(**PLOTLY_DARK)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        lt_reg = dff.groupby("Region")["Lead Time"].mean().reset_index()
        fig = px.bar(lt_reg, x="Region", y="Lead Time",
                     title="Average Lead Time by Region",
                     color="Lead Time", color_continuous_scale="Viridis")
        fig.update_layout(**PLOTLY_DARK)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        div_fin = dff.groupby("Division")[["Sales","Gross Profit"]].sum().reset_index()
        fig = px.bar(div_fin, x="Division", y=["Sales","Gross Profit"],
                     barmode="group", title="Sales vs Gross Profit by Division",
                     color_discrete_sequence=["#6c5ce7","#00cec9"])
        fig.update_layout(**PLOTLY_DARK)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        lt_fac = dff.groupby("Factory")["Lead Time"].mean().reset_index().sort_values("Lead Time")
        fig = px.bar(lt_fac, x="Lead Time", y="Factory", orientation="h",
                     title="Average Lead Time by Factory",
                     color="Lead Time", color_continuous_scale="Plasma")
        fig.update_layout(**PLOTLY_DARK)
        st.plotly_chart(fig, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        top_prod = dff.groupby("Product Name")["Gross Profit"].sum().nlargest(10).reset_index()
        fig = px.bar(top_prod, x="Gross Profit", y="Product Name", orientation="h",
                     title="Top 10 Products by Gross Profit",
                     color="Gross Profit", color_continuous_scale="Blues")
        fig.update_layout(**PLOTLY_DARK)
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        sample = dff.sample(min(1200, len(dff)), random_state=42)
        fig = px.scatter(sample, x="Distance_km", y="Profit Margin %",
                         color="Factory", size="Units", opacity=0.65,
                         title="Distance vs Profit Margin (by Factory)",
                         color_discrete_sequence=PALETTE)
        fig.update_layout(**PLOTLY_DARK)
        st.plotly_chart(fig, use_container_width=True)

    sec("🗺️ Factory Locations")
    fact_df = pd.DataFrame([
        {"Factory": k, "Lat": v["lat"], "Lon": v["lon"]} for k, v in FACTORIES.items()
    ])
    fig_map = px.scatter_mapbox(
        fact_df, lat="Lat", lon="Lon", hover_name="Factory",
        zoom=3, mapbox_style="carto-darkmatter",
        color_discrete_sequence=["#6c5ce7"],
        title="Nassau Candy — Factory Network",
    )
    fig_map.update_traces(marker=dict(size=20))
    fig_map.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=440, font_color="#ccc")
    st.plotly_chart(fig_map, use_container_width=True)

    sec("🤖 ML Model Evaluation")
    res_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
    fig_ml = px.bar(
        res_df.melt(id_vars="Model", value_vars=["RMSE","MAE","R²"]),
        x="Model", y="value", color="variable", barmode="group",
        title="Model Performance Comparison",
        color_discrete_sequence=["#6c5ce7","#fd79a8","#00cec9"],
    )
    fig_ml.update_layout(**PLOTLY_DARK)
    st.plotly_chart(fig_ml, use_container_width=True)

    m1, m2, m3 = st.columns(3)
    for col, (name, v) in zip([m1, m2, m3], results.items()):
        col.metric(
            f"{'🏆 ' if name == best_name else ''}{name}",
            f"R² = {v['R²']}",
            f"RMSE {v['RMSE']}  |  MAE {v['MAE']}",
        )

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — FACTORY SIMULATOR
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    sec("🏭 Factory Optimization Simulator")
    st.caption("Select a product, region & ship mode — see how lead time changes across every factory")

    s1, s2, s3 = st.columns(3)
    sim_product = s1.selectbox("Product", get_products(df), key="sim_p")
    sim_region  = s2.selectbox("Region",  get_region_categories(df), key="sim_r")
    sim_ship    = s3.selectbox("Ship Mode", get_ship_modes(df), key="sim_s")

    if st.button("▶  Run Simulation", key="btn_sim"):
        with st.spinner("Simulating all factory assignments…"):
            sim_result = simulate_factory_reassignment(
                df, trained, best_name, meta,
                sim_product, sim_region, sim_ship
            )

        curr_fac = PRODUCT_FACTORY.get(sim_product, "Unknown")
        st.info(f"🏭 **Current Factory for '{sim_product}':** {curr_fac}")

        r1, r2 = st.columns(2)
        with r1:
            colors = ["#fd79a8" if c else "#6c5ce7" for c in sim_result["Is Current"]]
            fig = go.Figure(go.Bar(
                x=sim_result["Factory"],
                y=sim_result["Predicted Lead Time (days)"],
                marker_color=colors,
                text=sim_result["Predicted Lead Time (days)"].apply(lambda x: f"{x:.0f}d"),
                textposition="outside",
            ))
            fig.update_layout(title="Predicted Lead Time by Factory",
                              xaxis_tickangle=-20, **PLOTLY_DARK)
            st.plotly_chart(fig, use_container_width=True)

        with r2:
            fig = px.scatter(
                sim_result,
                x="Distance (km)", y="Predicted Lead Time (days)",
                size="Est. Profit Margin (%)",
                color="Factory", text="Factory",
                title="Distance vs Lead Time  (bubble = profit margin)",
                color_discrete_sequence=PALETTE,
            )
            fig.update_traces(textposition="top center")
            fig.update_layout(**PLOTLY_DARK)
            st.plotly_chart(fig, use_container_width=True)

        sec("📋 Simulation Results Table")
        display = sim_result.copy()
        display["Is Current"] = display["Is Current"].map({True: "✅ Current", False: ""})
        st.dataframe(
            display.set_index("Rank"),
            use_container_width=True,
            column_config={
                "Confidence Score": st.column_config.ProgressColumn(
                    "Confidence", min_value=0, max_value=1, format="%.2f"
                ),
                "Lead Time Saving (days)": st.column_config.NumberColumn(format="%.1f"),
            },
        )

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — WHAT-IF ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    sec("🔮 What-If Scenario Comparison")
    st.caption("Manually pick an alternative factory and instantly see the impact vs. current assignment")

    w1, w2, w3, w4 = st.columns(4)
    wi_product = w1.selectbox("Product",              get_products(df), key="wi_p")
    wi_region  = w2.selectbox("Region",               get_region_categories(df), key="wi_r")
    wi_ship    = w3.selectbox("Ship Mode",             get_ship_modes(df), key="wi_s")
    wi_alt     = w4.selectbox("Alternative Factory",   get_factory_list(), key="wi_f")

    if st.button("🔄  Compare Scenarios", key="btn_wi"):
        with st.spinner("Running scenario comparison…"):
            sim_all = simulate_factory_reassignment(
                df, trained, best_name, meta,
                wi_product, wi_region, wi_ship
            )

        curr_row = sim_all[sim_all["Is Current"]]
        alt_row  = sim_all[sim_all["Factory"] == wi_alt]

        if curr_row.empty or alt_row.empty:
            st.error("Could not compute comparison — check selections.")
        else:
            curr_lt   = float(curr_row["Predicted Lead Time (days)"].values[0])
            alt_lt    = float(alt_row["Predicted Lead Time (days)"].values[0])
            curr_dist = float(curr_row["Distance (km)"].values[0])
            alt_dist  = float(alt_row["Distance (km)"].values[0])
            curr_prof = float(curr_row["Est. Profit Margin (%)"].values[0])
            alt_prof  = float(alt_row["Est. Profit Margin (%)"].values[0])
            curr_fac  = str(curr_row["Factory"].values[0])
            lt_delta  = round(curr_lt - alt_lt, 1)
            prof_delta= round(alt_prof - curr_prof, 2)

            # ── KPI row ───────────────────────────────────────────────────────
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("⏱️ Current Lead Time",  f"{curr_lt:.0f} days")
            m2.metric("⏱️ Alternative Lead Time", f"{alt_lt:.0f} days",
                      delta=f"{-lt_delta:+.1f} days", delta_color="inverse")
            m3.metric("🚚 Distance Change",   f"{alt_dist:.0f} km",
                      delta=f"{alt_dist - curr_dist:+.0f} km", delta_color="inverse")
            m4.metric("💰 Profit Margin Δ",   f"{alt_prof}%",
                      delta=f"{prof_delta:+.2f}%")

            # ── Side-by-side bar ──────────────────────────────────────────────
            comp_fig = go.Figure(data=[
                go.Bar(name=f"Current ({curr_fac})",
                       x=["Lead Time (days)","Distance (×10 km)","Profit Margin (%)"],
                       y=[curr_lt, curr_dist / 10, curr_prof],
                       marker_color="#fd79a8"),
                go.Bar(name=f"Proposed ({wi_alt})",
                       x=["Lead Time (days)","Distance (×10 km)","Profit Margin (%)"],
                       y=[alt_lt, alt_dist / 10, alt_prof],
                       marker_color="#6c5ce7"),
            ])
            comp_fig.update_layout(barmode="group",
                                   title="Current vs Proposed — Side-by-Side",
                                   **PLOTLY_DARK)
            st.plotly_chart(comp_fig, use_container_width=True)

            # ── Gauge chart ───────────────────────────────────────────────────
            gauge_max = max(curr_lt, alt_lt) * 1.3
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=alt_lt,
                delta={"reference": curr_lt, "decreasing": {"color": "#00cec9"},
                       "increasing": {"color": "#e17055"}},
                title={"text": f"Proposed Lead Time (days)<br><sub>vs current {curr_lt:.0f} days</sub>"},
                gauge={
                    "axis": {"range": [0, gauge_max]},
                    "bar":  {"color": "#6c5ce7"},
                    "steps": [{"range": [0, alt_lt], "color": "#1a1a3a"},
                               {"range": [alt_lt, curr_lt] if curr_lt > alt_lt
                                else [curr_lt, alt_lt], "color": "#00cec933"}],
                    "threshold": {"line": {"color": "#fd79a8", "width": 4},
                                  "thickness": 0.75, "value": curr_lt},
                },
            ))
            gauge_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                    font_color="white", height=320)
            st.plotly_chart(gauge_fig, use_container_width=True)

            # ── Verdict ───────────────────────────────────────────────────────
            if lt_delta > 0 and prof_delta >= -2:
                st.success(f"✅ **Recommended!** Moving '{wi_product}' to **{wi_alt}** saves "
                           f"**{lt_delta} days** with minimal profit impact ({prof_delta:+.2f}%).")
            elif lt_delta > 0:
                st.warning(f"⚠️ **Trade-off:** Saves {lt_delta} days but profit margin drops "
                           f"{abs(prof_delta):.2f}%. Review before committing.")
            else:
                st.error(f"❌ **Not recommended.** Alternative factory adds "
                         f"{abs(lt_delta)} days to lead time.")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — RECOMMENDATIONS
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    sec("🏆 Top Factory Reassignment Recommendations")

    rc1, rc2 = st.columns(2)
    rec_region = rc1.selectbox("Region", get_region_categories(df), key="rec_r")
    rec_ship   = rc2.selectbox("Ship Mode", get_ship_modes(df), key="rec_s")

    if st.button("🏆  Generate Recommendations", key="btn_rec"):
        with st.spinner(f"Generating top {top_n} recommendations…"):
            recs = generate_recommendations(
                df, trained, best_name, meta,
                rec_region, rec_ship,
                top_n=top_n, priority=priority_mode,
            )

        if recs.empty:
            st.info("No improvements found for this configuration.")
        else:
            st.success(f"Found **{len(recs)}** high-value reassignment opportunities "
                       f"(priority: **{priority_mode.upper()}**)")

            # ── Summary chart ─────────────────────────────────────────────────
            fig = px.bar(
                recs.reset_index(), x="Product", y="Lead Time Saving (days)",
                color="Est. Profit Margin (%)",
                color_continuous_scale="Viridis",
                text="Lead Time Saving (days)",
                title=f"Top {top_n} Recommendations — Lead Time Savings",
            )
            fig.update_traces(texttemplate="%{text:.1f}d", textposition="outside")
            fig.update_layout(xaxis_tickangle=-25, **PLOTLY_DARK)
            st.plotly_chart(fig, use_container_width=True)

            # ── Recommendation cards ──────────────────────────────────────────
            for rank, row in recs.iterrows():
                with st.expander(f"#{rank}  {row['Product']}  →  {row['Recommended Factory']}",
                                 expanded=(rank == 1)):
                    cc1, cc2, cc3, cc4 = st.columns(4)
                    cc1.metric("Current Factory",    row["Current Factory"])
                    cc2.metric("→ Move To",          row["Recommended Factory"])
                    cc3.metric("⏱️ Time Saved",      f"{row['Lead Time Saving (days)']} days",
                               delta=f"-{row['Lead Time Saving (%)']}%", delta_color="inverse")
                    cc4.metric("💰 Est. Margin",     f"{row['Est. Profit Margin (%)']}%")
                    conf_pct = int(row["Confidence Score"] * 100)
                    st.progress(conf_pct, text=f"Confidence Score: {conf_pct}%")

            # ── Full table ────────────────────────────────────────────────────
            sec("📋 Full Recommendations Table")
            st.dataframe(
                recs,
                use_container_width=True,
                column_config={
                    "Optimization Score": st.column_config.ProgressColumn(
                        "Score", min_value=0,
                        max_value=float(recs["Optimization Score"].max()),
                        format="%.2f",
                    ),
                    "Confidence Score": st.column_config.ProgressColumn(
                        "Confidence", min_value=0, max_value=1, format="%.2f"
                    ),
                },
            )

# ──────────────────────────────────────────────────────────────────────────────
# TAB 5 — RISK & IMPACT
# ──────────────────────────────────────────────────────────────────────────────
with tab5:
    sec("⚠️ Risk & Impact Analysis")

    r1, r2 = st.columns(2)

    with r1:
        # Profit margin risk by product
        margin_prod = dff.groupby("Product Name")["Profit Margin %"].mean().reset_index()
        margin_prod["Risk Level"] = margin_prod["Profit Margin %"].apply(
            lambda x: "🔴 High" if x < 40 else ("🟡 Medium" if x < 55 else "🟢 Low")
        )
        margin_prod = margin_prod.sort_values("Profit Margin %")
        fig = px.bar(
            margin_prod, x="Product Name", y="Profit Margin %",
            color="Risk Level",
            color_discrete_map={"🔴 High": "#e17055", "🟡 Medium": "#fdcb6e", "🟢 Low": "#00cec9"},
            title="Profit Margin Risk by Product",
        )
        fig.update_layout(xaxis_tickangle=-40, **PLOTLY_DARK)
        st.plotly_chart(fig, use_container_width=True)

    with r2:
        # Lead time heatmap: Region × Factory
        lt_heat = dff.groupby(["Region","Factory"])["Lead Time"].mean().reset_index()
        pivot   = lt_heat.pivot(index="Region", columns="Factory", values="Lead Time").fillna(0)
        fig = px.imshow(
            pivot, text_auto=".0f",
            color_continuous_scale="RdYlGn_r",
            title="Avg Lead Time Heatmap  (Region × Factory)",
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#ccc")
        st.plotly_chart(fig, use_container_width=True)

    # ── High-risk route alerts ────────────────────────────────────────────────
    sec("🚨 High-Risk Route Alerts  (Top 20% Slowest)")
    route_lt = dff.groupby(["Factory","Region"])["Lead Time"].mean().reset_index()
    threshold = route_lt["Lead Time"].quantile(0.80)
    high_risk = route_lt[route_lt["Lead Time"] > threshold].sort_values("Lead Time", ascending=False)

    if not high_risk.empty:
        for _, hr in high_risk.iterrows():
            st.error(
                f"⚠️ **{hr['Factory']}  →  {hr['Region']}**  |  "
                f"Avg Lead Time: **{hr['Lead Time']:.0f} days**"
            )
    else:
        st.success("No high-risk routes in current filter.")

    r3, r4 = st.columns(2)

    with r3:
        # Profit share by ship mode
        ps = dff.groupby("Ship Mode")["Gross Profit"].sum().reset_index()
        fig = px.pie(ps, values="Gross Profit", names="Ship Mode",
                     title="Gross Profit Share by Ship Mode",
                     color_discrete_sequence=PALETTE, hole=0.42)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#ccc")
        st.plotly_chart(fig, use_container_width=True)

    with r4:
        # Lead time trend by distance bucket
        dff2 = dff.copy()
        dff2["Dist Bucket"] = pd.cut(
            dff2["Distance_km"], bins=5,
            labels=["Very Close","Close","Mid","Far","Very Far"]
        )
        trend = dff2.groupby("Dist Bucket", observed=True)["Lead Time"].mean().reset_index()
        fig = px.line(trend, x="Dist Bucket", y="Lead Time", markers=True,
                      title="Avg Lead Time vs Distance Bucket",
                      color_discrete_sequence=["#6c5ce7"])
        fig.update_layout(**PLOTLY_DARK)
        st.plotly_chart(fig, use_container_width=True)

    # ── Executive summary table ───────────────────────────────────────────────
    sec("📋 Executive Risk Summary")
    high_risk_count = len(high_risk)
    low_margin_prods = int((dff.groupby("Product Name")["Profit Margin %"].mean() < 40).sum())
    worst_region = dff.groupby("Region")["Lead Time"].mean().idxmax()
    best_region  = dff.groupby("Region")["Lead Time"].mean().idxmin()
    worst_factory= dff.groupby("Factory")["Lead Time"].mean().idxmax()

    exec_df = pd.DataFrame({
        "Metric": [
            "High-risk routes (top 20% lead time)",
            "Products with profit margin < 40%",
            "Average profit margin (filtered data)",
            "Highest avg lead time — Region",
            "Lowest avg lead time — Region",
            "Slowest Factory",
        ],
        "Value": [
            str(high_risk_count),
            str(low_margin_prods),
            f"{dff['Profit Margin %'].mean():.1f}%",
            worst_region,
            best_region,
            worst_factory,
        ],
    })
    st.dataframe(exec_df, use_container_width=True, hide_index=True)
