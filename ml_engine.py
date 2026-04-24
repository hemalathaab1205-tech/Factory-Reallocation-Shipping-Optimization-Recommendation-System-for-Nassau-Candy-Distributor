"""
ml_engine.py
Nassau Candy Distributor – ML Modeling & Scenario Simulation Engine
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from data_engine import FACTORIES, PRODUCT_FACTORY, SHIP_MODE_ENC, _dist_km, _get_coord

warnings.filterwarnings("ignore")

FEATURE_COLS = ["Ship Mode Enc", "Region Enc", "Factory Enc", "Distance_km", "Units", "Product Enc"]


# ─────────────────────────────────────────────────────────────────────────────
# Train all three models, return results and the best one
# ─────────────────────────────────────────────────────────────────────────────
def train_models(df: pd.DataFrame):
    data = df[FEATURE_COLS + ["Lead Time"]].dropna()
    X = data[FEATURE_COLS]
    y = data["Lead Time"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURE_COLS)
    X_test_s  = pd.DataFrame(scaler.transform(X_test),      columns=FEATURE_COLS)

    models = {
        "Linear Regression":    LinearRegression(),
        "Random Forest":        RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
        "Gradient Boosting":    GradientBoostingRegressor(n_estimators=150, learning_rate=0.05,
                                                          max_depth=5, random_state=42),
    }

    results = {}
    trained = {}

    for name, mdl in models.items():
        if name == "Linear Regression":
            mdl.fit(X_train_s, y_train)
            preds = mdl.predict(X_test_s)
            trained[name] = {"model": mdl, "scaler": scaler}
        else:
            mdl.fit(X_train, y_train)
            preds = mdl.predict(X_test)
            trained[name] = {"model": mdl, "scaler": None}

        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae  = float(mean_absolute_error(y_test, preds))
        r2   = float(r2_score(y_test, preds))
        results[name] = {"RMSE": round(rmse, 2), "MAE": round(mae, 2), "R²": round(r2, 4)}

    # Best model = highest R²
    best_name = max(results, key=lambda k: results[k]["R²"])

    # Store region & factory categories for encoding during prediction
    region_cats  = sorted(df["Region"].dropna().unique().tolist())
    product_cats = sorted(df["Product Name"].dropna().unique().tolist())
    factory_cats = sorted(df["Factory"].dropna().unique().tolist())

    meta = {
        "region_cats":  region_cats,
        "product_cats": product_cats,
        "factory_cats": factory_cats,
    }

    return trained, results, best_name, meta


# ─────────────────────────────────────────────────────────────────────────────
# Predict lead time for a single feature vector
# ─────────────────────────────────────────────────────────────────────────────
def predict_lead_time(trained, best_name, meta, ship_mode, region, factory_name,
                      distance_km, units, product_name):
    region_enc  = meta["region_cats"].index(region)   if region       in meta["region_cats"]  else 0
    factory_enc = meta["factory_cats"].index(factory_name) if factory_name in meta["factory_cats"] else 0
    product_enc = meta["product_cats"].index(product_name) if product_name in meta["product_cats"] else 0
    ship_enc    = SHIP_MODE_ENC.get(ship_mode, 3)

    X = pd.DataFrame([[ship_enc, region_enc, factory_enc, distance_km, units, product_enc]],
                     columns=FEATURE_COLS)

    bundle = trained[best_name]
    mdl    = bundle["model"]
    scaler = bundle["scaler"]

    if scaler:
        X = pd.DataFrame(scaler.transform(X), columns=FEATURE_COLS)

    pred = mdl.predict(X)[0]
    return max(0.0, round(float(pred), 1))


# ─────────────────────────────────────────────────────────────────────────────
# Simulate reassigning a product to every possible factory
# ─────────────────────────────────────────────────────────────────────────────
def simulate_factory_reassignment(df, trained, best_name, meta,
                                  product_name, region, ship_mode, units=3):
    # Representative state for the region
    region_rows = df[df["Region"] == region]
    state = region_rows["State/Province"].mode().iloc[0] if not region_rows.empty else "Texas"

    current_factory = PRODUCT_FACTORY.get(product_name, list(FACTORIES.keys())[0])

    # Base profit margin for this product
    product_rows = df[df["Product Name"] == product_name]
    base_margin  = float(product_rows["Profit Margin %"].mean()) if not product_rows.empty else 60.0

    rows = []
    for factory_name in FACTORIES:
        dist = _dist_km(state, factory_name)

        pred_lt = predict_lead_time(
            trained, best_name, meta,
            ship_mode, region, factory_name,
            dist, units, product_name
        )

        # Profit impact: longer distance = slightly higher logistics cost
        dist_penalty  = (dist / 6000.0) * 4.0   # max ~4% at 6000 km
        adj_margin    = round(max(0.0, base_margin - dist_penalty), 2)
        confidence    = round(min(1.0, max(0.50, 1.0 - dist / 8000.0)), 2)

        rows.append({
            "Factory":                        factory_name,
            "Is Current":                     factory_name == current_factory,
            "Distance (km)":                  round(dist, 1),
            "Predicted Lead Time (days)":     pred_lt,
            "Est. Profit Margin (%)":         adj_margin,
            "Confidence Score":               confidence,
        })

    sim_df = pd.DataFrame(rows).sort_values("Predicted Lead Time (days)").reset_index(drop=True)

    # Lead-time saving vs current
    current_lt_series = sim_df.loc[sim_df["Is Current"], "Predicted Lead Time (days)"]
    current_lt = current_lt_series.values[0] if len(current_lt_series) else 0.0
    sim_df["Lead Time Saving (days)"] = (current_lt - sim_df["Predicted Lead Time (days)"]).round(1)
    sim_df["Rank"] = range(1, len(sim_df) + 1)

    return sim_df


# ─────────────────────────────────────────────────────────────────────────────
# Generate Top-N recommendations across all products
# ─────────────────────────────────────────────────────────────────────────────
def generate_recommendations(df, trained, best_name, meta,
                              region, ship_mode, top_n=5,
                              priority="balanced"):
    recs = []

    for product_name in sorted(df["Product Name"].dropna().unique()):
        sim = simulate_factory_reassignment(
            df, trained, best_name, meta, product_name, region, ship_mode
        )

        current_row = sim[sim["Is Current"]]
        best_alt    = sim[~sim["Is Current"]].head(1)

        if current_row.empty or best_alt.empty:
            continue

        curr_lt  = float(current_row["Predicted Lead Time (days)"].values[0])
        alt_lt   = float(best_alt["Predicted Lead Time (days)"].values[0])
        alt_fac  = str(best_alt["Factory"].values[0])
        margin   = float(best_alt["Est. Profit Margin (%)"].values[0])
        conf     = float(best_alt["Confidence Score"].values[0])
        lt_save  = round(curr_lt - alt_lt, 1)
        lt_pct   = round((lt_save / curr_lt * 100) if curr_lt > 0 else 0.0, 1)

        if lt_save <= 0:
            continue  # Only recommend if there's a genuine improvement

        if priority == "speed":
            score = lt_pct
        elif priority == "profit":
            score = margin
        else:  # balanced
            score = lt_pct * 0.6 + margin * 0.4

        recs.append({
            "Product":                    product_name,
            "Division":                   df.loc[df["Product Name"] == product_name, "Division"].mode().iloc[0],
            "Current Factory":            PRODUCT_FACTORY.get(product_name, "Unknown"),
            "Recommended Factory":        alt_fac,
            "Lead Time Saving (days)":    lt_save,
            "Lead Time Saving (%)":       lt_pct,
            "Est. Profit Margin (%)":     margin,
            "Confidence Score":           conf,
            "Optimization Score":         round(score, 2),
        })

    recs_df = (pd.DataFrame(recs)
               .sort_values("Optimization Score", ascending=False)
               .head(top_n)
               .reset_index(drop=True))
    recs_df.index += 1  # rank from 1
    return recs_df
