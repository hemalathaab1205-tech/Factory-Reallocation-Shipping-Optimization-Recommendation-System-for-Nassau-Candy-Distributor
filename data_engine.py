"""
data_engine.py
Nassau Candy Distributor – Data Preparation & Feature Engineering
Pure Python + pandas/numpy only (no geopy required)
"""

import math
import pandas as pd
import numpy as np


# ── Factory coordinates ──────────────────────────────────────────────────────
FACTORIES = {
    "Lot's O' Nuts":     {"lat": 32.881893, "lon": -111.768036},
    "Wicked Choccy's":   {"lat": 32.076176, "lon": -81.088371},
    "Sugar Shack":       {"lat": 48.119140, "lon": -96.181150},
    "Secret Factory":    {"lat": 41.446333, "lon": -90.565487},
    "The Other Factory": {"lat": 35.117500, "lon": -89.971107},
}

# ── Product → Factory mapping ─────────────────────────────────────────────────
PRODUCT_FACTORY = {
    "Wonka Bar - Nutty Crunch Surprise":  "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows":          "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious":     "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate":         "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel":  "Wicked Choccy's",
    "Laffy Taffy":                        "Sugar Shack",
    "SweeTARTS":                          "Sugar Shack",
    "Nerds":                              "Sugar Shack",
    "Fun Dip":                            "Sugar Shack",
    "Fizzy Lifting Drinks":               "Sugar Shack",
    "Everlasting Gobstopper":             "Secret Factory",
    "Hair Toffee":                        "The Other Factory",
    "Lickable Wallpaper":                 "Secret Factory",
    "Wonka Gum":                          "Secret Factory",
    "Kazookles":                          "The Other Factory",
}

# ── Ship mode encoding ────────────────────────────────────────────────────────
SHIP_MODE_ENC = {
    "Same Day":      0,
    "First Class":   1,
    "Second Class":  2,
    "Standard Class": 3,
}

# ── US state → approximate (lat, lon) ────────────────────────────────────────
# Used to estimate city location when city is not in our cache
STATE_COORDS = {
    "Alabama": (32.8, -86.8), "Alaska": (64.2, -153.4), "Arizona": (34.3, -111.1),
    "Arkansas": (34.8, -92.2), "California": (36.8, -119.4), "Colorado": (38.9, -105.5),
    "Connecticut": (41.6, -72.7), "Delaware": (39.0, -75.5), "Florida": (27.7, -81.7),
    "Georgia": (32.2, -83.4), "Hawaii": (20.3, -156.4), "Idaho": (44.4, -114.5),
    "Illinois": (40.0, -89.2), "Indiana": (39.9, -86.3), "Iowa": (42.1, -93.5),
    "Kansas": (38.5, -98.4), "Kentucky": (37.7, -84.9), "Louisiana": (31.2, -91.8),
    "Maine": (45.4, -69.0), "Maryland": (39.1, -76.8), "Massachusetts": (42.2, -71.5),
    "Michigan": (43.3, -84.5), "Minnesota": (45.7, -93.9), "Mississippi": (32.7, -89.7),
    "Missouri": (38.3, -92.5), "Montana": (46.9, -110.5), "Nebraska": (41.5, -99.9),
    "Nevada": (38.5, -117.1), "New Hampshire": (43.7, -71.6), "New Jersey": (40.1, -74.5),
    "New Mexico": (34.3, -106.0), "New York": (42.2, -74.9), "North Carolina": (35.6, -79.4),
    "North Dakota": (47.5, -100.4), "Ohio": (40.4, -82.8), "Oklahoma": (35.6, -97.0),
    "Oregon": (44.1, -120.5), "Pennsylvania": (40.6, -77.2), "Rhode Island": (41.7, -71.6),
    "South Carolina": (33.9, -81.0), "South Dakota": (44.4, -100.2), "Tennessee": (35.7, -86.7),
    "Texas": (31.1, -97.6), "Utah": (39.3, -111.1), "Vermont": (44.1, -72.7),
    "Virginia": (37.8, -78.2), "Washington": (47.4, -121.5), "West Virginia": (38.6, -80.6),
    "Wisconsin": (44.3, -89.6), "Wyoming": (42.8, -107.3),
    # Canada
    "Ontario": (50.0, -85.0), "Quebec": (53.0, -70.0), "British Columbia": (53.7, -127.6),
    "Alberta": (55.0, -115.0), "Manitoba": (55.0, -97.0), "Saskatchewan": (55.0, -106.0),
    "Nova Scotia": (45.0, -63.0), "New Brunswick": (46.5, -66.5),
    "Prince Edward Island": (46.4, -63.2), "Newfoundland and Labrador": (53.1, -57.7),
}

DEFAULT_COORD = (39.5, -98.35)  # Geographic centre of US


def _haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km — no external libraries needed."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _get_coord(state: str) -> tuple:
    return STATE_COORDS.get(state, DEFAULT_COORD)


def _dist_km(state: str, factory_name: str) -> float:
    coord = _get_coord(state)
    f = FACTORIES[factory_name]
    return _haversine_km(coord[0], coord[1], f["lat"], f["lon"])


def load_and_prepare(path: str) -> pd.DataFrame:
    """Load CSV, engineer all features, return clean DataFrame."""
    df = pd.read_csv(path)

    # ── Parse dates ───────────────────────────────────────────────────────────
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True, errors="coerce")
    df["Ship Date"]  = pd.to_datetime(df["Ship Date"],  dayfirst=True, errors="coerce")

    # ── Lead time (days) ──────────────────────────────────────────────────────
    df["Lead Time"] = (df["Ship Date"] - df["Order Date"]).dt.days

    # Keep only rows with positive lead times
    df = df[df["Lead Time"] > 0].copy()

    # ── Factory assignment ────────────────────────────────────────────────────
    df["Factory"] = df["Product Name"].map(PRODUCT_FACTORY)

    # ── Distance: state centroid → factory ────────────────────────────────────
    df["Distance_km"] = df.apply(
        lambda r: _dist_km(r["State/Province"], r["Factory"])
        if pd.notna(r["Factory"]) else np.nan,
        axis=1
    )

    # ── Financial metrics ─────────────────────────────────────────────────────
    df["Profit Margin %"] = (df["Gross Profit"] / df["Sales"] * 100).round(2)

    # ── Encodings ─────────────────────────────────────────────────────────────
    df["Ship Mode Enc"] = df["Ship Mode"].map(SHIP_MODE_ENC).fillna(3)

    region_cats = sorted(df["Region"].dropna().unique())
    df["Region Enc"] = df["Region"].apply(
        lambda x: region_cats.index(x) if x in region_cats else 0
    )

    factory_cats = sorted(df["Factory"].dropna().unique())
    df["Factory Enc"] = df["Factory"].apply(
        lambda x: factory_cats.index(x) if x in factory_cats else 0
    )

    # ── Product encoding ──────────────────────────────────────────────────────
    product_cats = sorted(df["Product Name"].dropna().unique())
    df["Product Enc"] = df["Product Name"].apply(
        lambda x: product_cats.index(x) if x in product_cats else 0
    )

    return df


def get_summary_kpis(df: pd.DataFrame) -> dict:
    return {
        "Total Orders":       f"{len(df):,}",
        "Avg Lead Time":      f"{df['Lead Time'].mean():.1f} days",
        "Total Sales":        f"${df['Sales'].sum():,.2f}",
        "Total Gross Profit": f"${df['Gross Profit'].sum():,.2f}",
        "Avg Profit Margin":  f"{df['Profit Margin %'].mean():.1f}%",
        "Avg Distance":       f"{df['Distance_km'].mean():,.1f} km",
    }


def get_region_categories(df: pd.DataFrame) -> list:
    return sorted(df["Region"].dropna().unique().tolist())


def get_ship_modes(df: pd.DataFrame) -> list:
    return sorted(df["Ship Mode"].dropna().unique().tolist())


def get_products(df: pd.DataFrame) -> list:
    return sorted(df["Product Name"].dropna().unique().tolist())


def get_factory_list() -> list:
    return list(FACTORIES.keys())
