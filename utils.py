import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st

DATA_PATH = "Online Energy Drink Orders.csv"

AGE_BINS = [17, 25, 35, 45, 55, 65, 71]
AGE_LABELS = ["18-25", "26-35", "36-45", "46-55", "56-65", "66+"]

LOYALTY_BINS = [0, 10, 25, 40, 51]
LOYALTY_LABELS = ["Low (1-10)", "Medium (11-25)", "High (26-40)", "Very High (41-50)"]

PRODUCT_ORDER = [
    "Gatorade Sports Drink",
    "Gatorade No Sugar",
    "G Active-No Sugar",
    "Gatorade Powder",
]

FLAVOUR_ORDER = ["Berry", "Blue Bolt", "Grape", "Lemon-Lime", "Orange", "Tropical"]

COLORS = {
    "primary": "#0068C9",
    "secondary": "#83C9FF",
    "accent": "#FF4B4B",
    "success": "#21C354",
    "warning": "#FACA2B",
    "neutral": "#808495",
    "palette": [
        "#0068C9", "#83C9FF", "#FF4B4B", "#21C354",
        "#FACA2B", "#FF8C00", "#A855F7", "#808495",
    ],
}


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    df["Age Group"] = pd.cut(df["Age"], bins=AGE_BINS, labels=AGE_LABELS, right=True)
    df["Loyalty Tier"] = pd.cut(
        df["Previous Purchases"], bins=LOYALTY_BINS, labels=LOYALTY_LABELS, right=True
    )
    df["Revenue"] = df["Purchase Amount (AUD)"]
    return df


def apply_filters(
    df: pd.DataFrame,
    gender: str,
    age_range: tuple[int, int],
    products: list[str],
    flavours: list[str],
) -> pd.DataFrame:
    filtered = df.copy()
    if gender != "All":
        filtered = filtered[filtered["Gender"] == gender]
    filtered = filtered[
        (filtered["Age"] >= age_range[0]) & (filtered["Age"] <= age_range[1])
    ]
    if products:
        filtered = filtered[filtered["Item Purchased"].isin(products)]
    if flavours:
        filtered = filtered[filtered["Flavour"].isin(flavours)]
    return filtered


@st.cache_data
def run_clustering(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    features = df[["Age", "Purchase Amount (AUD)", "Previous Purchases", "Review Rating"]].copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df = df.copy()
    df["Cluster"] = kmeans.fit_predict(scaled)

    cluster_profiles = (
        df.groupby("Cluster")
        .agg(
            avg_age=("Age", "mean"),
            avg_spend=("Purchase Amount (AUD)", "mean"),
            avg_prev_purchases=("Previous Purchases", "mean"),
            avg_rating=("Review Rating", "mean"),
            count=("Customer ID", "count"),
            top_product=("Item Purchased", lambda x: x.mode().iloc[0]),
            top_flavour=("Flavour", lambda x: x.mode().iloc[0]),
        )
        .reset_index()
    )

    label_map = _generate_cluster_labels(cluster_profiles)
    df["Segment"] = df["Cluster"].map(label_map)
    cluster_profiles["Segment"] = cluster_profiles["Cluster"].map(label_map)

    return df, cluster_profiles


def _generate_cluster_labels(profiles: pd.DataFrame) -> dict[int, str]:
    labels = {}
    for _, row in profiles.iterrows():
        cid = row["Cluster"]
        age = row["avg_age"]
        loyalty = row["avg_prev_purchases"]
        rating = row["avg_rating"]

        if age < 35 and loyalty < 20:
            name = "Young Explorers"
        elif age < 35 and loyalty >= 20:
            name = "Young Loyalists"
        elif age >= 35 and loyalty >= 30 and rating >= 3.8:
            name = "Satisfied Veterans"
        elif age >= 35 and loyalty >= 30:
            name = "Loyal Veterans"
        elif age >= 50 and loyalty < 20:
            name = "Mature Casuals"
        elif rating >= 4.0:
            name = "Happy Regulars"
        elif rating < 3.0:
            name = "At-Risk Buyers"
        else:
            name = f"Segment {cid + 1}"

        if name in labels.values():
            name = f"{name} ({int(age)}s)"
        labels[cid] = name

    return labels


def format_currency(val: float) -> str:
    return f"${val:,.2f}"


def format_pct(val: float) -> str:
    return f"{val:.1f}%"
