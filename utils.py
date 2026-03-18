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
    "primary": "#4FC3F7",
    "secondary": "#81D4FA",
    "accent": "#FF7043",
    "success": "#66BB6A",
    "warning": "#FFD54F",
    "neutral": "#B0BEC5",
    "palette": [
        "#4FC3F7", "#FF7043", "#66BB6A", "#FFD54F",
        "#AB47BC", "#26A69A", "#EF5350", "#78909C",
    ],
    "muted_blue": "#5C6BC0",
    "muted_pink": "#EC407A",
    "grid": "rgba(255,255,255,0.08)",
    "text": "#E0E0E0",
    "text_secondary": "#9E9E9E",
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


def dark_layout(fig, title: str = "", subtitle: str = "", height: int | None = None):
    """Apply a consistent dark theme to any Plotly figure."""
    layout_args = dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", color=COLORS["text"], size=13),
        title=dict(
            text=f"<b>{title}</b>" + (f"<br><span style='font-size:12px;color:{COLORS['text_secondary']}'>{subtitle}</span>" if subtitle else ""),
            font=dict(size=17),
            x=0.0,
            xanchor="left",
        ) if title else None,
        margin=dict(l=50, r=20, t=70 if title else 30, b=50),
        xaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
        yaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
        ),
    )
    if height:
        layout_args["height"] = height
    fig.update_layout(**{k: v for k, v in layout_args.items() if v is not None})
    return fig


def add_mean_line(fig, values, axis="x", name="Mean", color=None):
    """Add a dashed mean reference line to a figure."""
    mean_val = values.mean()
    line_color = color or COLORS["accent"]
    if axis == "x":
        fig.add_vline(
            x=mean_val, line_dash="dash", line_color=line_color, line_width=2,
            annotation_text=f"{name}: {mean_val:.1f}",
            annotation_font=dict(color=line_color, size=12),
            annotation_position="top",
        )
    else:
        fig.add_hline(
            y=mean_val, line_dash="dash", line_color=line_color, line_width=2,
            annotation_text=f"{name}: {mean_val:.1f}",
            annotation_font=dict(color=line_color, size=12),
        )
    return fig


def get_data_summary(df: pd.DataFrame) -> str:
    """Generate a text summary of the dataset for LLM context."""
    lines = [
        f"Dataset: Online Energy Drink Orders ({len(df):,} rows after filters)",
        f"Columns: {', '.join(df.columns)}",
        "",
        "Numerical summaries:",
        f"  Age: {df['Age'].min()}-{df['Age'].max()}, mean={df['Age'].mean():.1f}",
        f"  Purchase Amount (AUD): ${df['Purchase Amount (AUD)'].min():.2f}-${df['Purchase Amount (AUD)'].max():.2f}, mean=${df['Purchase Amount (AUD)'].mean():.2f}",
        f"  Review Rating: {df['Review Rating'].min():.1f}-{df['Review Rating'].max():.1f}, mean={df['Review Rating'].mean():.2f}",
        f"  Previous Purchases: {df['Previous Purchases'].min()}-{df['Previous Purchases'].max()}, mean={df['Previous Purchases'].mean():.1f}",
        "",
        "Categorical values:",
        f"  Gender: {', '.join(df['Gender'].unique())}",
        f"  Items: {', '.join(sorted(df['Item Purchased'].unique()))}",
        f"  Flavours: {', '.join(sorted(df['Flavour'].unique()))}",
        f"  Payment Methods: {', '.join(sorted(df['Payment Method'].unique()))}",
    ]
    return "\n".join(lines)


def format_currency(val: float) -> str:
    return f"${val:,.2f}"


def format_pct(val: float) -> str:
    return f"{val:.1f}%"
