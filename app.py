import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from utils import (
    load_data,
    apply_filters,
    run_clustering,
    format_currency,
    format_pct,
    dark_layout,
    add_mean_line,
    get_data_summary,
    PRODUCT_ORDER,
    FLAVOUR_ORDER,
    AGE_LABELS,
    LOYALTY_LABELS,
    COLORS,
)

st.set_page_config(
    page_title="Energy Drink Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

PLOTLY_COLORS = COLORS["palette"]


# ── Sidebar ──────────────────────────────────────────────────────────────────

df_raw = load_data()

st.sidebar.title("⚡ Filters")

gender = st.sidebar.radio("Gender", ["All", "Male", "Female"], horizontal=True)

age_min, age_max = int(df_raw["Age"].min()), int(df_raw["Age"].max())
age_range = st.sidebar.slider("Age Range", age_min, age_max, (age_min, age_max))

all_products = sorted(df_raw["Item Purchased"].unique())
products = st.sidebar.multiselect("Products", all_products, default=all_products)

all_flavours = sorted(df_raw["Flavour"].unique())
flavours = st.sidebar.multiselect("Flavours", all_flavours, default=all_flavours)

df = apply_filters(df_raw, gender, age_range, products, flavours)

if df.empty:
    st.warning("No data matches the current filters. Adjust the sidebar filters.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.caption(f"Showing **{len(df):,}** of {len(df_raw):,} orders")


# ── Tabs ─────────────────────────────────────────────────────────────────────

tabs = st.tabs([
    "📊 Executive Summary",
    "🎯 Customer Segmentation",
    "⭐ Satisfaction Drivers",
    "📦 Product Mix & Revenue",
    "👥 Demographic Patterns",
    "💳 Payment & Loyalty",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: EXECUTIVE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[0]:
    st.header("Executive Summary")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Orders", f"{len(df):,}")
    k2.metric("Total Revenue", format_currency(df["Revenue"].sum()))
    k3.metric("Avg Rating", f"{df['Review Rating'].mean():.2f}")
    k4.metric("Avg Previous Purchases", f"{df['Previous Purchases'].mean():.1f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        rev_by_product = (
            df.groupby("Item Purchased")["Revenue"]
            .sum()
            .reindex(PRODUCT_ORDER)
            .reset_index()
        )
        rev_by_product = rev_by_product.sort_values("Revenue", ascending=True)
        fig = px.bar(
            rev_by_product,
            y="Item Purchased",
            x="Revenue",
            color="Item Purchased",
            color_discrete_sequence=PLOTLY_COLORS,
            orientation="h",
            text=rev_by_product["Revenue"].apply(lambda v: f"${v:,.0f}"),
        )
        fig.update_traces(textposition="outside", textfont_size=12)
        dark_layout(fig, "Revenue by Product", "Sorted by total revenue, highest at top")
        fig.update_layout(showlegend=False, yaxis_title="", xaxis_title="Revenue (AUD)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        mean_rating = df["Review Rating"].mean()
        fig = px.histogram(
            df,
            x="Review Rating",
            nbins=15,
            color_discrete_sequence=[COLORS["primary"]],
            histnorm="percent",
        )
        fig.update_traces(marker_line_width=0.5, marker_line_color="rgba(0,0,0,0.3)")
        add_mean_line(fig, df["Review Rating"], axis="x", name="Avg", color=COLORS["accent"])
        dark_layout(fig, "Ratings are evenly spread (2.5 - 5.0)", f"Average rating: {mean_rating:.2f} — no strong skew")
        fig.update_layout(xaxis_title="Review Rating", yaxis_title="% of Orders")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        gender_counts = df["Gender"].value_counts().reset_index()
        gender_counts.columns = ["Gender", "Count"]
        gender_counts["Pct"] = (gender_counts["Count"] / gender_counts["Count"].sum() * 100).round(1)
        fig = px.pie(
            gender_counts,
            values="Count",
            names="Gender",
            color_discrete_sequence=[COLORS["primary"], COLORS["accent"]],
            hole=0.45,
        )
        fig.update_traces(
            textinfo="label+percent",
            textfont_size=14,
            marker=dict(line=dict(color="rgba(0,0,0,0.3)", width=1)),
        )
        dark_layout(fig, "Gender Split")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        mean_age = df["Age"].mean()
        fig = px.histogram(
            df,
            x="Age",
            nbins=20,
            color_discrete_sequence=[COLORS["secondary"]],
            histnorm="percent",
        )
        fig.update_traces(marker_line_width=0.5, marker_line_color="rgba(0,0,0,0.3)")
        add_mean_line(fig, df["Age"], axis="x", name="Avg Age", color=COLORS["accent"])
        dark_layout(fig, "Age is uniformly distributed (18-75)", f"Average age: {mean_age:.0f} — all age groups buy equally")
        fig.update_layout(xaxis_title="Age", yaxis_title="% of Orders")
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: CUSTOMER SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[1]:
    st.header("Customer Segmentation")
    st.caption("KMeans clustering on Age, Spend, Purchase History, and Rating")

    n_clusters = st.slider("Number of segments", 2, 6, 4)

    df_clustered, profiles = run_clustering(df, n_clusters)

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = px.scatter(
            df_clustered,
            x="Age",
            y="Previous Purchases",
            color="Segment",
            hover_data=["Review Rating", "Item Purchased", "Flavour", "Purchase Amount (AUD)"],
            color_discrete_sequence=PLOTLY_COLORS,
            opacity=0.45,
            marginal_x="box",
            marginal_y="box",
        )
        fig.update_traces(marker=dict(size=5, line=dict(width=0)), selector=dict(mode="markers"))
        dark_layout(fig, "Customer Segments: Age vs. Purchase History",
                    "Each dot is a customer; boxes show distributions per segment", height=500)
        fig.update_layout(legend_title="Segment")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        seg_counts = df_clustered["Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        seg_counts["Pct"] = (seg_counts["Count"] / seg_counts["Count"].sum() * 100).round(1)
        fig = px.pie(
            seg_counts,
            values="Count",
            names="Segment",
            color_discrete_sequence=PLOTLY_COLORS,
            hole=0.4,
        )
        fig.update_traces(
            textinfo="label+percent",
            textfont_size=11,
            marker=dict(line=dict(color="rgba(0,0,0,0.3)", width=1)),
        )
        dark_layout(fig, "Segment Sizes", height=500)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Segment Profiles")
    display_profiles = profiles[
        ["Segment", "count", "avg_age", "avg_spend", "avg_prev_purchases", "avg_rating", "top_product", "top_flavour"]
    ].copy()
    display_profiles.columns = [
        "Segment", "Customers", "Avg Age", "Avg Spend (AUD)", "Avg Prev Purchases",
        "Avg Rating", "Top Product", "Top Flavour",
    ]
    display_profiles["Avg Age"] = display_profiles["Avg Age"].round(1)
    display_profiles["Avg Spend (AUD)"] = display_profiles["Avg Spend (AUD)"].round(2)
    display_profiles["Avg Prev Purchases"] = display_profiles["Avg Prev Purchases"].round(1)
    display_profiles["Avg Rating"] = display_profiles["Avg Rating"].round(2)
    st.dataframe(display_profiles, use_container_width=True, hide_index=True)

    fig2 = px.scatter(
        df_clustered,
        x="Review Rating",
        y="Purchase Amount (AUD)",
        color="Segment",
        hover_data=["Age", "Previous Purchases"],
        color_discrete_sequence=PLOTLY_COLORS,
        opacity=0.35,
        marginal_x="violin",
        marginal_y="violin",
    )
    fig2.update_traces(marker=dict(size=4, line=dict(width=0)), selector=dict(mode="markers"))
    dark_layout(fig2, "Segments: Rating vs. Spend",
                "Violin plots on margins show density — look for clusters")
    st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: SATISFACTION DRIVERS
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[2]:
    st.header("Satisfaction Drivers")

    col1, col2 = st.columns(2)

    with col1:
        rating_by_product = (
            df.groupby("Item Purchased")["Review Rating"]
            .mean()
            .reindex(PRODUCT_ORDER)
            .reset_index()
        )
        rating_by_product = rating_by_product.sort_values("Review Rating", ascending=True)
        fig = px.bar(
            rating_by_product,
            y="Item Purchased",
            x="Review Rating",
            color="Item Purchased",
            color_discrete_sequence=PLOTLY_COLORS,
            orientation="h",
            text=rating_by_product["Review Rating"].apply(lambda v: f"{v:.2f}"),
        )
        fig.update_traces(textposition="outside", textfont_size=12)
        dark_layout(fig, "Avg Rating by Product", "All products rate similarly (~3.5)")
        fig.update_layout(showlegend=False, yaxis_title="", xaxis_title="Avg Rating")
        fig.update_xaxes(range=[0, 5])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        rating_by_flavour = (
            df.groupby("Flavour")["Review Rating"]
            .mean()
            .reindex(FLAVOUR_ORDER)
            .reset_index()
        )
        rating_by_flavour = rating_by_flavour.sort_values("Review Rating", ascending=True)
        fig = px.bar(
            rating_by_flavour,
            y="Flavour",
            x="Review Rating",
            color="Flavour",
            color_discrete_sequence=PLOTLY_COLORS,
            orientation="h",
            text=rating_by_flavour["Review Rating"].apply(lambda v: f"{v:.2f}"),
        )
        fig.update_traces(textposition="outside", textfont_size=12)
        dark_layout(fig, "Avg Rating by Flavour", "Minimal variation across flavours")
        fig.update_layout(showlegend=False, yaxis_title="", xaxis_title="Avg Rating")
        fig.update_xaxes(range=[0, 5])
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Rating Heatmap: Product × Flavour")
    heatmap_data = df.pivot_table(
        values="Review Rating",
        index="Item Purchased",
        columns="Flavour",
        aggfunc="mean",
    )
    heatmap_data = heatmap_data.reindex(index=PRODUCT_ORDER, columns=FLAVOUR_ORDER)

    fig = px.imshow(
        heatmap_data,
        text_auto=".2f",
        color_continuous_scale="RdYlGn",
        aspect="auto",
    )
    dark_layout(fig, "Avg Rating: Product × Flavour", "Green = higher rated combos, Red = lower")
    fig.update_layout(xaxis_title="Flavour", yaxis_title="Product")
    fig.update_traces(textfont_size=13)
    st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        rating_by_age = (
            df.groupby("Age Group", observed=True)["Review Rating"]
            .mean()
            .reindex(AGE_LABELS)
            .reset_index()
        )
        fig = px.line(
            rating_by_age,
            x="Age Group",
            y="Review Rating",
            markers=True,
            color_discrete_sequence=[COLORS["primary"]],
        )
        fig.update_traces(line=dict(width=3), marker=dict(size=10))
        dark_layout(fig, "Avg Rating by Age Group", "Flat trend — age doesn't drive satisfaction")
        fig.update_yaxes(range=[2.5, 5])
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        rating_by_gender = (
            df.groupby("Gender")["Review Rating"].mean().reset_index()
        )
        fig = px.bar(
            rating_by_gender,
            x="Gender",
            y="Review Rating",
            color="Gender",
            color_discrete_sequence=[COLORS["primary"], COLORS["accent"]],
            text=rating_by_gender["Review Rating"].apply(lambda v: f"{v:.2f}"),
        )
        fig.update_traces(textposition="outside", textfont_size=13)
        dark_layout(fig, "Avg Rating by Gender", "No significant gender difference")
        fig.update_layout(showlegend=False)
        fig.update_yaxes(range=[0, 5])
        st.plotly_chart(fig, use_container_width=True)

    rating_by_payment = (
        df.groupby("Payment Method")["Review Rating"].mean().sort_values().reset_index()
    )
    fig = px.bar(
        rating_by_payment,
        x="Review Rating",
        y="Payment Method",
        orientation="h",
        color_discrete_sequence=[COLORS["muted_blue"]],
        text=rating_by_payment["Review Rating"].apply(lambda v: f"{v:.2f}"),
    )
    fig.update_traces(textposition="outside", textfont_size=12)
    dark_layout(fig, "Avg Rating by Payment Method", "Sorted lowest to highest")
    fig.update_xaxes(range=[0, 5])
    fig.update_layout(yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Best & Worst Product-Flavour Combos")
    combo_ratings = (
        df.groupby(["Item Purchased", "Flavour"])["Review Rating"]
        .agg(["mean", "count"])
        .reset_index()
    )
    combo_ratings.columns = ["Product", "Flavour", "Avg Rating", "Orders"]
    combo_ratings = combo_ratings[combo_ratings["Orders"] >= 10]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top 5 Combos**")
        top5 = combo_ratings.nlargest(5, "Avg Rating").reset_index(drop=True)
        top5["Avg Rating"] = top5["Avg Rating"].round(2)
        st.dataframe(top5, use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Bottom 5 Combos**")
        bottom5 = combo_ratings.nsmallest(5, "Avg Rating").reset_index(drop=True)
        bottom5["Avg Rating"] = bottom5["Avg Rating"].round(2)
        st.dataframe(bottom5, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: PRODUCT MIX & REVENUE
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[3]:
    st.header("Product Mix & Revenue")

    col1, col2 = st.columns(2)

    with col1:
        rev_share = (
            df.groupby("Item Purchased")["Revenue"]
            .sum()
            .reindex(PRODUCT_ORDER)
            .reset_index()
        )
        fig = px.pie(
            rev_share,
            values="Revenue",
            names="Item Purchased",
            color_discrete_sequence=PLOTLY_COLORS,
            hole=0.45,
        )
        fig.update_traces(
            textinfo="label+percent",
            textfont_size=12,
            marker=dict(line=dict(color="rgba(0,0,0,0.3)", width=1)),
        )
        dark_layout(fig, "Revenue Share by Product", "Each product's contribution to total revenue")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        unit_sales = (
            df["Item Purchased"]
            .value_counts()
            .reindex(PRODUCT_ORDER)
            .reset_index()
        )
        unit_sales.columns = ["Product", "Units"]
        unit_sales = unit_sales.sort_values("Units", ascending=True)
        fig = px.bar(
            unit_sales,
            y="Product",
            x="Units",
            color="Product",
            color_discrete_sequence=PLOTLY_COLORS,
            orientation="h",
            text=unit_sales["Units"].apply(lambda v: f"{v:,}"),
        )
        fig.update_traces(textposition="outside", textfont_size=12)
        dark_layout(fig, "Unit Sales by Product", "Sorted by volume")
        fig.update_layout(showlegend=False, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    flavour_product = (
        df.groupby(["Item Purchased", "Flavour"])
        .size()
        .reset_index(name="Count")
    )
    fig = px.bar(
        flavour_product,
        x="Item Purchased",
        y="Count",
        color="Flavour",
        barmode="group",
        color_discrete_sequence=PLOTLY_COLORS,
        category_orders={"Item Purchased": PRODUCT_ORDER, "Flavour": FLAVOUR_ORDER},
    )
    dark_layout(fig, "Flavour Popularity by Product", "Grouped bars — compare flavour mix within each product")
    fig.update_layout(xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    tier_data = (
        df.groupby("Item Purchased")
        .agg(units=("Customer ID", "count"), revenue=("Revenue", "sum"))
        .reindex(PRODUCT_ORDER)
        .reset_index()
    )
    tier_data["Avg Price"] = tier_data["revenue"] / tier_data["units"]
    tier_data["Volume %"] = (tier_data["units"] / tier_data["units"].sum()) * 100
    tier_data["Revenue %"] = (tier_data["revenue"] / tier_data["revenue"].sum()) * 100

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=tier_data["Item Purchased"], y=tier_data["Volume %"],
        name="Volume %", marker_color=COLORS["primary"],
        text=tier_data["Volume %"].apply(lambda v: f"{v:.1f}%"), textposition="outside",
    ))
    fig.add_trace(go.Bar(
        x=tier_data["Item Purchased"], y=tier_data["Revenue %"],
        name="Revenue %", marker_color=COLORS["accent"],
        text=tier_data["Revenue %"].apply(lambda v: f"{v:.1f}%"), textposition="outside",
    ))
    fig.update_layout(barmode="group")
    dark_layout(fig, "Volume Share vs Revenue Share", "When revenue % > volume %, the product has a higher avg price")
    fig.update_layout(yaxis_title="Percentage (%)", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    cross_tab = df.pivot_table(
        values="Customer ID", index="Item Purchased", columns="Flavour",
        aggfunc="count",
    ).reindex(index=PRODUCT_ORDER, columns=FLAVOUR_ORDER).fillna(0)

    fig = px.imshow(
        cross_tab,
        text_auto=True,
        color_continuous_scale="Blues",
        aspect="auto",
    )
    dark_layout(fig, "Product × Flavour Order Count", "Darker = more orders for that combination")
    fig.update_layout(xaxis_title="Flavour", yaxis_title="Product")
    fig.update_traces(textfont_size=13)
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: DEMOGRAPHIC PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[4]:
    st.header("Demographic Patterns")

    col1, col2 = st.columns(2)

    with col1:
        age_dist = (
            df["Age Group"]
            .value_counts()
            .reindex(AGE_LABELS)
            .reset_index()
        )
        age_dist.columns = ["Age Group", "Count"]
        age_dist["Pct"] = (age_dist["Count"] / age_dist["Count"].sum() * 100).round(1)
        fig = px.bar(
            age_dist,
            x="Age Group",
            y="Count",
            color="Age Group",
            color_discrete_sequence=PLOTLY_COLORS,
            text=age_dist.apply(lambda r: f"{r['Count']:,} ({r['Pct']}%)", axis=1),
        )
        fig.update_traces(textposition="outside", textfont_size=11)
        dark_layout(fig, "Orders by Age Group", "Count and percentage of total orders")
        fig.update_layout(showlegend=False, xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        gender_age = (
            df.groupby(["Age Group", "Gender"], observed=True)
            .size()
            .reset_index(name="Count")
        )
        fig = px.bar(
            gender_age,
            x="Age Group",
            y="Count",
            color="Gender",
            barmode="group",
            color_discrete_sequence=[COLORS["primary"], COLORS["accent"]],
            category_orders={"Age Group": AGE_LABELS},
        )
        dark_layout(fig, "Gender × Age Group", "Male vs Female order volume by age bracket")
        fig.update_layout(xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    prod_age = (
        df.groupby(["Age Group", "Item Purchased"], observed=True)
        .size()
        .reset_index(name="Count")
    )
    fig = px.bar(
        prod_age,
        x="Age Group",
        y="Count",
        color="Item Purchased",
        barmode="stack",
        color_discrete_sequence=PLOTLY_COLORS,
        category_orders={"Age Group": AGE_LABELS, "Item Purchased": PRODUCT_ORDER},
    )
    dark_layout(fig, "Product Preference by Age Group", "Stacked bars — each color is a product")
    fig.update_layout(xaxis_title="Age Group", legend_title="Product")
    st.plotly_chart(fig, use_container_width=True)

    flav_age = df.pivot_table(
        values="Customer ID", index="Age Group", columns="Flavour",
        aggfunc="count", observed=False,
    ).reindex(index=AGE_LABELS, columns=FLAVOUR_ORDER).fillna(0)

    fig = px.imshow(
        flav_age,
        text_auto=True,
        color_continuous_scale="YlOrRd",
        aspect="auto",
    )
    dark_layout(fig, "Flavour × Age Group Heatmap", "Warmer colors = more orders")
    fig.update_layout(xaxis_title="Flavour", yaxis_title="Age Group")
    fig.update_traces(textfont_size=13)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Gender Comparison")
    gender_stats = (
        df.groupby("Gender")
        .agg(
            orders=("Customer ID", "count"),
            avg_spend=("Revenue", "mean"),
            avg_rating=("Review Rating", "mean"),
            avg_prev_purchases=("Previous Purchases", "mean"),
        )
        .reset_index()
    )
    gender_stats.columns = ["Gender", "Orders", "Avg Spend (AUD)", "Avg Rating", "Avg Prev Purchases"]
    gender_stats["Avg Spend (AUD)"] = gender_stats["Avg Spend (AUD)"].round(2)
    gender_stats["Avg Rating"] = gender_stats["Avg Rating"].round(2)
    gender_stats["Avg Prev Purchases"] = gender_stats["Avg Prev Purchases"].round(1)
    st.dataframe(gender_stats, use_container_width=True, hide_index=True)

    col3, col4 = st.columns(2)

    with col3:
        gender_product = (
            df.groupby(["Gender", "Item Purchased"]).size().reset_index(name="Count")
        )
        fig = px.bar(
            gender_product,
            x="Item Purchased",
            y="Count",
            color="Gender",
            barmode="group",
            color_discrete_sequence=[COLORS["primary"], COLORS["accent"]],
            category_orders={"Item Purchased": PRODUCT_ORDER},
        )
        dark_layout(fig, "Product Preference by Gender")
        fig.update_layout(xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        gender_flavour = (
            df.groupby(["Gender", "Flavour"]).size().reset_index(name="Count")
        )
        fig = px.bar(
            gender_flavour,
            x="Flavour",
            y="Count",
            color="Gender",
            barmode="group",
            color_discrete_sequence=[COLORS["primary"], COLORS["accent"]],
            category_orders={"Flavour": FLAVOUR_ORDER},
        )
        dark_layout(fig, "Flavour Preference by Gender")
        fig.update_layout(xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6: PAYMENT & LOYALTY
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[5]:
    st.header("Payment & Loyalty")

    col1, col2 = st.columns(2)

    with col1:
        pay_dist = df["Payment Method"].value_counts().reset_index()
        pay_dist.columns = ["Payment Method", "Count"]
        fig = px.pie(
            pay_dist,
            values="Count",
            names="Payment Method",
            color_discrete_sequence=PLOTLY_COLORS,
            hole=0.45,
        )
        fig.update_traces(
            textinfo="label+percent",
            textfont_size=11,
            marker=dict(line=dict(color="rgba(0,0,0,0.3)", width=1)),
        )
        dark_layout(fig, "Payment Method Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        pay_age = (
            df.groupby(["Age Group", "Payment Method"], observed=True)
            .size()
            .reset_index(name="Count")
        )
        fig = px.bar(
            pay_age,
            x="Age Group",
            y="Count",
            color="Payment Method",
            barmode="stack",
            color_discrete_sequence=PLOTLY_COLORS,
            category_orders={"Age Group": AGE_LABELS},
        )
        dark_layout(fig, "Payment Method by Age Group", "Stacked — each color is a payment method")
        fig.update_layout(xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        pay_gender = (
            df.groupby(["Gender", "Payment Method"]).size().reset_index(name="Count")
        )
        fig = px.bar(
            pay_gender,
            x="Payment Method",
            y="Count",
            color="Gender",
            barmode="group",
            color_discrete_sequence=[COLORS["primary"], COLORS["accent"]],
        )
        dark_layout(fig, "Payment Method by Gender")
        fig.update_layout(xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        mean_prev = df["Previous Purchases"].mean()
        fig = px.histogram(
            df,
            x="Previous Purchases",
            nbins=20,
            color_discrete_sequence=[COLORS["success"]],
            histnorm="percent",
        )
        fig.update_traces(marker_line_width=0.5, marker_line_color="rgba(0,0,0,0.3)")
        add_mean_line(fig, df["Previous Purchases"], axis="x", name="Avg", color=COLORS["accent"])
        dark_layout(fig, "Purchase History Distribution", f"Average: {mean_prev:.0f} previous purchases")
        fig.update_layout(xaxis_title="Previous Purchases", yaxis_title="% of Orders")
        st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(
        df,
        x="Previous Purchases",
        y="Review Rating",
        color="Item Purchased",
        opacity=0.3,
        color_discrete_sequence=PLOTLY_COLORS,
        category_orders={"Item Purchased": PRODUCT_ORDER},
        trendline="lowess",
    )
    fig.update_traces(marker=dict(size=4, line=dict(width=0)), selector=dict(mode="markers"))
    dark_layout(fig, "Loyalty vs Satisfaction",
                "LOWESS trend lines show whether repeat buyers rate differently")
    fig.update_layout(legend_title="Product")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Loyalty Tier Profiles")
    loyalty_profiles = (
        df.groupby("Loyalty Tier", observed=True)
        .agg(
            customers=("Customer ID", "count"),
            avg_rating=("Review Rating", "mean"),
            avg_spend=("Revenue", "mean"),
            top_product=("Item Purchased", lambda x: x.mode().iloc[0] if len(x) > 0 else "N/A"),
            top_flavour=("Flavour", lambda x: x.mode().iloc[0] if len(x) > 0 else "N/A"),
        )
        .reindex(LOYALTY_LABELS)
        .reset_index()
    )
    loyalty_profiles.columns = [
        "Loyalty Tier", "Customers", "Avg Rating", "Avg Spend (AUD)", "Top Product", "Top Flavour"
    ]
    loyalty_profiles["Avg Rating"] = loyalty_profiles["Avg Rating"].round(2)
    loyalty_profiles["Avg Spend (AUD)"] = loyalty_profiles["Avg Spend (AUD)"].round(2)
    st.dataframe(loyalty_profiles, use_container_width=True, hide_index=True)

    loyalty_product = (
        df.groupby(["Loyalty Tier", "Item Purchased"], observed=True)
        .size()
        .reset_index(name="Count")
    )
    fig = px.bar(
        loyalty_product,
        x="Loyalty Tier",
        y="Count",
        color="Item Purchased",
        barmode="group",
        color_discrete_sequence=PLOTLY_COLORS,
        category_orders={"Loyalty Tier": LOYALTY_LABELS, "Item Purchased": PRODUCT_ORDER},
    )
    dark_layout(fig, "Loyalty Tier × Product Mix", "Do loyal customers prefer different products?")
    fig.update_layout(xaxis_title="Loyalty Tier", legend_title="Product")
    st.plotly_chart(fig, use_container_width=True)
