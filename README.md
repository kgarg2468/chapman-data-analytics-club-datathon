# Chapman Data Analytics Club Datathon

Interactive Streamlit dashboard analyzing 3,900+ online energy drink orders for the Chapman Data Analytics Club Datathon.

## Dashboard Sections

1. **Executive Summary** — KPIs, revenue breakdown, rating distribution, demographics
2. **Customer Segmentation** — KMeans clustering with interactive segment exploration
3. **Satisfaction Drivers** — Rating analysis by product, flavour, demographics; product-flavour heatmap
4. **Product Mix & Revenue** — Revenue share, unit sales, flavour cross-tabs, price tier analysis
5. **Demographic Patterns** — Age group and gender breakdowns, preference heatmaps
6. **Payment & Loyalty** — Payment method analysis, loyalty tiers, retention signals

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tech Stack

- **Dashboard:** Streamlit
- **Charts:** Plotly
- **Data:** pandas, NumPy
- **ML:** scikit-learn (KMeans clustering)

## Dataset

`Online Energy Drink Orders.csv` — 3,901 records with fields: Customer ID, Age, Gender, Item Purchased, Purchase Amount (AUD), Review Rating, Payment Method, Previous Purchases, Flavour.
