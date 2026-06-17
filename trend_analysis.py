import streamlit as st # type: ignore
import pandas as pd
import plotly.express as px # type: ignore
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go # type: ignore

# ---------------------------------------------------
# Load Data
# ---------------------------------------------------
df = pd.read_csv("Data/MAIN_DATA.csv")

# ---------------------------------------------------
# Clean Drought_Class Properly
# ---------------------------------------------------
df["Drought_Class"] = (
    df["Drought_Class"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# Standardize names
df["Drought_Class"] = df["Drought_Class"].replace({
    "severe drought": "Severe drought",
    "moderate drought": "Moderate drought",
    "moderately wet": "Moderately Wet",
    "severely wet": "Severely Wet",
    "normal": "Normal"
})

# ---------------------------------------------------
# Trend Analysis Function
# ---------------------------------------------------
def trend_analysis(state=None, district=None):

    st.subheader("Drought Trend Analysis")

    # ---------------------------------------------------
    # Required Columns Check
    # ---------------------------------------------------
    required_cols = [
        "year",
        "district_name",
        "State",
        "Drought_Class"
    ]

    missing_cols = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        st.error(f"Missing columns: {missing_cols}")
        return

    # ---------------------------------------------------
    # Copy Data
    # ---------------------------------------------------
    filtered_df = df.copy()

    # ---------------------------------------------------
    # Apply Filters
    # ---------------------------------------------------
    if state is not None:

        filtered_df = filtered_df[
            filtered_df["State"] == state
        ]

    if district is not None:

        filtered_df = filtered_df[
            filtered_df["district_name"] == district
        ]

    # ---------------------------------------------------
    # Remove Normal Category
    # ---------------------------------------------------
    filtered_df = filtered_df[
        filtered_df["Drought_Class"] != "Normal"
    ]


    # ---------------------------------------------------
    # Empty Check
    # ---------------------------------------------------
    if filtered_df.empty:
        st.warning("No data available.")
        return

    # ---------------------------------------------------
    # Drought Order
    # ---------------------------------------------------
    drought_order = [
        "Severe drought",
        "Moderate drought",
        "Moderately Wet",
        "Severely Wet"
    ]

    print(
        filtered_df[
            filtered_df["year"] == 2014
        ]["Drought_Class"].value_counts()
    )
    # ---------------------------------------------------
    # Group Data Properly
    # ---------------------------------------------------
    trend_df = (
        filtered_df
        .groupby(
            ["year", "Drought_Class"]
        )
        .size()
        .reset_index(name="Count")
    )


    # ---------------------------------------------------
    # Keep Only Desired Categories
    # ---------------------------------------------------
    trend_df = trend_df[
        trend_df["Drought_Class"].isin(drought_order)
    ]

    # ---------------------------------------------------
    # Convert Year Properly
    # ---------------------------------------------------
    trend_df["year"] = trend_df["year"].astype(int)

    # ---------------------------------------------------
    # Sort Years Numerically
    # ---------------------------------------------------
    trend_df = trend_df.sort_values(by="year")

    # ---------------------------------------------------
    # Color Map
    # ---------------------------------------------------
    color_map = {
        "Severe drought": "#8B0000",
        "Moderate drought": "#E53935",
        "Moderately Wet": "#90CAF9",
        "Severely Wet": "#42A5F5"
    }

    # ---------------------------------------------------
    # Title
    # ---------------------------------------------------
    if state is None:

        title = "India Drought Trend Analysis"

    elif district is None:

        title = f"{state} Drought Trend Analysis"

    else:

        title = (
            f"{district}, {state} "
            f"Drought Trend Analysis"
        )

    # ---------------------------------------------------
    # Plot
    # ---------------------------------------------------
    fig = px.bar(
        trend_df,
        x="year",
        y="Count",
        color="Drought_Class",
        category_orders={
            "Drought_Class": drought_order,
            "year": sorted(trend_df["year"].unique())
        },
        color_discrete_map=color_map,
        title=title,
        barmode="stack"
    )

    # ---------------------------------------------------
    # Layout
    # ---------------------------------------------------
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Count (Monthly)",
        legend_title="Drought Class",
        height=650,
        template="plotly_white",
        hovermode="x unified"
    )

    fig.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=sorted(trend_df["year"].unique())
    )

    # ---------------------------------------------------
    # Plot in Streamlit
    # ---------------------------------------------------
    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ===================================================
    # Climate Variable Trend Analysis
    # ===================================================

    st.subheader("Climate Variable Trends")

    # ---------------------------------------------------
    # Year Range Selector
    # ---------------------------------------------------

    available_years = sorted(
        filtered_df["year"].unique()
    )

    start_year, end_year = st.select_slider(
        "Select Year Range",
        options=available_years,
        value=(
            min(available_years),
            max(available_years)
        )
    )

    # Filter selected years
    filtered_df = filtered_df[
        (filtered_df["year"] >= start_year) &
        (filtered_df["year"] <= end_year)
    ]


    # Temperature
    filtered_df = filtered_df[
        (filtered_df["temp_mean"] >= 5) &
        (filtered_df["temp_mean"] <= 45)
    ]

    # Precipitation
    filtered_df = filtered_df[
        (filtered_df["precip_mean"] >= 0)
    ]

    # SPEI
    filtered_df = filtered_df[
        (filtered_df["SPEI_3"] >= -5) &
        (filtered_df["SPEI_3"] <= 5)
    ]
    
    # ---------------------------------------------------
    # Monthly aggregation
    # ---------------------------------------------------
    climate_df = (
        filtered_df
        .groupby(["year", "month"])[
            ["precip_mean", "temp_mean", "SPEI_3"]
        ]
        .mean()
        .reset_index()
    )

    # ---------------------------------------------------
    # Create datetime column
    # ---------------------------------------------------
    climate_df["Date"] = pd.to_datetime(
        climate_df["year"].astype(str)
        + "-"
        + climate_df["month"].astype(str)
        + "-01"
    )

    # ---------------------------------------------------
    # Normalize variables
    # ---------------------------------------------------
    scaler = StandardScaler()

    climate_df[
        ["precip_norm", "temp_norm", "spei_norm"]
    ] = scaler.fit_transform(
        climate_df[
            ["precip_mean", "temp_mean", "SPEI_3"]
        ]
    )

    # ---------------------------------------------------
    # Create figure
    # ---------------------------------------------------
    fig2 = go.Figure()

    # Precipitation
    fig2.add_trace(
        go.Scatter(
            x=climate_df["Date"],
            y=climate_df["precip_norm"],
            mode="lines",
            name="Precipitation",
            line=dict(
                color="#1E88E5",
                width=2
            )
        )
    )

    # Temperature
    fig2.add_trace(
        go.Scatter(
            x=climate_df["Date"],
            y=climate_df["temp_norm"],
            mode="lines",
            name="Temperature",
            line=dict(
                color="#E53935",
                width=2
            )
        )
    )

    # SPEI
    fig2.add_trace(
        go.Scatter(
            x=climate_df["Date"],
            y=climate_df["spei_norm"],
            mode="lines",
            name="SPEI-3",
            line=dict(
                color="#43A047",
                width=2
            )
        )
    )

    # ---------------------------------------------------
    # Layout
    # ---------------------------------------------------
    fig2.update_layout(
        title="",
        template="plotly_white",
        height=600,
        hovermode="x unified",

        xaxis=dict(
            title="Year",

            # show only years
            tickformat="%Y",

            # yearly spacing
            dtick="M12",

        ),

        yaxis=dict(
            title="Normalized Value"
        ),

        legend_title="Variables"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    for val in sorted(
        df["Drought_Class"]
        .dropna()
        .astype(str)
        .unique()
    ):
        print(repr(val))
