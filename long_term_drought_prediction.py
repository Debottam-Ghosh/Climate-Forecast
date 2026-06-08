# ============================================================
# Long-Term SPEI Drought Forecast
# ============================================================

import base64
import numpy as np
import pandas as pd
import streamlit as st  # type: ignore
import matplotlib.pyplot as plt
from html import escape

from rainfall_model2 import run_rainfall_kalman
from temperature_model import run_temperature_kalman, MONTH_NAMES
from drought_model4 import run_drought_model
import streamlit.components.v1 as components #type: ignore

from reportlab.platypus import ( #type: ignore
    SimpleDocTemplate,
    Paragraph,
    Spacer
)   

from reportlab.lib.styles import getSampleStyleSheet #type: ignore
from reportlab.lib.pagesizes import letter #type: ignore
from reportlab.lib import colors #type: ignore
from reportlab.platypus.tables import Table, TableStyle #type: ignore
from reportlab.platypus.flowables import HRFlowable #type: ignore
import streamlit.components.v1 as components # type: ignore
from io import BytesIO
from dotenv import load_dotenv # type: ignore
import os

load_dotenv()

# ============================================================
# CONSTANTS
# ============================================================

MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
    "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
    "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
}

C_SPEI = '#7B2D8B'
C_OBS  = '#D85A30'
C_MOD  = '#F0A500'
C_EXT  = '#8B0000'
C_NORM = '#1D9E75'


spi_scale = 3


def _render_download_link(label, data_bytes, file_name, mime, accent_color):
    href = f"data:{mime};base64,{base64.b64encode(data_bytes).decode('utf-8')}"
    st.markdown(
        f'''
        <a download="{escape(file_name)}" href="{href}" target="_self"
           style="display:block;width:100%;box-sizing:border-box;
                  text-align:center;padding:0.62rem 1rem;margin:0.25rem 0;
                  background:{accent_color};color:white;text-decoration:none;
                  border-radius:6px;font-weight:600;font-size:13px;">
            {escape(label)}
        </a>
        ''',
        unsafe_allow_html=True,
    )


# ============================================================
# LOAD DISTRICT DATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_climate_data():
    df = pd.read_csv("Data/climate_master.csv")

    df.columns = [c.strip().lower() for c in df.columns]
    df["state"] = (
        df["state"]
        .astype(str)
        .str.strip()
        .replace({"Telegana": "Telangana"})
        .str.lower()
    )
    df["district"] = (
        df["district"]
        .astype(str)
        .str.strip()
        .replace({"Jangoan": "Jangoan"})
        .str.lower()
    )

    return df


def get_district_data(df, state, district):

    state = str(state).strip().lower()
    district = str(district).strip().lower()

    sub = (
        df[
            (df["state"] == state) &
            (df["district"] == district)
        ]
        .sort_values(["year", "month"])
        .reset_index(drop=True)
    )

    if sub.empty:
        raise ValueError(
            f"No climate data found for state={state!r}, district={district!r}"
        )

    yrs  = sub["year"].values.astype(int)
    mos  = sub["month"].values.astype(int) - 1

    rain = np.maximum(
        sub["rain_mean"].values.astype(float),
        0.001
    )

    temp = sub["temp_mean"].values.astype(float)

    lat = (
        float(sub["latitude"].dropna().iloc[0])
        if "latitude" in sub.columns and
           sub["latitude"].notna().any()
        else 20.0
    )

    return yrs, mos, rain, temp, lat


# ============================================================
# MAIN FUNCTION
# ============================================================
def long_term_drought_prediction(state, district, y, m):

    st.subheader("Long-Term Drought Forecast")

    # --------------------------------------------------------
    # USER INPUT VALIDATION
    # --------------------------------------------------------

    if y < 2026 or y > 2036:
        st.error("Year must be between 2026 and 2036")
        return

    m = m.upper()

    if m not in MONTH_MAP:
        st.error("Month must be one of JAN ... DEC")
        return

    target_month = MONTH_MAP[m]

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = load_climate_data()

    yrs, mos, rain, temp, lat = get_district_data(
        df,
        state,
        district
    )

    # --------------------------------------------------------
    # TRAINING SPLIT
    # TRAINING: till 2025 OCT
    # --------------------------------------------------------

    train_mask = (
        (yrs < 2025) |
        (
            (yrs == 2025) &
            ((mos + 1) <= 10)
        )
    )

    # --------------------------------------------------------
    # FORECAST HORIZON
    # --------------------------------------------------------
   
    start_year  = 2025
    start_month = 11

    forecast_months = (
        (y - start_year) * 12 +
        (target_month - start_month) + 1
    )

    end_idx = forecast_months

    # --------------------------------------------------------
    # RUN RAINFALL MODEL
    # --------------------------------------------------------

    try:

        with st.spinner("Running rainfall model..."):

            R = run_rainfall_kalman(
            rain,
            mos,
            yrs,
            train_mask,
            n_paths=500,
            seed=42,
            representative='mean',
            forecast_horizon=end_idx
        )

        with st.spinner("Running temperature model..."):

            T = run_temperature_kalman(
            temp,
            mos,
            yrs,
            train_mask,
            n_paths=500,
            seed=42,
            q=0.05,
            forecast_horizon=end_idx
        )

        with st.spinner("Computing SPEI forecast..."):

            D = run_drought_model(
            R,
            T,
            scale=3,
            lat_deg=lat,
            representative='mean'
        )

    except Exception as e:

        st.exception(e)
        return

    # --------------------------------------------------------
    # EXTRACT FORECAST
    # --------------------------------------------------------

    spei_fc = D["spei_fc"]

    mean_ = spei_fc["mean"][:end_idx]
    med_  = spei_fc["med"][:end_idx]

    p05 = spei_fc["p05"][:end_idx]
    p10 = spei_fc["p10"][:end_idx]
    p90 = spei_fc["p90"][:end_idx]
    p95 = spei_fc["p95"][:end_idx]

    years_plot = []
    months_plot = []

    cy = 2025
    cm = 11

    for _ in range(end_idx):

        years_plot.append(cy)
        months_plot.append(cm - 1)

        cm += 1

        if cm > 12:
            cm = 1
            cy += 1

    labels = [
        f"{years_plot[i]}-{MONTH_NAMES[months_plot[i]]}"
        for i in range(len(years_plot))
    ]

    x = np.arange(len(labels))

    # ========================================================
    # PLOT
    # ========================================================

    fig, ax = plt.subplots(figsize=(18, 7))

    # --------------------------------------------------------
    # DROUGHT CATEGORY BANDS
    # --------------------------------------------------------

    ax.axhspan(
        -3.5, -2.0,
        color=C_EXT,
        alpha=0.10
    )

    ax.axhspan(
        -2.0, -1.0,
        color=C_MOD,
        alpha=0.10
    )

    ax.axhspan(
        -1.0, 3.5,
        color=C_NORM,
        alpha=0.05
    )

    # --------------------------------------------------------
    # THRESHOLD LINES
    # --------------------------------------------------------

    ax.axhline(
        -1.0,
        color=C_MOD,
        linestyle='--',
        linewidth=1.2,
        label='Moderate drought threshold'
    )

    ax.axhline(
        -2.0,
        color=C_EXT,
        linestyle='--',
        linewidth=1.2,
        label='Extreme drought threshold'
    )

    ax.axhline(
        0,
        color='black',
        linestyle=':',
        linewidth=1
    )

    # --------------------------------------------------------
    # UNCERTAINTY BANDS
    # --------------------------------------------------------

    ax.fill_between(
        x,
        p05,
        p95,
        color=C_SPEI,
        alpha=0.15,
        label='5th–95th percentile'
    )

    ax.fill_between(
        x,
        p10,
        p90,
        color=C_SPEI,
        alpha=0.25,
        label='10th–90th percentile'
    )

    # --------------------------------------------------------
    # MEAN / MEDIAN
    # --------------------------------------------------------

    ax.plot(
        x,
        mean_,
        color=C_SPEI,
        linewidth=2.5,
        label='Mean SPEI'
    )

    ax.plot(
        x,
        med_,
        color=C_OBS,
        linewidth=2,
        linestyle='--',
        label='Median SPEI'
    )

    # --------------------------------------------------------
    # LABELS
    # --------------------------------------------------------

    ax.text(
        len(x)-1,
        -0.9,
        "No drought",
        fontsize=9,
        ha='right'
    )

    ax.text(
        len(x)-1,
        -1.9,
        "Moderate drought",
        fontsize=9,
        ha='right'
    )

    ax.text(
        len(x)-1,
        -2.9,
        "Extreme drought",
        fontsize=9,
        ha='right'
    )

    # --------------------------------------------------------
    # AXES
    # --------------------------------------------------------

    ax.set_xticks(x[::3])

    ax.set_xticklabels(
        [labels[i] for i in range(0, len(labels), 3)],
        rotation=40,
        ha='right'
    )

    ax.set_ylim(-3.2, 3.2)

    ax.set_ylabel("SPEI-3")

    ax.set_title(
        f"{district}, {state} — Long-Term SPEI Forecast",
        fontsize=14,
        fontweight='bold'
    )

    ax.grid(True, alpha=0.15)

    ax.legend(
        fontsize=9,
        ncol=2,
        loc='upper left'
    )

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


    # =========================================================
    # FORECASTED DROUGHT CATEGORY TIMELINE (SPEI ONLY)
    # =========================================================

    st.markdown("### Forecasted Monthly Drought Categories (SPEI)")

    st.caption(
        "Category thresholds (SPEI): "
        "No drought >= -1.0 | "
        "Moderate drought >= -2.0 and < -1.0 | "
        "Extreme drought < -2.0"
    )

    # ---------------------------------------------------------
    # Build forecast periods
    # ---------------------------------------------------------

    forecast_periods = []

    for yy, mm in zip(years_plot, months_plot):

        forecast_periods.append(
            f"{yy} {MONTH_NAMES[mm]}"
        )

    # ---------------------------------------------------------
    # Build year -> indices mapping
    # ---------------------------------------------------------

    from collections import OrderedDict

    year_map = OrderedDict()

    for i, p in enumerate(forecast_periods):

        yy = int(p.split()[0])

        year_map.setdefault(yy, []).append(i)

    forecast_years = list(year_map.keys())

    # ---------------------------------------------------------
    # Default tab = 2026
    # ---------------------------------------------------------

    default_idx = 0

    if 2026 in forecast_years:
        default_idx = forecast_years.index(2026)

    tabs = st.tabs([str(y) for y in forecast_years])

    # ---------------------------------------------------------
    # HTML generator
    # ---------------------------------------------------------

    def _forecast_html(periods, no_p, mod_p, ext_p):

        fc_cells = ''

        fc_n_none = 0
        fc_n_mod  = 0
        fc_n_ext  = 0

        for p, nd, md, ex in zip(periods, no_p, mod_p, ext_p):

            cats = [
                ('None', nd, '#E1F5EE', '#0F6E56'),
                ('Mod',  md, '#F0A500', '#4A2800'),
                ('Ext',  ex, '#8B0000', 'white')
            ]

            label, _, bg, fc = max(cats, key=lambda x: x[1])

            if label == 'None':
                fc_n_none += 1
            elif label == 'Mod':
                fc_n_mod += 1
            else:
                fc_n_ext += 1

            mon = p.split(' ')[1][:3]
            yr  = str(p.split(' ')[0])[2:]

            fc_cells += (
                f'<div style="flex:1;min-width:0;text-align:center;'
                f'background:{bg};border-radius:4px;'
                f'padding:6px 2px;margin:1px;">'

                f'<div style="font-size:10px;'
                f'color:{fc};font-weight:700;">{mon}</div>'

                f'<div style="font-size:8px;'
                f'color:{fc};opacity:.9;">\'{yr}</div>'

                f'</div>'
            )

        return f"""
        <div style="
            background:white;
            font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
            border:1.5px solid #7B2D8B;
            border-radius:12px;
            padding:14px;
            margin-bottom:10px;
        ">

            <div style="
                font-size:14px;
                font-weight:700;
                color:#7B2D8B;
                margin-bottom:12px;
            ">
                SPEI-{spi_scale} Forecast — Dominant Category
            </div>

            <div style="
                display:flex;
                gap:2px;
                flex-wrap:nowrap;
                margin-bottom:10px;
            ">
                {fc_cells}
            </div>

            <div style="
                display:flex;
                gap:12px;
                flex-wrap:wrap;
                margin-bottom:12px;
                font-size:11px;
            ">
                <span style="color:#0F6E56;">■ No drought</span>
                <span style="color:#B45309;">■ Moderate</span>
                <span style="color:#8B0000;">■ Extreme</span>
            </div>

            <div style="
                display:flex;
                gap:8px;
                flex-wrap:wrap;
            ">

                <span style="
                    background:#D1FAE5;
                    color:#065F46;
                    padding:3px 10px;
                    border-radius:20px;
                    font-size:11px;
                    font-weight:600;
                ">
                    No drought: {fc_n_none}
                </span>

                <span style="
                    background:#FEF3C7;
                    color:#B45309;
                    padding:3px 10px;
                    border-radius:20px;
                    font-size:11px;
                    font-weight:600;
                ">
                    Moderate: {fc_n_mod}
                </span>

                <span style="
                    background:#FEE2E2;
                    color:#7F1D1D;
                    padding:3px 10px;
                    border-radius:20px;
                    font-size:11px;
                    font-weight:600;
                ">
                    Extreme: {fc_n_ext}
                </span>

            </div>

        </div>
        """

    # ---------------------------------------------------------
    # Render year-wise tabs
    # ---------------------------------------------------------

    for tab, yy in zip(tabs, forecast_years):

        with tab:

            idxs = year_map[yy]

            periods_y = [forecast_periods[i] for i in idxs]

            no_p_y = D['spei_prob']['no_drought'][idxs]
            mod_p_y = D['spei_prob']['moderate'][idxs]
            ext_p_y = D['spei_prob']['extreme'][idxs]

            html_y = _forecast_html(
                periods_y,
                no_p_y,
                mod_p_y,
                ext_p_y
            )

            components.html(
                html_y,
                height=230,
                scrolling=False
            )

    # ========================================================
    # SUMMARY TABLE
    # ========================================================

    out_df = pd.DataFrame({
        "Year": years_plot,
        "Month": [MONTH_NAMES[m] for m in months_plot],
        "Mean": mean_,
        "Median": med_,
        "P05": p05,
        "P10": p10,
        "P90": p90,
        "P95": p95
    })

    st.markdown("### Forecast Summary")

    csv = out_df.round(3).to_csv(index=False).encode("utf-8")

    _render_download_link(
        label="Download Forecast",
        data_bytes=csv,
        file_name="summary.csv",
        mime="text/csv",
        accent_color="#1D4E89",
    )

    return out_df

    from collections import defaultdict
    import json

# ==========================================================================================
# AI-GENERATED ACTIONS
# ==========================================================================================


def AI_generated_actions(state, district, y, m, out_df):

    import json
    from pathlib import Path
    from collections import defaultdict
    from groq import Groq # type: ignore

    client = Groq(
        api_key=os.getenv("GROQ_API_KEY")
    )

    # =========================================================
    # VALIDATION
    # =========================================================

    if out_df is None:
        st.warning("Run forecast first.")
        return


    if y < 2026 or y > 2036:
        st.error("Year must be between 2026 and 2036")
        return

    m = m.upper()

    if m not in MONTH_MAP:
        st.error("Invalid month")
        return

    target_month = MONTH_MAP[m]

    forecast_compact = []

    for _, row in out_df.round(2).iterrows():

        mean_val = float(row["Mean"])

        if mean_val < -2:
            category = "Extreme Drought"

        elif mean_val < -1:
            category = "Moderate Drought"

        else:
            category = "No Drought"

        forecast_compact.append({

            "Year": int(row["Year"]),
            "Month": row["Month"],
            "SPEI": mean_val,
            "Category": category
        })

    forecast_compact_str = json.dumps(
        forecast_compact,
        separators=(",", ":")
    )


    # ============================================================
    # CREATE STRUCTURED JSON
    # ============================================================

    forecast_json_structured = {
        "State": state,
        "District": district,
        "Time Series Data": []
    }

    # ------------------------------------------------------------
    # GROUP MONTHS YEAR-WISE
    # ------------------------------------------------------------

    year_map = defaultdict(list)

    for _, row in out_df.round(3).iterrows():

        mean_val = float(row["Mean"])

        # --------------------------------------------------------
        # DROUGHT CATEGORY
        # --------------------------------------------------------

        if mean_val < -2:
            drought_category = "Extreme Drought"

        elif mean_val < -1:
            drought_category = "Moderate Drought"

        else:
            drought_category = "No Drought"

        # --------------------------------------------------------
        # APPEND MONTH DATA
        # --------------------------------------------------------

        year_map[int(row["Year"])].append({

            "Month": row["Month"],

            "Drought Category": drought_category,

            "Forecasted Data": {

                "Mean SPEI-3": mean_val,

                "Median SPEI-3": float(row["Median"]),

                "10th Quantile": float(row["P10"]),

                "90th Quantile": float(row["P90"])
            }
        })

    # ------------------------------------------------------------
    # BUILD FINAL JSON STRUCTURE
    # ------------------------------------------------------------

    for year, month_data in year_map.items():

        forecast_json_structured["Time Series Data"].append({

            "Year": year,

            "Month-wise Data": month_data
        })

    # ============================================================
    # CONVERT TO JSON STRING
    # ============================================================

    forecast_json_structured_str = json.dumps(
        forecast_json_structured,
        indent=2
    )


    # =========================================================
    # CROP DATA
    # =========================================================

    crop_text = "No crop data available."

    crop_file = Path("Data/telangana_crops_water_requirements.json")

    if crop_file.exists():

        try:

            with open(crop_file, "r", encoding="utf-8") as f:
                crop_json = json.load(f)

            district_entry = None

            for item in crop_json:

                if (
                    item.get("District", "").strip().lower()
                    == district.strip().lower()
                ):
                    district_entry = item
                    break

            if district_entry and "Crops" in district_entry:

                crops = district_entry["Crops"]

                if len(crops) > 0:

                    crop_lines = []

                    for crop in crops:

                        crop_name = crop.get(
                            "Crop Name",
                            "Unknown"
                        )

                        water_req = crop.get(
                            "Water Requirement - (MM / Hectare)",
                            "NA"
                        )

                        crop_lines.append(
                            f"{crop_name} "
                            f"(Water Requirement: {water_req} mm/hectare)"
                        )

                    crop_text = "\n".join(crop_lines)

        except:
            pass

    # =========================================================
    # GROQ HELPER
    # =========================================================

    def ask_groq(prompt, max_tokens=250):

        response = client.chat.completions.create(

            model="llama-3.1-8b-instant",

            messages=[
                {
                    "role": "system",
                    "content":
                    "You are a professional agricultural drought intelligence assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.4,
            max_tokens=max_tokens
        )

        return (
            response
            .choices[0]
            .message
            .content
        )


    # =========================================================
    # STYLING
    # =========================================================

    st.markdown("""
    <style>

    .ai-main-title{
        font-size:42px;
        font-weight:800;
        color:#6B21A8;
        margin-bottom:28px;
    }

    .ai-card{

        background:linear-gradient(
            135deg,
            #ffffff 0%,
            #faf5ff 100%
        );

        border-radius:28px;

        padding:30px;

        margin-bottom:24px;

        border:1px solid rgba(139,92,246,0.12);

        box-shadow:
            0 10px 30px rgba(0,0,0,0.05);

    }

    .ai-section-title{

        font-size:28px;

        font-weight:800;

        color:#7B2D8B;

        margin-bottom:20px;
    }

    .ai-divider{

        height:1px;

        background:linear-gradient(
            90deg,
            rgba(139,92,246,0.25),
            rgba(139,92,246,0.02)
        );

        margin-bottom:20px;
    }

    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="ai-main-title">
            AI Climate Intelligence
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
    <style>
    iframe {
        margin-bottom: -20px !important;
        display: block;
    }
    </style>
    """, unsafe_allow_html=True)

    

    
    
    

    # =========================================================
    # SUMMARY
    # =========================================================

    summary_placeholder = st.empty()

    with st.spinner("Generating Summary..."):

        summary_prompt = f"""
    You are an agricultural climate analyst.

    Below is long-term SPEI drought forecast data.

    {forecast_compact_str}

    TASK:
    1. This the data of SPEI-3 of the particular State and District.
    2. Read the ENTIRE forecast horizon carefully.
    3. Understand the overall drought trend across all years and months.
    4. Write a 4-5 line summary.
    5. DO NOT use technical jargon like quantile, median, probability etc.
    6. Explain in very simple language understandable by farmers.
    7. Mention whether conditions improve, worsen or fluctuate.

    IMPORTANT CLIMATE RULE:
    Positive SPEI values indicate wetter and favorable conditions.
    Negative SPEI values indicate drought conditions.

    IMPORTANT:
    1. Do NOT use markdown.
    2. Do NOT use bullet symbols.
    3. Do NOT use headings.
    4. Use plain readable paragraphs.
    5. Separate ideas using line breaks only.
    """

    # SUMMARY
    with st.spinner("Generating Summary..."):
        if "summary_text" not in st.session_state:
            st.session_state.summary_text = ask_groq(summary_prompt, max_tokens=900)
        
        summary_text = st.session_state.summary_text

    st.markdown("### 🌤️ Summary")
    st.markdown("---")
    for line in st.session_state.summary_text.split('\n'):
        if line.strip():
            st.markdown(f"🔹 {line.strip()}")
    st.markdown("---")
    st.markdown("---")



    # =========================================================
    # ACTIONS
    # =========================================================

    actions_placeholder = st.empty()

    with st.spinner("Generating Recommendations..."):

        actions_prompt = f"""
    You are an agricultural drought management expert.

    Forecast Data:

    {forecast_compact_str}

    TASK:
    1. Read the complete forecast timeline.
    2. Suggest NOT MORE THAN 6 practical recommended actions like irrigation, groundwater, crop management, water conservation, reservoir planning, farmer preparedness.
    4. Use bullet points.
    5. Keep suggestions realistic and actionable.

    IMPORTANT CLIMATE RULE:
    Positive SPEI values indicate wetter and favorable conditions.
    Negative SPEI values indicate drought conditions.

    IMPORTANT:
    1. Do NOT use markdown.
    2. Do NOT use bullet symbols.
    3. Do NOT use headings.
    4. Use plain readable paragraphs.
    5. Separate ideas using line breaks only.
    """

    # ACTIONS
    with st.spinner("Generating Recommendations..."):
        if "actions_text" not in st.session_state:
            st.session_state.actions_text = ask_groq(actions_prompt, max_tokens=900)
        
        actions_text = st.session_state.actions_text

    st.markdown("### 🚜 Recommended Actions")
    st.markdown("---")
    for line in st.session_state.actions_text.split('\n'):
        if line.strip():
            st.markdown(f"🔹 {line.strip()}")
    st.markdown("---")
    st.markdown("---")

    # =========================================================
    # CROPS
    # =========================================================

    crop_placeholder = st.empty()

    with st.spinner("Generating Crop Recommendations..."):

        crop_prompt = f"""
    You are an agricultural crop advisory expert.

    Forecast Data:

    {forecast_compact_str}

    Available Crops:

    {crop_text}

    TASK:
    1. Analyze the complete drought forecast.
    2. Recommend suitable 4-5 crops. Not more than that. Don't add extra lines.
    3. Lower SPEI means higher drought risk.
    4. If drought risk is high, prefer crops with lower water requirement.
    5. Explain WHY the crops are suitable.
    6. Mention water-efficient crops prominently if needed.
    7. Write in simple farmer-friendly language.

    IMPORTANT CLIMATE RULE:
    Positive SPEI values indicate wetter and favorable conditions.
    Negative SPEI values indicate drought conditions.

    IMPORTANT:
    1. Do NOT use markdown.
    2. Do NOT use bullet symbols.
    3. Do NOT use headings.
    4. Use plain readable paragraphs.
    5. Separate ideas using line breaks only.
    """

    # CROPS
    with st.spinner("Generating Crop Recommendations..."):
        if "crop_text_output" not in st.session_state:
            st.session_state.crop_text_output = ask_groq(crop_prompt, max_tokens=900)

        crop_text_output = st.session_state.crop_text_output


    st.markdown("### 🌾 Crop Recommendations")
    st.markdown("---")
    for line in st.session_state.crop_text_output.split('\n'):
        if line.strip():
            st.markdown(f"🔹 {line.strip()}")
    st.markdown("---")
    st.markdown("---")

    # =========================================================
    # STRATEGY
    # =========================================================

    strategy_placeholder = st.empty()

    with st.spinner("Generating Strategic Insights..."):

        strategy_prompt = f"""
    You are a long-term climate adaptation strategist.

    Forecast Data:

    {forecast_compact_str}

    TASK:
    1. Analyze the ENTIRE forecast horizon.
    2. Give 4-5 strategic long-term planning insights. Not more than that. Don't add extra lines.
    3. Focus on:
    - climate adaptation
    - irrigation infrastructure
    - groundwater sustainability
    - crop diversification
    - resilience planning
    4. Keep concise but insightful.


    IMPORTANT CLIMATE RULE:
    Positive SPEI values indicate wetter and favorable conditions.
    Negative SPEI values indicate drought conditions.

    IMPORTANT:
    1. Do NOT use markdown.
    2. Do NOT use bullet symbols.
    3. Do NOT use headings.
    4. Use plain readable paragraphs.
    5. Separate ideas using line breaks only.
    """

    # STRATEGY
    with st.spinner("Generating Strategic Insights..."):
        if "strategy_text" not in st.session_state:
            st.session_state.strategy_text = ask_groq(strategy_prompt, max_tokens=900)
        
        strategy_text = st.session_state.strategy_text

    st.markdown("### 📈 Strategic Planning Insights")
    st.markdown("---")
    for line in st.session_state.strategy_text.split('\n'):
        if line.strip():
            st.markdown(f"🔹 {line.strip()}")
    st.markdown("---")
    st.markdown("---")
    
    # =========================================================
    # PDF GENERATION
    # =========================================================

    def clean_text(text):

        replacements = {
            "🌤️": "",
            "🚜": "",
            "🌾": "",
            "📈": "",
            "⚡": "",
            "•": "-",
            "—": "-"
        }

        for k, v in replacements.items():
            text = text.replace(k, v)

        return text


    pdf_buffer = BytesIO()

    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=30
    )

    styles = getSampleStyleSheet()

    title_style = styles["Title"]
    heading_style = styles["Heading2"]
    body_style = styles["BodyText"]

    elements = []

    # =========================================================
    # TITLE
    # =========================================================

    title = (
        f"Drought Forecasting for "
        f"{district}, {state} "
        f"till {m}, {y}"
    )

    elements.append(
        Paragraph(title, title_style)
    )

    elements.append(Spacer(1, 18))

    elements.append(
        HRFlowable(
            width="100%",
            thickness=1.2,
            color=colors.HexColor("#7B2D8B")
        )
    )

    elements.append(Spacer(1, 22))

    # =========================================================
    # SECTION ADDER
    # =========================================================

    def add_pdf_section(title, content):

        elements.append(
            Paragraph(
                f"<font color='#7B2D8B'><b>{title}</b></font>",
                heading_style
            )
        )

        elements.append(Spacer(1, 8))

        cleaned = clean_text(content)

        paragraphs = cleaned.split("\n")

        for para in paragraphs:

            para = para.strip()

            if para:

                elements.append(
                    Paragraph(
                        para,
                        body_style
                    )
                )

                elements.append(
                    Spacer(1, 6)
                )

        elements.append(
            Spacer(1, 18)
        )

    # =========================================================
    # ADD SECTIONS
    # =========================================================

    add_pdf_section(
        "Summary",
        summary_text
    )

    add_pdf_section(
        "Recommended Actions",
        actions_text
    )

    add_pdf_section(
        "Crop Recommendations",
        crop_text_output
    )

    add_pdf_section(
        "Strategic Planning Insights",
        strategy_text
    )

    # =========================================================
    # BUILD PDF
    # =========================================================

    doc.build(elements)

    pdf_buffer.seek(0)

    # =========================================================
    # DOWNLOAD BUTTON
    # =========================================================

    _render_download_link(
        label="📄 Download Full AI Report",
        data_bytes=pdf_buffer.getvalue(),
        file_name=(
            f"drought_forecast_"
            f"{district}_{state}_{m}_{y}.pdf"
        ),
        mime="application/pdf",
        accent_color="#7B2D8B",
    )

def long_term_forecast(state, district, y, m):

    st.markdown("## 🌵 Model Forecast")

    tab1, tab2 = st.tabs([
        "Forecast",
        "Insights"
    ])

    with tab1:
        out_df = long_term_drought_prediction(
            state,
            district,
            y,
            m
        )

    with tab2:
        AI_generated_actions(state, district, y, m, out_df)

    