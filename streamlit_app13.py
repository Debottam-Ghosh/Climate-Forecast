"""
streamlit_app.py
================
Climate Forecast Dashboard — Rainfall, Temperature, SPI/SPEI Drought

Run:
    streamlit run streamlit_app13.py

Required files in the same folder:
    rainfall_model2.py
    temperature_model.py
    drought_model4.py
    enso_helper.py

Usage:
    1. App preloads Data/climate_master.csv automatically.
       Required columns:
           state, district, year, month, rain_mean, temp_mean
       Optional: latitude, longitude
    2. Select State → District from the auto-populated dropdowns.
    3. Adjust parameters; ONI data is preloaded from ONI_text_to_csv_columns.csv.
    4. Click ▶ Run Forecast.
"""

import io
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import plotly.graph_objects as go # type: ignore
import streamlit as st # type: ignore

warnings.filterwarnings('ignore')

import streamlit.components.v1 as components # type: ignore
from rainfall_model2   import run_rainfall_kalman, MONTHS
from temperature_model import run_temperature_kalman, MONTH_NAMES
from drought_model4    import run_drought_model, plot_drought
from trend_analysis import trend_analysis


def _reset_show_trend_graph():
    st.session_state.show_trend_graph = False


def _push_panel(panel_name):
    history = st.session_state.get("panel_history", [])
    history = [panel_name] + [item for item in history if item != panel_name]
    st.session_state.panel_history = history
    st.session_state.active_panel = panel_name


def _request_trend_run():
    st.session_state.show_trend_graph = True
    _push_panel("trend")


def _request_model_run():
    st.session_state.run_model_requested = True
    _push_panel("model")


def _request_forecast_run(state, district, forecast_end_year, forecast_end_month):
    st.session_state.forecast_active = True
    st.session_state.forecast_state = state
    st.session_state.forecast_district = district
    st.session_state.forecast_end_year = forecast_end_year
    st.session_state.forecast_end_month = forecast_end_month
    _push_panel("forecast")


def _clean_text(value):
    if value is None:
        return None
    text = str(value).strip()
    return text or None

# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "Climate Hazard Forecasting · DiCRA",
    page_icon  = "🌩",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

def _startup_healthcheck():
    """Fail fast on missing local files (useful for Streamlit Cloud deploys)."""
    required_paths = [
        "Data/climate_master.csv",
        "rainfall_model2.py",
        "temperature_model.py",
        "drought_model4.py",
        "enso_helper.py",
    ]
    missing = [p for p in required_paths if not Path(p).exists()]
    if missing:
        st.error("Deployment check failed: missing required files.")
        st.code("\n".join(missing), language="text")
        st.stop()

_startup_healthcheck()


@st.cache_data(show_spinner=False)
def load_default_oni_bytes():
    oni_path = Path("ONI_text_to_csv_columns.csv")
    if not oni_path.exists():
        return None
    return oni_path.read_bytes()


@st.cache_data(show_spinner=False)
def load_master_trend_data():
    """Load the drought trend source used by Historic Trend controls."""
    master_path = Path("Data/MAIN_DATA.csv")
    if not master_path.exists():
        return None

    try:
        df = pd.read_csv(master_path)
        # Keep only Telangana and West Bengal
        allowed_states = ["Telangana", "West Bengal"]

        df = df[df["State"].isin(allowed_states)]

    except Exception:
        return None

    required = ["State", "district_name", "year", "Drought_Class"]
    if any(col not in df.columns for col in required):
        return None

    return df

# ─────────────────────────────────────────────────────────────────────────────
# THEME  — Option 1: dark header, clean cards
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Hide default Streamlit chrome ───────────────────────── */
#MainMenu, footer { visibility: hidden; }
header { background: transparent !important; }
[data-testid="stToolbar"] { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
.stDeployButton { display: none; }

/* ── Hide sidebar collapse/toggle button entirely ────────── */
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
button[kind="header"],
header button {
    display: none !important;
    visibility: hidden !important;
}

/* ── Dark top header bar ─────────────────────────────────── */
.cf-header {
    background: #0F2D52;
    padding: 0.7rem 2rem;
    display: flex;
    align-items: center;
    gap: 1.2rem;
    margin: -1rem -1rem 1.5rem -1rem;
    border-bottom: 3px solid #1D9E75;
}
.cf-header-logo {
    font-size: 20px;
    color: white;
    font-weight: 500;
    letter-spacing: -0.3px;
    white-space: nowrap;
}
.cf-header-sep { flex: 1; }

/* ── Sidebar — original light style ─────────────────────── */
[data-testid="stSidebar"] {
    background: #F7F8FA;
    border-right: 1px solid #E2E6EA;
}
[data-testid="stSidebar"] .stMarkdown h3 {
    font-size: 13px !important;
    font-weight: 700 !important;
    color: #0F2D52 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 1.2rem 0 0.5rem !important;
    padding-bottom: 5px;
    border-bottom: 2px solid #1D9E75;
}
[data-testid="stSidebar"] .stMarkdown h4 {
    font-size: 10px !important;
    font-weight: 700 !important;
    color: #6B7280 !important;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin: 0.9rem 0 0.3rem !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stFileUploader label {
    font-size: 12px !important;
    color: #374151 !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: #1D4E89 !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 0.5rem 1rem !important;
    width: 100% !important;
    transition: background 0.15s;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #185FA5 !important;
}
[data-testid="stSidebar"] .stButton > button:disabled {
    background: #9CA3AF !important;
    color: #E5E7EB !important;
}
[data-testid="stSidebar"] .streamlit-expanderHeader {
    font-size: 11px !important;
    font-weight: 500 !important;
    color: #6B7280 !important;
    background: #F0F4F8 !important;
    border-radius: 6px !important;
}
[data-testid="stSidebar"] hr {
    border-color: #E2E6EA !important;
    margin: 0.6rem 0 !important;
}

/* ── Metric cards — tinted backgrounds, no progress bars ─── */
.cf-metric-row {
    display: grid;
    grid-template-columns: repeat(6, minmax(0, 1fr));
    gap: 10px;
    margin: 0.8rem 0 1.4rem;
}
.cf-metric {
    background: rgba(29, 78, 137, 0.07);
    border-radius: 10px;
    padding: 0.75rem 0.9rem;
    border: 1px solid rgba(29, 78, 137, 0.15);
    position: relative;
    overflow: hidden;
}
.cf-metric::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: #1D4E89;
    border-radius: 10px 10px 0 0;
}
.cf-metric.green {
    background: rgba(29, 158, 117, 0.07);
    border-color: rgba(29, 158, 117, 0.18);
}
.cf-metric.green::before  { background: #1D9E75; }
.cf-metric.purple {
    background: rgba(123, 45, 139, 0.07);
    border-color: rgba(123, 45, 139, 0.15);
}
.cf-metric.purple::before { background: #7B2D8B; }
.cf-metric.amber {
    background: rgba(216, 90, 48, 0.07);
    border-color: rgba(216, 90, 48, 0.15);
}
.cf-metric.amber::before  { background: #D85A30; }
.cf-metric-label {
    font-size: 10px;
    font-weight: 600;
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
}
.cf-metric-value {
    font-size: 22px;
    font-weight: 500;
    color: #111827;
    line-height: 1.1;
}
.cf-metric-unit {
    font-size: 10px;
    color: #9CA3AF;
    margin-left: 2px;
    font-weight: 400;
}

/* ── Section headers ─────────────────────────────────────── */
.cf-section {
    font-size: 11px;
    font-weight: 600;
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding-bottom: 6px;
    border-bottom: 1px solid #E5E7EB;
    margin: 1.2rem 0 0.8rem;
}

/* ── Tabs ────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
    padding: 0;
    border-bottom: 2px solid #E5E7EB;
    margin-bottom: 1.2rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0 !important;
    padding: 10px 22px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #6B7280 !important;
    background: #F3F4F6 !important;
    border: 1.5px solid #E5E7EB !important;
    border-bottom: none !important;
    margin-bottom: -2px !important;
    transition: all 0.15s ease;
    letter-spacing: 0.01em;
}
.stTabs [data-baseweb="tab"]:hover {
    background: #E8ECF0 !important;
    color: #374151 !important;
}
.stTabs [aria-selected="true"] {
    background: white !important;
    color: #1D4E89 !important;
    border-color: #CBD5E1 !important;
    border-bottom-color: white !important;
    box-shadow: 0 -2px 0 #1D4E89 inset !important;
}

/* ── Welcome card ────────────────────────────────────────── */
.cf-welcome {
    background: linear-gradient(135deg, #0F2D52 0%, #1D4E89 100%);
    border-radius: 12px;
    padding: 2.5rem 2rem;
    color: white;
    text-align: center;
    margin: 2rem 0;
}
.cf-welcome h2 {
    font-size: 22px;
    font-weight: 500;
    margin: 0 0 0.5rem;
    color: white;
}
.cf-welcome p {
    font-size: 14px;
    color: rgba(255,255,255,0.7);
    margin: 0;
}

/* ── Info strip ──────────────────────────────────────────── */
.cf-strip {
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    font-size: 12px;
    color: #1E40AF;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Status / spinner ────────────────────────────────────── */
[data-testid="stStatusWidget"] {
    border-radius: 8px !important;
    border: 1px solid #E5E7EB !important;
}

/* ── Expander ────────────────────────────────────────────── */
.streamlit-expanderHeader {
    font-size: 12px !important;
    font-weight: 500 !important;
    color: #6B7280 !important;
    background: #F9FAFB !important;
    border-radius: 6px !important;
}

/* ── Divider ─────────────────────────────────────────────── */
hr { border-color: #F3F4F6 !important; margin: 1rem 0 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# COLOURS  (shared across all plot functions)
# ─────────────────────────────────────────────────────────────────────────────

C_KF  = '#1D4E89'
C_TMP = '#7B2D8B'
C_OBS = '#D85A30'
C_LTA = '#2E8B57'
C_TR  = '#888780'


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING  (handles both TL schema and WB schema)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_files(file_bytes_list, file_names):
    """
    Read one or more uploaded Excel files and merge into a single DataFrame.
    Normalises column names and fills missing lat/lon with NaN.
    Returns DataFrame with columns:
        state, district, year, month, rain_mean, temp_mean, latitude, longitude
    """
    frames = []
    for raw_bytes, name in zip(file_bytes_list, file_names):
        try:
            df = pd.read_excel(io.BytesIO(raw_bytes))
            df.columns = [c.strip().lower() for c in df.columns]

            # Required columns — flexible name matching
            rename = {}
            for col in df.columns:
                if col in ('yr', 'year'):         rename[col] = 'year'
                if col in ('mo', 'month'):        rename[col] = 'month'
                if col in ('rain', 'rainfall'):   rename[col] = 'rain_mean'
                if col in ('temp', 'temperature'):rename[col] = 'temp_mean'
                if col in ('lat',):               rename[col] = 'latitude'
                if col in ('lon', 'long'):        rename[col] = 'longitude'
            df.rename(columns=rename, inplace=True)

            required = ['state', 'district', 'year', 'month',
                        'rain_mean', 'temp_mean']
            missing  = [c for c in required if c not in df.columns]
            if missing:
                st.warning(f"⚠ {name}: skipping — missing columns: {missing}")
                continue

            # Add lat/lon columns if absent
            for col in ('latitude', 'longitude'):
                if col not in df.columns:
                    df[col] = np.nan

            frames.append(df[required + ['latitude', 'longitude']])
        except Exception as e:
            st.warning(f"⚠ {name}: could not read ({e})")

    if not frames:
        return None
    merged = pd.concat(frames, ignore_index=True)
    # Drop rows missing key values
    merged.dropna(subset=['year', 'month', 'rain_mean', 'temp_mean'],
                  inplace=True)
    merged['year']  = merged['year'].astype(int)
    merged['month'] = merged['month'].astype(int)
    return merged


def get_district_data(df, district):
    """
    Extract arrays for one district from the merged DataFrame.
    Returns: yrs, mos (0-indexed), rain, temp, lat, lon
    """
    sub = (df[df['district'] == district]
           .sort_values(['year', 'month'])
           .reset_index(drop=True))
    yrs  = sub['year'].values
    mos  = sub['month'].values - 1          # 0-indexed
    rain = np.maximum(sub['rain_mean'].values.astype(float), 0.001)
    temp = sub['temp_mean'].values.astype(float)

    lats = sub['latitude'].values.astype(float)
    lons = sub['longitude'].values.astype(float)
    lat  = float(lats[~np.isnan(lats)][0]) if (~np.isnan(lats)).any() else np.nan
    lon  = float(lons[~np.isnan(lons)][0]) if (~np.isnan(lons)).any() else np.nan
    return yrs, mos, rain, temp, lat, lon


# ─────────────────────────────────────────────────────────────────────────────
# MODEL RUNNER  (cached by key parameters)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_models(yrs, mos, rain, temp, lat,
               train_end, train_end_month_num, n_paths, seed, dlm_q,
               q_mu0, q_gamma, enso_bytes, spi_scale,
               representative='mean'):
    """
    Run all three models and return result dicts.
    Cached: re-runs only when inputs change.
    """
    train_mask = (
        (yrs < train_end) |
        ((yrs == train_end) & ((mos + 1) <= train_end_month_num))
    )

    # ENSO (optional)
    enso_arr = None
    if enso_bytes is not None:
        try:
            from enso_helper import load_and_align as _load_enso
            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix='.csv',
                                             delete=False) as tmp:
                tmp.write(enso_bytes)
                tmp_path = tmp.name
            enso_arr, _ = _load_enso(tmp_path, yrs, mos, standardise=True)
            os.unlink(tmp_path)
        except Exception as e:
            st.warning(f"ONI load failed: {e} — running without ENSO.")

    # Rainfall
    R = run_rainfall_kalman(
            rain, mos, yrs, train_mask,
            n_paths=n_paths, seed=seed,
            q_mu0=q_mu0, q_gamma=q_gamma,
            enso=enso_arr,
            representative=representative)

    # Temperature
    T = run_temperature_kalman(
            temp, mos, yrs, train_mask,
            n_paths=n_paths, seed=seed, q=dlm_q)

    # Drought
    lat_use = lat if not np.isnan(lat) else 20.0
    DR = run_drought_model(
        R, T, scale=spi_scale, lat_deg=lat_use,
        representative=representative)

    return R, T, DR


# ─────────────────────────────────────────────────────────────────────────────
# AXIS HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _axis_labels(years_all, months_all, T_tr, T_te, ctx):
    yr_ctx  = years_all[T_tr - ctx: T_tr]
    mo_ctx  = months_all[T_tr - ctx: T_tr]
    yr_te   = years_all[T_tr:]
    mo_te   = months_all[T_tr:]
    ctx_lbl = [f"{yr_ctx[i]} {MONTH_NAMES[mo_ctx[i]]}" for i in range(ctx)]
    fc_lbl  = [f"{yr_te[i]} {MONTH_NAMES[mo_te[i]]}"   for i in range(T_te)]
    x_ctx   = np.arange(-ctx, 0)
    x_fc    = np.arange(T_te)
    xt      = list(range(-ctx, 0, 3)) + list(range(0, T_te, 3))
    xl      = ([ctx_lbl[i] for i in range(0, ctx, 3)] +
               [fc_lbl[i]  for i in range(0, T_te, 3)])
    return x_ctx, x_fc, xt, xl


def _metrics_box(ax, acc, unit):
    txt = (f"RMSE  = {acc['rmse']:.3f} {unit}\n"
           f"MAE   = {acc['mae']:.3f} {unit}\n"
           f"BIAS  = {acc['bias']:+.3f} {unit}\n"
           f"SKILL = {acc['skill']:.3f}\n"
           f"90%CI = {acc['cov90']:.0f}%")
    ax.text(0.013, 0.975, txt,
            transform=ax.transAxes, fontsize=7.5,
            va='top', ha='left', family='monospace',
            bbox=dict(boxstyle='round,pad=0.4',
                      fc='white', ec='#bbbbbb', alpha=0.88))


# ─────────────────────────────────────────────────────────────────────────────
# PLOT FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def make_rain_paths(R, district, n_paths, ctx_months=18):
    """Plot 1.1 — all rainfall ensemble paths."""
    ctx    = min(ctx_months, R['T_tr'])
    T_te   = R['T_te']
    x_ctx, x_fc, xt, xl = _axis_labels(
        R['years_all'], R['months_all'], R['T_tr'], T_te, ctx)
    lta_fc  = R['lta_mo'][R['mo_te']]
    lta_ctx = R['lta_mo'][R['months_all'][R['T_tr'] - ctx: R['T_tr']]]

    # ── Sensible y-axis ceiling ───────────────────────────────────────────────
    # Use 99th percentile of ensemble + 20% headroom, but at least 1.5× the
    # observed max. This prevents a handful of extreme paths from collapsing
    # the visible range to a thin strip near zero.
    ens_p99   = float(np.percentile(R['X_kalman'], 99))
    obs_max   = float(np.nanmax(R['rain_te'])) if R['T_te'] > 0 else 0
    tr_max    = float(np.nanmax(R['rain_tr'][-ctx:]))
    y_ceil    = max(ens_p99 * 1.20, obs_max * 1.5, tr_max * 1.2, 1.0)

    fig, ax = plt.subplots(figsize=(13, 4.5))
    fig.suptitle(f"{district}  —  Figure 1.1: All simulated paths for rainfall",
                 fontsize=10, fontweight='bold')

    rep_series = R['fc'].get('rep', R['fc']['mean'])
    rep_label  = R['fc'].get('rep_label', 'Mean')

    n_show  = min(400, n_paths)
    idx_s   = np.linspace(0, n_paths - 1, n_show, dtype=int)
    for i in idx_s:
        # Clip each path to y_ceil so extreme paths don't distort the plot
        path = np.clip(R['X_kalman'][i], 0, y_ceil)
        ax.plot(x_fc, path, lw=0.25, color=C_KF, alpha=0.07)

        ax.plot(x_ctx, R['rain_tr'][-ctx:], lw=1.2, color=C_TR,
            ls='--', alpha=0.7, label='Training data')
        ax.axvline(-0.5, lw=1.0, color='black', alpha=0.2, ls=':')
        ax.plot(x_fc, rep_series, lw=2.2, color=C_KF,
            zorder=5, label=rep_label)
    ax.scatter(x_fc, R['rain_te'], s=38, color=C_OBS, zorder=6,
               label='Observed', edgecolors='white', linewidths=0.4)
    ax.plot(x_ctx, lta_ctx, lw=1.3, color=C_LTA, ls='-.', alpha=0.85,
            label='Historical average')
    ax.plot(x_fc,  lta_fc,  lw=1.3, color=C_LTA, ls='-.', alpha=0.85)

    ax.set_xticks(xt); ax.set_xticklabels(xl, rotation=38, ha='right', fontsize=7)
    ax.set_xlim(-ctx, T_te - 0.2)
    ax.set_ylim(0, y_ceil)
    ax.set_ylabel('Average Rainfall (mm/day)', fontsize=9)
    handles, labels = ax.get_legend_handles_labels()
    deduped = dict(zip(labels, handles))
    ax.legend(
        deduped.values(),
        deduped.keys(),
        fontsize=7.5,
        ncol=1,
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    ax.grid(True, alpha=0.18)
    plt.tight_layout()
    return fig, ctx, x_ctx, x_fc, xt, xl, lta_ctx, lta_fc


def make_rain_forecast(R, district, ctx, x_ctx, x_fc, xt, xl,
                       lta_ctx, lta_fc):
    """
    Plot 1.2 — Rainfall forecasted vs observed (interactive Plotly).
    Hover shows: period, observed, Kalman mean, 90% CI, LTA.
    """
    T_te     = R['T_te']
    acc      = R['acc']

    # ── Build unified string x-axis ──────────────────────────────────────────
    # ctx labels come from xl up to the split; fc labels are the rest
    n_xt     = len(xt)
    ctx_xl   = [xl[i] for i in range(len(xt)) if xt[i] < 0]
    fc_xl    = [xl[i] for i in range(len(xt)) if xt[i] >= 0]

    # Full label lists for every point
    ctx_labels = [f"{R['years_all'][R['T_tr']-ctx+i]} "
                  f"{MONTH_NAMES[R['months_all'][R['T_tr']-ctx+i]]}"
                  for i in range(ctx)]
    fc_labels  = [f"{R['years_all'][R['T_tr']+i]} "
                  f"{MONTH_NAMES[R['mo_te'][i]]}"
                  for i in range(T_te)]

    x_all    = ctx_labels + fc_labels                # full x label list
    x_ctx_s  = ctx_labels                            # context portion
    x_fc_s   = fc_labels                             # forecast portion
    rep_series = R['fc'].get('rep', R['fc']['mean'])
    rep_label  = R['fc'].get('rep_label', 'Mean')

    # ── colour helpers ────────────────────────────────────────────────────────
    def _rgba(hex_col, alpha):
        h  = hex_col.lstrip('#')
        r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
        return f'rgba({r},{g},{b},{alpha})'

    fig = go.Figure()

    # ── 90% CI band (p05–p95) ────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x_fc_s, y=R['fc']['p95'].tolist(),
        mode='lines', line=dict(width=0),
        showlegend=False, hoverinfo='skip', name='_p95'))

    fig.add_trace(go.Scatter(
        x=x_fc_s, y=R['fc']['p05'].tolist(),
        mode='lines', line=dict(width=0),
        fill='tonexty',
        fillcolor=_rgba(C_KF, 0.12),
            name='5th–95th quantile (90% CI)',
        hoverinfo='skip'))

    # ── 80% CI band (p10–p90) ────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x_fc_s, y=R['fc']['p90'].tolist(),
        mode='lines', line=dict(width=0),
        showlegend=False, hoverinfo='skip', name='_p90'))

    fig.add_trace(go.Scatter(
        x=x_fc_s, y=R['fc']['p10'].tolist(),
        mode='lines', line=dict(width=0),
        fill='tonexty',
        fillcolor=_rgba(C_KF, 0.20),
            name='10th–90th quantile (80% CI)',
        hoverinfo='skip'))

    # ── Quantile boundary lines ───────────────────────────────────────────────
    for vals, pct, dash in [
                (R['fc']['p95'], '95th quantile', 'dash'),
                (R['fc']['p90'], '90th quantile', 'dot'),
                (R['fc']['p10'], '10th quantile', 'dot'),
                (R['fc']['p05'], '5th quantile',  'dash')]:
        fig.add_trace(go.Scatter(
            x=x_fc_s, y=vals.tolist(),
            mode='lines',
            line=dict(color=C_KF, width=0.9, dash=dash),
            opacity=0.6,
            name=pct,
            hovertemplate=f'<b>{pct}</b>: %{{y:.3f}} mm/day<extra></extra>'))

    # ── Training context ──────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x_ctx_s, y=R['rain_tr'][-ctx:].tolist(),
        mode='lines',
        line=dict(color=C_TR, width=1.3, dash='dash'),
        opacity=0.65,
        name='Training data',
        hovertemplate='<b>%{x}</b><br>Training: %{y:.3f} mm/day<extra></extra>'))

    # ── LTA ──────────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x_ctx_s + x_fc_s,
        y=lta_ctx.tolist() + lta_fc.tolist(),
        mode='lines',
        line=dict(color=C_LTA, width=1.6, dash='dashdot'),
        opacity=0.9,
        name='Historical average',
        hovertemplate='<b>%{x}</b><br>LTA: %{y:.3f} mm/day<extra></extra>'))

    # ── Kalman mean ───────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x_fc_s, y=rep_series.tolist(),
        mode='lines',
        line=dict(color=C_KF, width=2.6),
            name=rep_label,
            hovertemplate=(f'<b>%{{x}}</b><br>{rep_label}: %{{y:.3f}} mm/day'
                           '<extra></extra>')))

    # ── Observed points — split by inside/outside 90% CI ─────────────────────
    obs_in_x,  obs_in_y  = [], []
    obs_out_x, obs_out_y = [], []
    obs_text_in, obs_text_out = [], []

    for i in range(T_te):
        o, lo, hi = R['rain_te'][i], R['fc']['p05'][i], R['fc']['p95'][i]
        lbl = (f"<b>{x_fc_s[i]}</b><br>"
               f"Observed: {o:.3f} mm/day<br>"
               f"{rep_label}: {rep_series[i]:.3f} mm/day<br>"
               f"90% CI: [{lo:.3f}, {hi:.3f}]<br>"
               f"LTA: {lta_fc[i]:.3f} mm/day")
        if lo <= o <= hi:
            obs_in_x.append(x_fc_s[i]); obs_in_y.append(o)
            obs_text_in.append(lbl)
        else:
            obs_out_x.append(x_fc_s[i]); obs_out_y.append(o)
            obs_text_out.append(lbl)

    if obs_in_x:
        fig.add_trace(go.Scatter(
            x=obs_in_x, y=obs_in_y,
            mode='markers',
            marker=dict(color=C_OBS, size=8,
                        line=dict(color='white', width=1)),
            name='Observed (inside CI)',
            hovertemplate='%{text}<extra></extra>',
            text=obs_text_in))

    if obs_out_x:
        fig.add_trace(go.Scatter(
            x=obs_out_x, y=obs_out_y,
            mode='markers',
            marker=dict(color=C_OBS, size=9,
                        line=dict(color='black', width=1.8)),
            name='Observed (outside 90% CI)',
            hovertemplate='%{text}<extra></extra>',
            text=obs_text_out))

    # ── Train/test vertical boundary ──────────────────────────────────────────
    fig.add_vline(
        x=ctx_labels[-1],
        line=dict(color='black', width=1.0, dash='dot'),
        opacity=0.25)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=f"<b>{district}</b>  —  Figure 1.2: Forecasted vs observed rainfall",
            font=dict(size=13),
            y=0.97, x=0.0, xanchor='left', yanchor='top'),
        yaxis=dict(title='Average Rainfall (mm/day)', rangemode='tozero',
                   gridcolor='rgba(0,0,0,0.08)'),
        xaxis=dict(
            tickangle=-40,
            tickfont=dict(size=9),
            gridcolor='rgba(0,0,0,0.06)',
            categoryorder='array',
            categoryarray=x_ctx_s + x_fc_s),
        legend=dict(
            orientation='h', yanchor='bottom', y=1.08,
            xanchor='left', x=0, font=dict(size=10)),
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=500,
        margin=dict(l=60, r=30, t=120, b=100))

    return fig


def make_temp_paths(T_res, district, n_paths, ctx_months=18):
    """Plot 2.1 — all temperature ensemble paths."""
    ctx   = min(ctx_months, T_res['T_tr'])
    T_te  = T_res['T_te']
    x_ctx, x_fc, xt, xl = _axis_labels(
        T_res['years_all'], T_res['months_all'],
        T_res['T_tr'], T_te, ctx)
    lta_fc  = T_res['lta_mo'][T_res['mo_te']]
    lta_ctx = T_res['lta_mo'][T_res['months_all'][T_res['T_tr'] - ctx: T_res['T_tr']]]

    fig, ax = plt.subplots(figsize=(13, 4.5))
    fig.suptitle(f"{district}  —  Figure 2.1: All simulted paths for temperature",
                 fontsize=10, fontweight='bold')

    n_show = min(400, n_paths)
    idx_s  = np.linspace(0, n_paths - 1, n_show, dtype=int)
    for i in idx_s:
        ax.plot(x_fc, T_res['paths'][i], lw=0.25, color=C_TMP, alpha=0.07)

    ax.plot(x_ctx, T_res['temp_tr'][-ctx:], lw=1.2, color=C_TR,
            ls='--', alpha=0.7, label='Training data')
    ax.axvline(-0.5, lw=1.0, color='black', alpha=0.2, ls=':')
    ax.plot(x_fc, T_res['fc']['mean'], lw=2.2, color=C_TMP,
            zorder=5, label='Mean')
    ax.scatter(x_fc, T_res['temp_te'], s=38, color=C_OBS, zorder=6,
               label='Observed', edgecolors='white', linewidths=0.4)
    ax.plot(x_ctx, lta_ctx, lw=1.3, color=C_LTA, ls='-.', alpha=0.85,
            label='Historical average')
    ax.plot(x_fc,  lta_fc,  lw=1.3, color=C_LTA, ls='-.', alpha=0.85)

    ax.set_xticks(xt); ax.set_xticklabels(xl, rotation=38, ha='right', fontsize=7)
    ax.set_xlim(-ctx, T_te - 0.2)
    ax.set_ylabel('Average temperature (°C/day)', fontsize=9)
    ax.legend(fontsize=7.5, ncol=4, loc='upper left')
    ax.grid(True, alpha=0.18)
    plt.tight_layout()
    return fig, ctx, x_ctx, x_fc, xt, xl, lta_ctx, lta_fc


def make_temp_forecast(T_res, district, ctx, x_ctx, x_fc, xt, xl,
                       lta_ctx, lta_fc):
    """
    Plot 2.2 — Temperature forecasted vs observed (interactive Plotly).
    Hover shows: period, observed, Kalman mean, 90% CI, LTA.
    """
    T_te  = T_res['T_te']
    acc   = T_res['acc']

    # ── Build string x-axis ───────────────────────────────────────────────────
    ctx_labels = [f"{T_res['years_all'][T_res['T_tr']-ctx+i]} "
                  f"{MONTH_NAMES[T_res['months_all'][T_res['T_tr']-ctx+i]]}"
                  for i in range(ctx)]
    fc_labels  = [f"{T_res['years_all'][T_res['T_tr']+i]} "
                  f"{MONTH_NAMES[T_res['mo_te'][i]]}"
                  for i in range(T_te)]

    x_ctx_s = ctx_labels
    x_fc_s  = fc_labels

    def _rgba(hex_col, alpha):
        h = hex_col.lstrip('#')
        r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
        return f'rgba({r},{g},{b},{alpha})'

    fig = go.Figure()

    # ── 90% CI band ───────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x_fc_s, y=T_res['fc']['p95'].tolist(),
        mode='lines', line=dict(width=0),
        showlegend=False, hoverinfo='skip', name='_p95'))

    fig.add_trace(go.Scatter(
        x=x_fc_s, y=T_res['fc']['p05'].tolist(),
        mode='lines', line=dict(width=0),
        fill='tonexty',
        fillcolor=_rgba(C_TMP, 0.12),
        name='5th–95th quantile (90% CI)',
        hoverinfo='skip'))

    # ── 80% CI band ───────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x_fc_s, y=T_res['fc']['p90'].tolist(),
        mode='lines', line=dict(width=0),
        showlegend=False, hoverinfo='skip', name='_p90'))

    fig.add_trace(go.Scatter(
        x=x_fc_s, y=T_res['fc']['p10'].tolist(),
        mode='lines', line=dict(width=0),
        fill='tonexty',
        fillcolor=_rgba(C_TMP, 0.20),
        name='10th–90th quantile (80% CI)',
        hoverinfo='skip'))

    # ── Quantile boundary lines ───────────────────────────────────────────────
    for vals, pct, dash in [
            (T_res['fc']['p95'], '95th quantile', 'dash'),
            (T_res['fc']['p90'], '90th quantile', 'dot'),
            (T_res['fc']['p10'], '10th quantile', 'dot'),
            (T_res['fc']['p05'], '5th quantile',  'dash')]:
        fig.add_trace(go.Scatter(
            x=x_fc_s, y=vals.tolist(),
            mode='lines',
            line=dict(color=C_TMP, width=0.9, dash=dash),
            opacity=0.6,
            name=pct,
            hovertemplate=f'<b>{pct}</b>: %{{y:.2f}} °C<extra></extra>'))

    # ── Training data ──────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x_ctx_s, y=T_res['temp_tr'][-ctx:].tolist(),
        mode='lines',
        line=dict(color=C_TR, width=1.3, dash='dash'),
        opacity=0.65,
        name='Training data',
        hovertemplate='<b>%{x}</b><br>Training: %{y:.2f} °C<extra></extra>'))

    # ── Historical average ──────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x_ctx_s + x_fc_s,
        y=lta_ctx.tolist() + lta_fc.tolist(),
        mode='lines',
        line=dict(color=C_LTA, width=1.6, dash='dashdot'),
        opacity=0.9,
        name='Historical average',
        hovertemplate='<b>%{x}</b><br>Historical average: %{y:.2f} °C<extra></extra>'))

    # ── Kalman mean ───────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x_fc_s, y=T_res['fc']['mean'].tolist(),
        mode='lines',
        line=dict(color=C_TMP, width=2.6),
        name='Mean',
        hovertemplate='<b>%{x}</b><br>Mean: %{y:.2f} °C<extra></extra>'))

    # ── Observed points — colour by inside/outside CI ─────────────────────────
    obs_in_x,  obs_in_y  = [], []
    obs_out_x, obs_out_y = [], []
    obs_text_in, obs_text_out = [], []

    for i in range(T_te):
        o  = T_res['temp_te'][i]
        lo = T_res['fc']['p05'][i]
        hi = T_res['fc']['p95'][i]
        lbl = (f"<b>{x_fc_s[i]}</b><br>"
               f"Observed: {o:.2f} °C<br>"
             f"Mean: {T_res['fc']['mean'][i]:.2f} °C<br>"
               f"90% CI: [{lo:.2f}, {hi:.2f}] °C<br>"
               f"LTA: {lta_fc[i]:.2f} °C")
        if lo <= o <= hi:
            obs_in_x.append(x_fc_s[i]); obs_in_y.append(o)
            obs_text_in.append(lbl)
        else:
            obs_out_x.append(x_fc_s[i]); obs_out_y.append(o)
            obs_text_out.append(lbl)

    if obs_in_x:
        fig.add_trace(go.Scatter(
            x=obs_in_x, y=obs_in_y,
            mode='markers',
            marker=dict(color=C_OBS, size=8, symbol='triangle-up',
                        line=dict(color='white', width=1)),
            name='Observed (inside CI)',
            hovertemplate='%{text}<extra></extra>',
            text=obs_text_in))

    if obs_out_x:
        fig.add_trace(go.Scatter(
            x=obs_out_x, y=obs_out_y,
            mode='markers',
            marker=dict(color=C_OBS, size=9, symbol='triangle-up',
                        line=dict(color='black', width=1.8)),
            name='Observed (outside 90% CI)',
            hovertemplate='%{text}<extra></extra>',
            text=obs_text_out))

    # ── Train/test boundary ───────────────────────────────────────────────────
    fig.add_vline(
        x=ctx_labels[-1],
        line=dict(color='black', width=1.0, dash='dot'),
        opacity=0.25)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=f"<b>{district}</b>  —  Figure 2.2: Forecasted vs observed temperature",
            font=dict(size=13),
            y=0.97, x=0.0, xanchor='left', yanchor='top'),
        yaxis=dict(title='Average temperature (°C/day)',
                   gridcolor='rgba(0,0,0,0.08)'),
        xaxis=dict(
            tickangle=-40,
            tickfont=dict(size=9),
            gridcolor='rgba(0,0,0,0.06)',
            categoryorder='array',
            categoryarray=x_ctx_s + x_fc_s),
        legend=dict(
            orientation='h', yanchor='bottom', y=1.08,
            xanchor='left', x=0, font=dict(size=10)),
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=500,
        margin=dict(l=60, r=30, t=120, b=100))

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

oni_bytes = load_default_oni_bytes()

with st.sidebar:

    df_all = pd.read_csv("Data/climate_master.csv")

    df_all.columns = [
        c.strip().lower()
        for c in df_all.columns
    ]


    df_all['state'] = (
        df_all['state']
        .astype(str)
        .str.strip()
        .replace({"Telegana": "Telangana"})
        .str.lower()
    )

    df_all['district'] = (
        df_all['district']
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # =====================================================================
    # COMMON LOCATION SELECTOR
    # =====================================================================

    st.markdown("### 📍 Location")

    common_df = df_all.copy()

    selected_state = None
    selected_district = None

    if common_df is not None:

        # ---------------------------------------------------
        # STATE
        # ---------------------------------------------------

        states_avail = sorted(
            common_df["state"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if not states_avail:
            st.warning("No historical trend states found in Data/MAIN_DATA.csv.")
            selected_state = None
            selected_district = None
            st.session_state.selected_state = None
            st.session_state.selected_district = None
            st.session_state.selected_state_trend = None
            st.session_state.selected_district_trend = None
        else:
            selected_state_display = st.selectbox(
                "Select State",
                options=states_avail,
                key="global_state"
            )

            selected_state_trend = _clean_text(selected_state_display)
            selected_state = selected_state_trend.lower() if selected_state_trend else None

            # ---------------------------------------------------
            # DISTRICT
            # ---------------------------------------------------

            districts_avail = sorted(
                common_df[
                    common_df["state"] == selected_state
                ]["district"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            if not districts_avail:
                st.warning(
                    f"No districts found for {selected_state_trend}."
                )
                selected_district = None
                selected_district_trend = None
            else:
                selected_district_display = st.selectbox(
                    "Select District",
                    options=districts_avail,
                    key="global_district"
                )

                selected_district_trend = _clean_text(selected_district_display)
                selected_district = (
                    selected_district_trend.lower()
                    if selected_district_trend else None
                )

            # ---------------------------------------------------
            # SAVE GLOBALLY
            # ---------------------------------------------------

            st.session_state.selected_state = (
                selected_state
                .strip()
                .lower()
            )

            st.session_state.selected_district = (
                selected_district
                .strip()
                .lower()
            )           
            
            st.session_state.selected_state_trend = selected_state_trend
            st.session_state.selected_district_trend = selected_district_trend

    st.markdown("---")

    # =====================================================================
    # HISTORIC TREND
    # =====================================================================

    st.markdown("---")

    if "show_trend" not in st.session_state:
        st.session_state.show_trend = False

    if "show_trend_graph" not in st.session_state:
        st.session_state.show_trend_graph = False

    # ---------------------------------------------------
    # Expander
    # ---------------------------------------------------

    with st.expander("Historical Trend"):

        st.session_state.show_trend = True

        trend_state = (
            st.session_state
            .get("selected_state", "")
            .title()
        )

        trend_district = (
            st.session_state
            .get("selected_district", "")
            .title()
        )

        st.button(
            "Show Trend",
            key="show_trend_btn",
            use_container_width=True,
            on_click=_request_trend_run
        )

        # =========================================
        # Divider
        # =========================================

        st.divider()

        
    # =====================================================================
    # MODEL EVALUATION
    # =====================================================================

    # ---------------------------------------------------
    # Expander
    # ---------------------------------------------------

    with st.expander("Model Validation"):

        st.session_state.model_evaluation = True
        sel_state = st.session_state.get("selected_state")
        sel_district = st.session_state.get("selected_district")
        

        if sel_state is None or sel_district is None:

            st.warning("Please select State and District above.")

            run_btn = False

        st.markdown("### ⚙️ Parameters")

        st.markdown("#### Training period ends before:")

        train_end = st.number_input(
            "Year",
            2021,
            2025,
            2024
        )

        month_names = [
            "January", "February", "March", "April",
            "May", "June", "July", "August",
            "September", "October", "November", "December"
        ]

        train_end_month = st.selectbox(
            "Month",
            month_names,
            index=11
        )

        train_end_month_num = month_names.index(train_end_month) + 1

        n_paths = st.select_slider(
            "Ensemble paths",
            options=[500, 1000, 2000, 3000, 5000, 7000, 8000, 10000], value=3000)
        
        spi_scale = 3
        rain_rep = "Mean"

        seed = 42
        dlm_q = 0.05
        q_mu0 = 1e-4
        q_gamma = 1e-7

        run_model_btn = st.button(
            "▶ Run Model",
            disabled=(sel_state is None or sel_district is None),
            use_container_width=True,
            key="run_model_button"
            ,on_click=_request_model_run
        )


    # =====================================================================
    # FUTURE FORECAST
    # =====================================================================

    # ---------------------------------------------------
    # Expander
    # ---------------------------------------------------

    with st.expander("Future Forecast"):
        st.session_state.Future_Forecast = True
        run_future_forecast = False
        forecast_end_year = 2026
        forecast_end_month = "December"

        lt_state = st.session_state.get("selected_state")
        lt_district = st.session_state.get("selected_district")

        
        forecast_end_year = st.number_input(
            "Forecast end year",
            2026,
            2030,
            2026
        )

        month_names = [
            "January", "February", "March", "April",
            "May", "June", "July", "August",
            "September", "October", "November", "December"
            ]

        forecast_end_month = st.selectbox(
            "Forecast end month",
            month_names,
            index=11
            )

        forecast_end_month_num = month_names.index(forecast_end_month) + 1

        st.button(
            "▶ Forecast",
            use_container_width=True,
            key="future_forecast_button",
            on_click=_request_forecast_run,
            args=(lt_state, lt_district, forecast_end_year, forecast_end_month)
        )

        run_future_forecast = st.session_state.get("forecast_active", False)

    # =========================================
    # Divider
    # =========================================

    st.divider()
        
# ─────────────────────────────────────────────────────────────────────────────
# MAIN PANEL
# ─────────────────────────────────────────────────────────────────────────────

# ── Dark header bar ───────────────────────────────────────────────────────────
st.markdown(f"""
<div class="cf-header">
  <div class="cf-header-logo">⚡ Climate Hazard Forecasting</div>
  <div class="cf-header-sep"></div>
</div>
""", unsafe_allow_html=True)

if "panel_history" not in st.session_state:
        initial_panel = st.session_state.get("active_panel")
        st.session_state.panel_history = [initial_panel] if initial_panel else []

panel_order = [
        panel for panel in st.session_state.get("panel_history", [])
        if panel in {"forecast", "trend", "model"}
]
panel_containers = {panel: st.container() for panel in panel_order}




# =====================================================================
# MAIN PANEL — FUTURE FORECAST
# =====================================================================

forecast_panel = panel_containers.get("forecast")
if run_future_forecast and forecast_panel is not None:
    forecast_panel.__enter__()
    from long_term_drought_prediction import long_term_forecast

    long_term_forecast(
        state=st.session_state.get("forecast_state", lt_state),
        district=st.session_state.get("forecast_district", lt_district),
        y=st.session_state.get("forecast_end_year", forecast_end_year),
        m=st.session_state.get("forecast_end_month", forecast_end_month)[:3].upper()
    )
    forecast_panel.__exit__(None, None, None)

# ─────────────────────────────────────────────────────────────────────────────
# HISTORIC TREND PANEL
# ─────────────────────────────────────────────────────────────────────────────
trend_panel = panel_containers.get("trend")
if st.session_state.show_trend and trend_panel is not None:
    trend_panel.__enter__()

    if st.session_state.get("show_trend_graph", False):

        st.markdown("## 📊 Historic Drought Trend")

        # ---------------------------------------------------
        # Whole India
        # ---------------------------------------------------

        if trend_state == "All States":

            trend_analysis(
                state=None,
                district=None
            )

        # ---------------------------------------------------
        # Whole State
        # ---------------------------------------------------

        elif trend_district == "All Districts":

            trend_analysis(
                state=trend_state,
                district=None
            )

        # ---------------------------------------------------
        # State + District
        # ---------------------------------------------------

        else:

            trend_analysis(
                state=trend_state,
                district=trend_district
            )
    trend_panel.__exit__(None, None, None)
# ── Gate: no data loaded yet ─────────────────────────────────────────────────

if df_all is None:
    st.stop()

pass

# ── Selection summary (selection remains in sidebar) ────────────────────────

sel_state = st.session_state.get("selected_state")
sel_district = st.session_state.get("selected_district")

if sel_state is None or sel_district is None:
        st.info("Select State and District in the sidebar.")
        st.stop()

# Compute train/test split from current train_end setting
sub = df_all[(df_all['state'] == sel_state) &
                                        (df_all['district'] == sel_district)].sort_values(['year', 'month'])
if sub.empty:

    st.error(
        f"No climate data found for "
        f"{sel_district.title()}, "
        f"{sel_state.title()}"
    )

    st.stop()

yr_min     = int(sub['year'].min());  yr_max = int(sub['year'].max())
n_total    = len(sub)
n_train    = int((sub['year'] <= train_end).sum())
n_test     = n_total - n_train
tr_yr_min  = int(sub['year'].min())
tr_yr_max  = int(sub[sub['year'] <= train_end]['year'].max()) if n_train > 0 else '—'
te_yr_min  = int(sub[sub['year'] >  train_end]['year'].min()) if n_test  > 0 else '—'
te_yr_max  = int(sub['year'].max())
lat_vals   = sub['latitude'].dropna()
lon_vals   = sub['longitude'].dropna()
lat_str    = f"{lat_vals.iloc[0]:.2f}°N" if len(lat_vals) > 0 else "—"
lon_str    = f"{lon_vals.iloc[0]:.2f}°E" if len(lon_vals) > 0 else "—"

# ── Gate: not yet run ─────────────────────────────────────────────────────────
model_panel = panel_containers.get("model")
if model_panel is not None:
    model_panel.__enter__()
else:
    if not panel_order:
        st.markdown(
            '<div class="cf-strip">'
            'Select an option from the sidebar.'
            '</div>',
            unsafe_allow_html=True
        )
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# RUN MODELS
# ─────────────────────────────────────────────────────────────────────────────

yrs, mos, rain, temp, lat, lon = get_district_data(df_all, sel_district)
train_mask = (
    (yrs < train_end) |
    ((yrs == train_end) & ((mos + 1) <= train_end_month_num))
)

T_tr_total = int(train_mask.sum())
T_te_total = int((~train_mask).sum())

if T_tr_total < 24:
    st.error(f"Only {T_tr_total} training months for **{sel_district}** "
             f"(train end = {train_end}). Need ≥ 24. "
             f"Try lowering the training end year.")
    st.stop()

if T_te_total < 3:
    st.error(f"Only {T_te_total} forecast months. "
             f"Try raising the training end year.")
    st.stop()

lat_display = f"{lat:.2f}°N" if not np.isnan(lat) else "N/A (using 20.0°N)"
lat_use     = lat if not np.isnan(lat) else 20.0

with st.spinner(f"Running forecast for **{sel_district}** …"):
    try:
        rain_rep_key = "mean"
        R, T, DR = run_models(
            yrs, mos, rain, temp, lat_use,
            train_end, train_end_month_num, n_paths, int(seed),
            dlm_q, q_mu0, q_gamma,
            oni_bytes, spi_scale,
            representative=rain_rep_key)
    except Exception as e:
        st.error("❌ Model failed")
        st.exception(e)
        st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# INFO STRIP + METRIC CARDS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("## 🧪 Model Validation")

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────


tab1, tab2, tab3 = st.tabs([
    "🌧  Rainfall",
    "🌡  Temperature",
    f"🏜  Drought  (SPEI-{spi_scale})"
])


# ══ Tab 1: Rainfall ══════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="cf-section">All simulated Paths</div>',
                unsafe_allow_html=True)
    fig11, ctx_r, x_ctx_r, x_fc_r, xt_r, xl_r, lta_ctx_r, lta_fc_r = \
        make_rain_paths(R, sel_district, n_paths)
    st.pyplot(fig11, use_container_width=True)
    plt.close(fig11)

    st.markdown('<div class="cf-section">Forecasted vs observed</div>',
                unsafe_allow_html=True)
    fig12 = make_rain_forecast(R, sel_district,
                               ctx_r, x_ctx_r, x_fc_r, xt_r, xl_r,
                               lta_ctx_r, lta_fc_r)
    st.plotly_chart(fig12, use_container_width=True)

    st.markdown('<div class="cf-section">Rainfall Model Quality</div>',
                unsafe_allow_html=True)
    st.table(pd.DataFrame({
            'Metric': ['RMSE (mm/day)', 'MAE (mm/day)', 'BIAS (mm/day)',
                       'Skill score', '90% CI coverage'],
            'Value':  [f"{R['acc']['rmse']:.4f}",
                       f"{R['acc']['mae']:.4f}",
                       f"{R['acc']['bias']:+.4f}",
                       f"{R['acc']['skill']:.4f}",
                       f"{R['acc']['cov90']:.1f}%"],
        }).set_index('Metric'))


# ══ Tab 2: Temperature ═══════════════════════════════════════════════════════
with tab2:
    # Keep context arrays for Plot 2.2 but do not render Plot 2.1.
    fig21, ctx_t, x_ctx_t, x_fc_t, xt_t, xl_t, lta_ctx_t, lta_fc_t = \
        make_temp_paths(T, sel_district, n_paths)
    plt.close(fig21)

    st.markdown('<div class="cf-section">Forecasted vs observed</div>',
                unsafe_allow_html=True)
    fig22 = make_temp_forecast(T, sel_district,
                               ctx_t, x_ctx_t, x_fc_t, xt_t, xl_t,
                               lta_ctx_t, lta_fc_t)
    st.plotly_chart(fig22, use_container_width=True)

    st.markdown('<div class="cf-section">Temperature Model Quality</div>',
                unsafe_allow_html=True)
    st.table(pd.DataFrame({
            'Metric': ['RMSE (°C)', 'MAE (°C)', 'BIAS (°C)',
                       'Skill score', '90% CI coverage'],
            'Value':  [f"{T['acc']['rmse']:.4f}",
                       f"{T['acc']['mae']:.4f}",
                       f"{T['acc']['bias']:+.4f}",
                       f"{T['acc']['skill']:.4f}",
                       f"{T['acc']['cov90']:.1f}%"],
        }).set_index('Metric'))


# ══ Tab 3: Drought ═══════════════════════════════════════════════════════════
with tab3:
    
    yr_ctx_dr = R['years_all'][R['T_tr'] - DR['ctx']: R['T_tr']]
    mo_ctx_dr = R['months_all'][R['T_tr'] - DR['ctx']: R['T_tr']]
    fig3 = plot_drought(
        DR, sel_district,
        x_fc   = np.arange(R['T_te']),
        xt     = xt_r,
        xl     = xl_r,
        yr_te  = R['years_all'][R['T_tr']:],
        mo_te  = R['mo_te'],
        yr_ctx = yr_ctx_dr,
        mo_ctx = mo_ctx_dr,
    )
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

    st.markdown('<div class="cf-section">Forecasted vs observed: Monthly drought risk categories</div>',
                unsafe_allow_html=True)
    st.caption(
        "Category thresholds (SPEI): No drought >= -1.0 | "
        "Moderate drought >= -2.0 and < -1.0 | Extreme drought < -2.0"
    )

    def _obs_category(val):
        """Classify observed SPI/SPEI into 3 exclusive categories."""
        if np.isnan(val):
            return '#E5E7EB', '#6B7280', 'N/A'
        elif val >= -1.0:
            return '#E1F5EE', '#0F6E56', 'None'
        elif val >= -2.0:
            return '#F0A500', '#4A2800', 'Mod'
        else:
            return '#8B0000', 'white',   'Ext'

    def _drought_gauge_html(index_name, color_accent,
                             no_p, mod_p, ext_p,
                             periods, obs_vals):
        """
        HTML card with 3 exclusive categories:
          no_drought (≥ −1.0), moderate (−2.0 to −1.0), extreme (< −2.0)
        Dominant category = argmax of the three probabilities per month.
        """
        n = len(periods)

        # ── Forecast timeline — dominant category ─────────────────────────────
        fc_cells = ''
        for p, nd, m, e in zip(periods,
                                no_p.tolist(), mod_p.tolist(), ext_p.tolist()):
            # Dominant = whichever probability is highest
            cats = [('None', nd, '#E1F5EE', '#0F6E56'),
                    ('Mod',  m,  '#F0A500', '#4A2800'),
                    ('Ext',  e,  '#8B0000', 'white')]
            label, _, bg, fc = max(cats, key=lambda x: x[1])
            mon = p.split(' ')[1][:3]
            yr  = str(p.split(' ')[0])[2:]
            fc_cells += (
                f'<div style="flex:1;min-width:0;text-align:center;'
                f'background:{bg};border-radius:4px;padding:4px 2px;margin:1px;">'
                f'<div style="font-size:9px;color:{fc};font-weight:600;">{mon}</div>'
                f'<div style="font-size:8px;color:{fc};opacity:.9;">\'{yr}</div>'
                f'</div>')

        # ── Observed timeline ─────────────────────────────────────────────────
        obs_cells = ''
        for p, v in zip(periods, obs_vals):
            bg, fc, cat = _obs_category(float(v) if not np.isnan(v) else np.nan)
            mon = p.split(' ')[1][:3]
            yr  = str(p.split(' ')[0])[2:]
            obs_cells += (
                f'<div style="flex:1;min-width:0;text-align:center;'
                f'background:{bg};border-radius:4px;padding:4px 2px;margin:1px;">'
                f'<div style="font-size:9px;color:{fc};font-weight:600;">{mon}</div>'
                f'<div style="font-size:8px;color:{fc};opacity:.9;">\'{yr}</div>'
                f'</div>')

        # ── Forecast counts (dominant category) ───────────────────────────────
        fc_n_none = fc_n_mod = fc_n_ext = 0
        for nd, m, e in zip(no_p.tolist(), mod_p.tolist(), ext_p.tolist()):
            cats = [('none', nd), ('mod', m), ('ext', e)]
            dom  = max(cats, key=lambda x: x[1])[0]
            if   dom == 'ext':  fc_n_ext  += 1
            elif dom == 'mod':  fc_n_mod  += 1
            else:               fc_n_none += 1

        # ── Observed counts ───────────────────────────────────────────────────
        obs_clean = [v for v in obs_vals if not np.isnan(v)]
        n_obs     = len(obs_clean) or 1
        obs_n_mod  = sum(1 for v in obs_clean if -2.0 <= v < -1.0)
        obs_n_ext  = sum(1 for v in obs_clean if v < -2.0)
        obs_n_none = n_obs - obs_n_mod - obs_n_ext

        return f"""<!DOCTYPE html><html><head>
<meta charset="utf-8">
<style>
  * {{ box-sizing:border-box; margin:0; padding:0;
       font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; }}
  body {{ background:transparent; padding:4px; }}
  .lbl {{ font-size:10px; font-weight:700; color:#9CA3AF;
          text-transform:uppercase; letter-spacing:.07em; margin-bottom:5px; }}
  .strip {{ display:flex; gap:2px; flex-wrap:nowrap; margin-bottom:4px; }}
  .legend {{ display:flex; gap:10px; flex-wrap:wrap; margin-bottom:8px; }}
  .leg-item {{ font-size:10px; }}
  .count-row {{ display:flex; gap:6px; flex-wrap:wrap; margin-bottom:10px; }}
  .count-pill {{ font-size:10px; font-weight:600; padding:2px 9px;
                 border-radius:20px; }}
</style></head><body>
<div style="background:white;border:1.5px solid {color_accent};
            border-radius:10px;overflow:hidden;">

  <div style="background:{color_accent};padding:10px 14px;
              display:flex;align-items:center;gap:8px;">
    <div style="font-size:13px;font-weight:600;color:white;">{index_name}</div>
    <div style="font-size:11px;color:rgba(255,255,255,.7);flex:1;">
      {spi_scale}-month · {n} months</div>
  </div>

  <div style="padding:13px;">

    <div class="lbl">Forecast — dominant category per month</div>
    <div class="strip">{fc_cells}</div>
    <div class="legend">
      <span class="leg-item" style="color:#0F6E56;">■ No drought</span>
      <span class="leg-item" style="color:#B45309;">■ Moderate</span>
      <span class="leg-item" style="color:#8B0000;">■ Extreme</span>
    </div>
    <div class="count-row">
      <span class="count-pill" style="background:#D1FAE5;color:#065F46;">
        No drought: {fc_n_none}</span>
      <span class="count-pill" style="background:#FEF3C7;color:#B45309;">
        Moderate: {fc_n_mod}</span>
      <span class="count-pill" style="background:#FEE2E2;color:#7F1D1D;">
        Extreme: {fc_n_ext}</span>
    </div>

    <div style="border-top:1px dashed #E5E7EB;margin:4px 0 10px;"></div>

    <div class="lbl">Observed — actual SPI/SPEI category</div>
    <div class="strip">{obs_cells}</div>
    <div class="legend">
      <span class="leg-item" style="color:#0F6E56;">■ No drought</span>
      <span class="leg-item" style="color:#B45309;">■ Moderate</span>
      <span class="leg-item" style="color:#8B0000;">■ Extreme</span>
    </div>
    <div class="count-row">
      <span class="count-pill" style="background:#D1FAE5;color:#065F46;">
        No drought: {obs_n_none}</span>
      <span class="count-pill" style="background:#FEF3C7;color:#B45309;">
        Moderate: {obs_n_mod}</span>
      <span class="count-pill" style="background:#FEE2E2;color:#7F1D1D;">
        Extreme: {obs_n_ext}</span>
    </div>

  </div>
</div>
</body></html>"""

    yr_te_arr = R['years_all'][R['T_tr']:]
    mo_te_arr = R['mo_te']
    periods   = [f"{yr_te_arr[i]} {MONTH_NAMES[mo_te_arr[i]]}"
                 for i in range(R['T_te'])]

    spi_obs_safe  = None   # SPI removed
    spei_obs_safe = np.where(np.isfinite(DR['spei_obs']),
                              DR['spei_obs'], np.nan)

    spei_html = _drought_gauge_html(
        f"SPEI-{spi_scale}  ·  Standardised Precipitation-Evapotranspiration Index",
        "#7B2D8B",
        DR['spei_prob']['no_drought'],
        DR['spei_prob']['moderate'],
        DR['spei_prob']['extreme'],
        periods,
        spei_obs_safe)

    components.html(spei_html, height=290, scrolling=False)

    # ── KDE section ───────────────────────────────────────────────────────────
    st.markdown(
        '<div class="cf-section">'
        f'Monthly Predictive distributions of SPEI-{spi_scale}'
        '</div>', unsafe_allow_html=True)
    st.caption(
        
        "Green = No drought | Amber = Moderate | Dark red = Extreme | "
        "Dotted line = Observed mean")

    # ── Build year → month index mapping ─────────────────────────────────────
    yr_te_arr_k = R['years_all'][R['T_tr']:]
    mo_te_arr_k = R['mo_te']

    from collections import OrderedDict as _OD
    year_month_map = _OD()
    for i in range(1, R['T_te']):
        yr  = int(yr_te_arr_k[i])
        lbl = MONTH_NAMES[mo_te_arr_k[i]]
        year_month_map.setdefault(yr, []).append((i, lbl))

    avail_years = list(year_month_map.keys())

    # ── KDE helper ────────────────────────────────────────────────────────────
    from scipy.stats import gaussian_kde as _gkde

    def _make_month_kde(ensemble, color_line, idx, month_name, index_label):
        """
        KDE figure for one forecast month. Height 260px, full-width layout.
        Falls back to histogram if KDE is singular (near-constant ensemble).
        """
        vals = ensemble[:, idx]
        vals = vals[np.isfinite(vals)]
        if len(vals) < 10:
            return None

        # Add tiny jitter to prevent singular covariance matrix
        rng = np.random.default_rng(42)
        std_v = float(np.std(vals))
        if std_v < 1e-4:
            vals = vals + rng.normal(0, 1e-3, len(vals))

        x_min = max(float(vals.min()) - 0.4, -4.5)
        x_max = min(float(vals.max()) + 0.4,  3.5)
        x_arr = np.linspace(x_min, x_max, 300)

        # Try KDE — fall back to histogram if still fails
        try:
            kde    = _gkde(vals, bw_method='silverman')
            y_arr  = kde(x_arr)
            use_kde = True
        except Exception:
            use_kde = False

        fig = go.Figure()

        if use_kde:
            # Shaded regions
            def _shade(x_lo, x_hi, fill_col, name):
                mask = (x_arr >= x_lo) & (x_arr <= x_hi)
                if not mask.any():
                    return
                xs = np.concatenate([[x_lo], x_arr[mask], [x_hi]])
                ys = np.concatenate([[float(kde(np.array([x_lo]))[0])],
                                      y_arr[mask],
                                      [float(kde(np.array([x_hi]))[0])]])
                fig.add_trace(go.Scatter(
                    x=np.concatenate([xs, xs[::-1]]),
                    y=np.concatenate([ys, np.zeros(len(ys))]),
                    fill='toself', fillcolor=fill_col,
                    line=dict(width=0), showlegend=False,
                    hoverinfo='skip'))

            _shade(x_min, -2.0, 'rgba(139,0,0,0.20)',    'Extreme')
            _shade(-2.0,  -1.0, 'rgba(240,165,0,0.20)',  'Moderate')
            _shade(-1.0, x_max, 'rgba(29,158,117,0.14)', 'No drought')

            # Boundary lines
            for xv, col in [(-2.0, '#8B0000'), (-1.0, '#B45309')]:
                if x_min < xv < x_max:
                    fig.add_vline(x=xv,
                                  line=dict(color=col, width=1.0, dash='dash'),
                                  opacity=0.65)

            # KDE curve
            fig.add_trace(go.Scatter(
                x=x_arr, y=y_arr, mode='lines',
                line=dict(color=color_line, width=2.2),
                showlegend=False,
                hovertemplate=f'%{{x:.2f}}: %{{y:.3f}}<extra></extra>'))
        else:
            # Fallback: histogram
            fig.add_trace(go.Histogram(
                x=vals.tolist(), nbinsx=30,
                marker_color=color_line, opacity=0.6,
                histnorm='probability density',
                showlegend=False,
                hovertemplate='%{x:.2f}: %{y:.3f}<extra></extra>'))
            fig.add_annotation(
                xref='paper', yref='paper', x=0.5, y=0.5,
                text='histogram — low variance',
                showarrow=False, font=dict(size=9, color='#9CA3AF'))

        # Mean line
        mean_v = float(np.mean(vals))
        fig.add_vline(x=mean_v,
                      line=dict(color=color_line, width=1.3, dash='dot'),
                      opacity=0.9)

        # Probability box
        p_nd  = float((vals >= -1.0).mean() * 100)
        p_mod = float(((vals < -1.0) & (vals >= -2.0)).mean() * 100)
        p_ext = float((vals < -2.0).mean() * 100)
        fig.add_annotation(
            xref='paper', yref='paper', x=0.97, y=0.97,
            text=(f"None {p_nd:.0f}%<br>"
                  f"Mod  {p_mod:.0f}%<br>"
                  f"Ext  {p_ext:.0f}%"),
            showarrow=False, align='right',
            bgcolor='rgba(255,255,255,0.88)',
            bordercolor='#cccccc', borderwidth=1,
            font=dict(size=9, family='monospace'))

        fig.update_layout(
            title=dict(
                text=f"<b>{index_label}</b> · {month_name}",
                font=dict(size=11), x=0.5, xanchor='center'),
            xaxis=dict(
                title=dict(text='Index value', font=dict(size=9)),
                tickfont=dict(size=8),
                gridcolor='rgba(0,0,0,0.06)',
                zeroline=True, zerolinecolor='rgba(0,0,0,0.12)'),
            yaxis=dict(
                title=dict(text='Density', font=dict(size=9)),
                tickfont=dict(size=8),
                gridcolor='rgba(0,0,0,0.06)'),
            plot_bgcolor='white', paper_bgcolor='white',
            height=260,
            margin=dict(l=45, r=10, t=36, b=40),
            hovermode='x')
        return fig

    # ── Year tabs ─────────────────────────────────────────────────────────────
    year_tab_labels = [str(y) for y in avail_years]
    year_tabs        = st.tabs(year_tab_labels)

    NCOLS = 3   # months per row

    for tab, yr in zip(year_tabs, avail_years):
        months_in_yr = year_month_map[yr]
        with tab:
            st.markdown(
                f'<div style="font-size:11px;color:#6B7280;margin-bottom:8px;">'
                f'{len(months_in_yr)} forecast months · '
                f'SPEI-{spi_scale} predictive distribution per month</div>',
                unsafe_allow_html=True)

            # Render months in rows of NCOLS
            for row_start in range(0, len(months_in_yr), NCOLS):
                row_months = months_in_yr[row_start: row_start + NCOLS]
                cols = st.columns(NCOLS)

                for col, (idx, mon_lbl) in zip(cols, row_months):
                    with col:
                        fig_spei = _make_month_kde(
                            DR['spei_ensemble'], '#7B2D8B',
                            idx, mon_lbl, f'SPEI-{spi_scale}')
                        if fig_spei:
                            st.plotly_chart(fig_spei,
                                            use_container_width=True,
                                            key=f'spei_{yr}_{idx}')

                # Thin divider between month rows
                if row_start + NCOLS < len(months_in_yr):
                    st.markdown(
                        '<hr style="border:none;border-top:0.5px solid #E5E7EB;'
                        'margin:4px 0 8px;">',
                        unsafe_allow_html=True)

if model_panel is not None and "model" in panel_order:
    model_panel.__exit__(None, None, None)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem;padding:1.2rem 1.5rem;
                        background:#F7F8FA;border-top:1px solid #E2E6EA;
                        border-radius:8px;display:flex;align-items:center;
                        flex-wrap:wrap;gap:1rem;">
    <div style="display:flex;align-items:center;gap:8px;">
        <div style="background:#0F2D52;color:white;font-size:11px;
                                font-weight:700;padding:3px 10px;border-radius:4px;
                                letter-spacing:0.04em;">
            IIT GUWAHATI
        </div>
        <span style="font-size:11px;color:#6B7280;">
            Indian Institute of Technology Guwahati
        </span>
    </div>
    <div style="flex:1;min-width:200px;"></div>
    <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
        <span style="font-size:11px;color:#9CA3AF;">Developed by</span>
        <a href="https://www.linkedin.com/in/dr-dipankar-mondal-8789329b/"
             target="_blank"
             style="display:flex;align-items:center;gap:5px;text-decoration:none;">
            <div style="width:26px;height:26px;border-radius:50%;
                                    background:#1D4E89;display:flex;align-items:center;
                                    justify-content:center;font-size:10px;font-weight:700;
                                    color:white;flex-shrink:0;">DM</div>
            <span style="font-size:11px;color:#1D4E89;font-weight:500;">
                Dr. Dipankar Mondal
            </span>
        </a>
        <span style="font-size:11px;color:#D1D5DB;">·</span>
        <a href="https://www.linkedin.com/in/debottam-ghosh/"
             target="_blank"
             style="display:flex;align-items:center;gap:5px;text-decoration:none;">
            <div style="width:26px;height:26px;border-radius:50%;
                                    background:#1D9E75;display:flex;align-items:center;
                                    justify-content:center;font-size:10px;font-weight:700;
                                    color:white;flex-shrink:0;">DG</div>
            <span style="font-size:11px;color:#1D9E75;font-weight:500;">
                Debottam Ghosh
            </span>
        </a>
    </div>
    <div style="width:100%;margin-top:8px;padding-top:8px;
                            border-top:1px solid #E5E7EB;font-size:10px;color:#9CA3AF;
                            text-align:center;">
        © 2026 Dr. Dipankar Mondal &amp; Debottam Ghosh, IIT Guwahati.
        All rights reserved.
    </div>
</div>
""", unsafe_allow_html=True)
