import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np
import io

st.set_page_config(
    page_title="Sales — All Hotels",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# with st.sidebar:
#     st.image("https://img.icons8.com/fluency/96/hotel.png", width=60)
#     st.markdown("## 🏨 Ikhlas Hotel")
#     st.markdown("---")
#     st.page_link("pages/home.py",              label="🏠 Home")
#     st.page_link("pages/data_sales_single.py", label="📊 Sales — Single Hotel")
#     st.page_link("pages/data_sales_all.py",    label="📈 Sales — All Hotels")
#     st.page_link("pages/puan_yasmin.py",       label="👩‍💼 Puan Yasmin")

st.title("📈 Sales — All Hotels")
st.caption("Upload a combined CSV (Date, Hotel, Sales, Bilik Sold) to explore performance across all hotels.")
st.markdown("---")

# Fixed hotel order — PJ added at end when available
HOTEL_ORDER = ["ZI","KZ","BI","NC","MF","ST","JJ","PN","PL","NL","PJ","PD"]

# Harmonious muted qualitative palette — easy on the eyes, clearly distinct
HOTEL_PALETTE = [
    "#ffcc00",  # ZI  — steel blue
    "#ff9900",  # KZ  — sage green
    "#ff6600",  # BI  — warm amber
    "#cc3399",  # NC  — muted coral
    "#990066",  # MF  — soft teal
    "#3399cc",  # ST  — golden yellow
    "#006699",  # JJ  — dusty mauve
    "#ccee66",  # PN  — soft rose
    "#99cc33",  # PL  — warm brown
    "#669900",  # NL  — warm grey
    "#466900",  # PJ  — deep teal
    "#5f7c8a",  # PD — muted blue-grey
]

def parse_date_col(series):
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y"):
        try:
            return pd.to_datetime(series, format=fmt)
        except Exception:
            pass
    return pd.to_datetime(series, dayfirst=True, errors="coerce")

def find_col(df, candidates):
    cols_upper = {c.strip().upper(): c for c in df.columns}
    for c in candidates:
        if c.upper() in cols_upper:
            return cols_upper[c.upper()]
    return None

# ── File upload ───────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload combined hotels CSV", type=["csv"],
    help="CSV must have: Date (DD/MM/YYYY), Hotel, Sales, Bilik Sold"
)

if uploaded is None:
    st.info("👆 Please upload a CSV file. Expected format:")
    st.dataframe(pd.DataFrame({
        "Date":       ["01/01/2024","01/01/2024","02/01/2024"],
        "Hotel":      ["ZI","KZ","ZI"],
        "Sales":      [12000, 8500, 13500],
        "Bilik Sold": [45, 30, 48],
    }), use_container_width=False)
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

with st.expander("🔍 Preview raw data", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)

date_col  = find_col(df, ["DATE","TARIKH","TRX DATE","TRANSACTION DATE"])
hotel_col = find_col(df, ["HOTEL","PROPERTY","HOTEL CODE","HOTEL NAME","CODE"])
sales_col = find_col(df, ["SALES","JUALAN","AMOUNT","REVENUE","TOTAL"])
bilik_col = find_col(df, ["BILIK SOLD","BILIK","ROOMS","ROOMS SOLD","ROOM SOLD","BILIK DIJUAL"])

missing = [n for n,c in [("Date",date_col),("Hotel",hotel_col),
                          ("Sales",sales_col),("Bilik Sold",bilik_col)] if c is None]
if missing:
    st.error(f"Could not auto-detect columns: **{', '.join(missing)}**. Found: {list(df.columns)}")
    st.stop()

df = df.rename(columns={date_col:"Date", hotel_col:"Hotel",
                         sales_col:"Sales", bilik_col:"BilikSold"})
df["Date"]      = parse_date_col(df["Date"])
df["Hotel"]     = df["Hotel"].astype(str).str.strip().str.upper()
df["Sales"]     = pd.to_numeric(df["Sales"],     errors="coerce").fillna(0)
df["BilikSold"] = pd.to_numeric(df["BilikSold"], errors="coerce").fillna(0)
df.dropna(subset=["Date"], inplace=True)
df.sort_values("Date", inplace=True)

date_min = df["Date"].min().date()
date_max = df["Date"].max().date()

# Order hotels by fixed sequence; unknowns appended at end
present   = set(df["Hotel"].unique())
all_hotels = [h for h in HOTEL_ORDER if h in present] + sorted(present - set(HOTEL_ORDER))
hotel_color_map = {h: HOTEL_PALETTE[i % len(HOTEL_PALETTE)] for i,h in enumerate(all_hotels)}

# ── Filters ───────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
with c1:
    sel_hotels = st.multiselect("Filter by Hotel", options=all_hotels, default=all_hotels)
with c2:
    start_date = st.date_input("From Date", value=date_min, min_value=date_min, max_value=date_max)
with c3:
    end_date   = st.date_input("To Date",   value=date_max, min_value=date_min, max_value=date_max)

if not sel_hotels:
    st.warning("Please select at least one hotel.")
    st.stop()
if start_date > end_date:
    st.error("Start date must be before end date.")
    st.stop()

fdf = df[
    df["Hotel"].isin(sel_hotels) &
    (df["Date"].dt.date >= start_date) &
    (df["Date"].dt.date <= end_date)
]
if fdf.empty:
    st.warning("No data found for the selected filters.")
    st.stop()

# ── Pivot ─────────────────────────────────────────────────────────────────────
pivot_sales = (fdf.groupby(["Date","Hotel"])["Sales"]
               .sum().unstack(fill_value=0)
               .reindex(columns=sel_hotels, fill_value=0))
pivot_bilik = (fdf.groupby(["Date","Hotel"])["BilikSold"]
               .sum().unstack(fill_value=0)
               .reindex(columns=sel_hotels, fill_value=0))

dates_idx = pivot_sales.index
n_dates   = len(dates_idx)
n_hotels  = len(sel_hotels)
x_center  = np.arange(n_dates)

# ── KPIs ──────────────────────────────────────────────────────────────────────
st.markdown("### 📌 Summary")
k1,k2,k3,k4 = st.columns(4)
k1.metric("Total Sales",      f"RM {fdf['Sales'].sum():,.0f}")
k2.metric("Total Bilik Sold", f"{fdf['BilikSold'].sum():,.0f}")
k3.metric("Hotels",           n_hotels)
k4.metric("Days",             n_dates)
st.markdown("---")

# ── Chart type selector ───────────────────────────────────────────────────────
chart_type = st.radio(
    "Select Chart Type",
    options=["📊 Grouped Bar", "🟥 Heatmap"],
    horizontal=True
)
st.markdown("")

# Shared x-tick helper
def get_xticks(n, idx, step_override=None):
    step   = step_override or max(1, n // 18)
    pos    = x_center[::step]
    labels = [pd.Timestamp(idx[i]).strftime("%d %b\n'%y") for i in range(0, n, step)]
    return pos, labels

def style_ax(ax):
    ax.set_facecolor("#f7f7f7")
    ax.tick_params(colors="#333")
    ax.yaxis.label.set_color("#333")
    for spine in ax.spines.values():
        spine.set_edgecolor("#ccc")
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.5, zorder=0)

# ═════════════════════════════════════════════════════════════════════════════
# GROUPED BAR (Sales) + LINE (Bilik Sold)
# ═════════════════════════════════════════════════════════════════════════════
if chart_type == "📊 Grouped Bar":

    group_w = 0.8
    bar_w   = group_w / max(n_hotels, 1)
    offsets = np.linspace(-(group_w - bar_w) / 2,
                           (group_w - bar_w) / 2,
                           n_hotels)

    fig_w = min(32, max(12, n_dates * n_hotels * 0.18 + 4))
    fig, (ax_s, ax_b) = plt.subplots(
        1, 2, figsize=(fig_w, 5.5),
        gridspec_kw={"wspace": 0.38}
    )
    fig.patch.set_facecolor("white")
    style_ax(ax_s)
    style_ax(ax_b)

    # ── LEFT: Grouped bars — Sales ───────────────────────────────────────────
    for i, hotel in enumerate(sel_hotels):
        vals  = pivot_sales[hotel].values if hotel in pivot_sales.columns else np.zeros(n_dates)
        color = hotel_color_map[hotel]
        ax_s.bar(x_center + offsets[i], vals, bar_w * 0.92,
                 color=color, alpha=0.9, label=hotel, zorder=3)

    ax_s.set_title("Sales (RM)", color="#222",
                   fontsize=11, fontweight="bold", pad=8)
    ax_s.set_ylabel("Sales (RM)", color="#333", fontsize=10)
    ax_s.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v,_: f"RM {v/1000:,.0f}k" if v>=1000 else f"RM {v:,.0f}")
    )
    ax_s.tick_params(axis="y", labelcolor="#333", labelsize=8)
    ax_s.set_xlabel("Date", color="#333", fontsize=10)

    # ── RIGHT: Grouped bars — Bilik Sold ─────────────────────────────────────
    for i, hotel in enumerate(sel_hotels):
        vals  = pivot_bilik[hotel].values if hotel in pivot_bilik.columns else np.zeros(n_dates)
        color = hotel_color_map[hotel]
        ax_b.bar(x_center + offsets[i], vals, bar_w * 0.92,
                 color=color, alpha=0.9, zorder=3)

    ax_b.set_title("Bilik Sold", color="#222",
                   fontsize=11, fontweight="bold", pad=8)
    ax_b.set_ylabel("Bilik Sold", color="#333", fontsize=10)
    ax_b.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:,.0f}"))
    ax_b.tick_params(axis="y", labelcolor="#333", labelsize=8)
    ax_b.set_xlabel("Date", color="#333", fontsize=10)

    # X ticks for both
    tick_pos, tick_labels = get_xticks(n_dates, dates_idx)
    for ax in (ax_s, ax_b):
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, color="#333", fontsize=7.5, ha="center")
        ax.set_xlim(x_center[0] - 0.6, x_center[-1] + 0.6)

    # Shared legend
    handles = [mpatches.Patch(color=hotel_color_map[h], label=h) for h in sel_hotels]
    fig.legend(handles=handles, loc="lower center", ncol=min(n_hotels, 8),
               facecolor="white", edgecolor="#ccc", labelcolor="#333",
               fontsize=8.5, title="Hotel", title_fontsize=9,
               bbox_to_anchor=(0.5, -0.06))
    plt.suptitle("Daily Sales & Bilik Sold by Hotel",
                 fontsize=12, fontweight="bold", color="#111", y=1.01)
    plt.tight_layout()
    dl_filename = "sales_all_hotels_bar.png"

# ═════════════════════════════════════════════════════════════════════════════
# HEATMAP — side by side
# ═════════════════════════════════════════════════════════════════════════════
else:
    # Matrices: rows = hotels, cols = dates
    mat_sales = pivot_sales.T.reindex(sel_hotels).fillna(0).values
    mat_bilik = pivot_bilik.T.reindex(sel_hotels).fillna(0).values

    step      = max(1, n_dates // 20)
    tick_pos  = list(range(0, n_dates, step))
    tick_lbls = [pd.Timestamp(dates_idx[i]).strftime("%d %b\n'%y") for i in tick_pos]

    row_h  = max(0.5, min(1.2, 10 / max(n_hotels, 1)))
    fig_h  = row_h * n_hotels + 2.0
    fig_w  = min(36, max(14, n_dates * 0.3 + 3)) * 2   # doubled for side-by-side

    fig, (ax_s, ax_b) = plt.subplots(
        1, 2, figsize=(fig_w, fig_h),
        gridspec_kw={"wspace": 0.35}
    )
    fig.patch.set_facecolor("white")

    def draw_heatmap(ax, matrix, title, cmap, fmt_fn):
        vmax = matrix.max() if matrix.max() > 0 else 1
        im   = ax.imshow(matrix, aspect="auto", cmap=cmap,
                         interpolation="nearest", vmin=0, vmax=vmax)

        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.ax.tick_params(labelsize=7, colors="#444")
        cbar.outline.set_edgecolor("#ccc")

        # Annotate cells when not too dense
        if n_dates <= 40:
            for row in range(matrix.shape[0]):
                for col in range(matrix.shape[1]):
                    val = matrix[row, col]
                    if val > 0:
                        txt_color = "white" if (val / vmax) > 0.6 else "#222"
                        ax.text(col, row, fmt_fn(val),
                                ha="center", va="center",
                                fontsize=6.5, color=txt_color, fontweight="bold")

        ax.set_title(title, color="#222", fontsize=11, fontweight="bold", pad=10)
        ax.set_yticks(range(n_hotels))
        ax.set_yticklabels(sel_hotels, fontsize=9, color="#333")
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lbls, fontsize=7.5, color="#333", ha="center")
        ax.tick_params(length=0)
        # Cell dividers
        ax.set_xticks(np.arange(-0.5, n_dates, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_hotels, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    draw_heatmap(ax_s, mat_sales, "🟩  Sales (RM)", "YlGn",
                 lambda v: f"{v/1000:.0f}k" if v >= 1000 else f"{v:.0f}")
    draw_heatmap(ax_b, mat_bilik, "🟧  Bilik Sold", "YlOrRd",
                 lambda v: f"{v:.0f}")

    plt.suptitle("Daily Sales & Bilik Sold Heatmap — Hotels × Dates",
                 fontsize=13, fontweight="bold", color="#111", y=1.02)
    plt.tight_layout()
    dl_filename = "sales_all_hotels_heatmap.png"

# ── Download + render ─────────────────────────────────────────────────────────
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
buf.seek(0)
st.download_button("⬇️ Download Chart as PNG", buf,
                   file_name=dl_filename, mime="image/png")
st.pyplot(fig)

# ── Data tables ───────────────────────────────────────────────────────────────
with st.expander("📋 Data — Sales by Hotel (per day)"):
    show_s = pivot_sales.copy()
    show_s.index = show_s.index.strftime("%d/%m/%Y")
    st.dataframe(show_s.style.format("RM {:.0f}").background_gradient(cmap="Greens"),
                 use_container_width=True)
with st.expander("📋 Data — Bilik Sold by Hotel (per day)"):
    show_b = pivot_bilik.copy()
    show_b.index = show_b.index.strftime("%d/%m/%Y")
    st.dataframe(show_b.style.format("{:.0f}").background_gradient(cmap="Blues"),
                 use_container_width=True)

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray; font-size:0.85rem;'>"
    "Ikhlas Hotel Management System &nbsp;|&nbsp; Internal Use Only &nbsp;|&nbsp; Developed by AimynKifli &nbsp;|&nbsp; 2026"
    "</p>",
    unsafe_allow_html=True
)
