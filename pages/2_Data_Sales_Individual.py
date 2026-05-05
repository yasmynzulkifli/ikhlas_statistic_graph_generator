import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import io, re

st.set_page_config(
    page_title="Sales — Single Hotel",
    page_icon="📊",
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

# ── Constants ─────────────────────────────────────────────────────────────────
HOTEL_CODES = ["Choose Hotel","ZI","KZ","BI","NC","MF","ST","JJ","PN","PL","NL","PJ","PD"]
# ── Session state init ────────────────────────────────────────────────────────
for k, v in {
    "stage": "upload",
    "df": None,
    "orig_df": None,
    "file_name": "",
    "hotel": "Choose Hotel",
    "edit_buffer_df": None,
    "ambiguous_reviewed": False,
    "uploader_key": 0,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

def reset_to_upload():
    st.session_state.stage            = "upload"
    st.session_state.df               = None
    st.session_state.orig_df          = None
    st.session_state.file_name        = ""
    st.session_state.hotel            = "Choose Hotel"
    st.session_state.edit_buffer_df   = None
    st.session_state.ambiguous_reviewed = False
    st.session_state.uploader_key     += 1

# ── Date parsing — handles mixed formats like 1/9/2025 and 13/09/25 ──────────
def parse_single_date(raw: str):
    """Try multiple formats; return (Timestamp, ambiguous_flag)."""
    raw = str(raw).strip()
    # Normalise separators
    raw_n = re.sub(r"[.\-]", "/", raw)
    parts = raw_n.split("/")

    if len(parts) == 3:
        a, b, c = parts
        # Detect short year
        if len(c) == 2:
            c = "20" + c
        # If first part > 12 it must be day
        if len(a) <= 2 and int(a) > 12:
            # unambiguous: D/M/YYYY
            try:
                return pd.Timestamp(f"{c}-{int(b):02d}-{int(a):02d}"), False
            except Exception:
                pass
        # If second part > 12 it must be day (M/D/YYYY American)
        if len(b) <= 2 and int(b) > 12:
            try:
                return pd.Timestamp(f"{c}-{int(a):02d}-{int(b):02d}"), False
            except Exception:
                pass
        # Both ≤ 12: ambiguous — default to D/M/YYYY but flag it
        if len(a) <= 2 and int(a) <= 12 and len(b) <= 2 and int(b) <= 12:
            try:
                ts = pd.Timestamp(f"{c}-{int(b):02d}-{int(a):02d}")
                return ts, True
            except Exception:
                pass

    # Fallback
    try:
        return pd.to_datetime(raw, dayfirst=True), False
    except Exception:
        return pd.NaT, False

def normalize_columns(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Find Date/Sales/BilikSold columns, parse dates, return clean df."""
    cols_up = {c.strip().upper(): c for c in df_raw.columns}

    def fc(cands):
        for c in cands:
            if c.upper() in cols_up:
                return cols_up[c.upper()]
        return None

    date_col  = fc(["DATE","TARIKH","TRX DATE","TRANSACTION DATE"])
    sales_col = fc(["SALES","JUALAN","AMOUNT","REVENUE","TOTAL"])
    bilik_col = fc(["BILIK SOLD","BILIK","ROOMS","ROOMS SOLD","ROOM SOLD","BILIK DIJUAL"])

    missing = [n for n,c in [("Date",date_col),("Sales",sales_col),("Bilik Sold",bilik_col)] if c is None]
    if missing:
        st.error(f"Cannot find columns: **{', '.join(missing)}**. Found: {list(df_raw.columns)}")
        st.stop()

    rows = []
    for _, row in df_raw.iterrows():
        ts, amb = parse_single_date(str(row[date_col]))
        rows.append({
            "date_raw":  str(row[date_col]).strip(),
            "date":      ts,
            "sales":     pd.to_numeric(row[sales_col], errors="coerce"),
            "bilik_sold":pd.to_numeric(row[bilik_col], errors="coerce"),
            "ambiguous": amb,
        })
    out = pd.DataFrame(rows)
    out["sales"]      = out["sales"].fillna(0)
    out["bilik_sold"] = out["bilik_sold"].fillna(0)
    return out.sort_values("date").reset_index(drop=True)

# ── Aggregation ───────────────────────────────────────────────────────────────
def aggregate(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    d = df.copy()
    if granularity == "Weekly":
        d["period"] = d["date"].dt.to_period("W").apply(lambda p: p.start_time)
    elif granularity == "Monthly":
        d["period"] = d["date"].dt.to_period("M").apply(lambda p: p.start_time)
    else:
        d["period"] = d["date"]
    return d.groupby("period", as_index=False).agg(
        sales=("sales","sum"), bilik_sold=("bilik_sold","sum")
    ).sort_values("period")

# ── KPIs ──────────────────────────────────────────────────────────────────────
def kpis(df: pd.DataFrame) -> dict:
    valid = df.dropna(subset=["date"])
    period = (f"{valid['date'].min().strftime('%d %b %Y')} – "
              f"{valid['date'].max().strftime('%d %b %Y')}") if not valid.empty else "–"
    best_s_idx = df["sales"].idxmax()    if not df["sales"].empty else None
    best_r_idx = df["bilik_sold"].idxmax() if not df["bilik_sold"].empty else None
    return {
        "total_sales":       df["sales"].sum(),
        "total_rooms":       int(df["bilik_sold"].sum()),
        "period":            period,
        "best_sales_day":    df.loc[best_s_idx,"date"].strftime("%d %b %Y") if best_s_idx is not None else "–",
        "best_sales_value":  df.loc[best_s_idx,"sales"] if best_s_idx is not None else None,
        "best_rooms_day":    df.loc[best_r_idx,"date"].strftime("%d %b %Y") if best_r_idx is not None else "–",
        "best_rooms_value":  df.loc[best_r_idx,"bilik_sold"] if best_r_idx is not None else None,
    }

# ── Dual-axis chart ───────────────────────────────────────────────────────────
def draw_dual_axis(gdf: pd.DataFrame, hotel: str, granularity: str):
    import seaborn as sns

    if granularity == "Daily":
        labels = gdf["period"].dt.strftime("%d-%m-%Y").tolist()
    elif granularity == "Weekly":
        labels = gdf["period"].dt.strftime("W%V %Y").tolist()
    else:
        labels = gdf["period"].dt.strftime("%b %Y").tolist()

    rooms = gdf["bilik_sold"].astype(float).values
    sales = gdf["sales"].astype(float).values

    sns.set(style="whitegrid", rc={"grid.alpha": 0.3})
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Bars — Bilik Sold (left axis)
    bars = ax1.bar(labels, rooms, color="#FFB2B2", label="Room Sold")
    ax1.set_ylabel("Rooms Sold (units)")
    ax1.tick_params(axis="x", rotation=45)
    for r, v in zip(bars, rooms):
        ax1.text(r.get_x() + r.get_width() / 2, r.get_height(),
                 f"{int(v)}", ha="center", va="bottom", fontsize=10)

    # Line — Sales (right axis)
    ax2 = ax1.twinx()
    ax2.plot(labels, sales, linestyle="--", linewidth=1,
             marker="o", label="Sales (RM)", color="#400030", markersize=2)
    ax2.set_ylabel("Sales (RM)")
    for x_pos, y_val in zip(range(len(labels)), sales):
        ax2.text(x_pos + 0.1, y_val, f"RM{y_val:,.2f}",
                 ha="left", va="top", fontsize=8, color="#400030")

    # Trend line across first to last sales point
    if len(sales) >= 2:
        ax2.plot([0, len(labels) - 1], [sales[0], sales[-1]],
                 linestyle="-", color="#E36A6A", linewidth=1.5, label="Sales Trend")

    # Title
    t1 = gdf["period"].min().strftime("%d-%m-%Y")
    t2 = gdf["period"].max().strftime("%d-%m-%Y")
    ax1.set_title(f"Rooms vs Sales — {hotel}\n{t1} to {t2}")

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")

    fig.tight_layout()
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE TITLE
# ═══════════════════════════════════════════════════════════════════════════════
st.title("📊 Sales — Single Hotel")
st.caption("Upload a daily sales CSV (Date, Sales, Bilik Sold) to review, edit and visualise.")
st.markdown("---")

# ── STAGE: UPLOAD ─────────────────────────────────────────────────────────────
if st.session_state.stage == "upload":
    with st.container(border=True):
        st.markdown("#### 📂 Upload CSV File")
        st.caption("Select your daily sales report file to begin the analysis.")

        up = st.file_uploader(
            "Drop your CSV/Excel file here or click to browse",
            type=["csv","xlsx"], accept_multiple_files=False,
            label_visibility="collapsed",
            key=f"uploader_{st.session_state.uploader_key}"
        )
        proceed = st.button("Proceed to Data Review →", type="primary",
                            disabled=(up is None), use_container_width=True)

    if proceed:
        try:
            file_bytes = up.getvalue()
            st.session_state["uploaded_file_bytes"] = file_bytes
            st.session_state["uploaded_file_name"]  = up.name
            if up.name.lower().endswith(".xlsx"):
                df_raw = pd.read_excel(io.BytesIO(file_bytes))
            else:
                df_raw = pd.read_csv(io.BytesIO(file_bytes))
            df = normalize_columns(df_raw)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()

        st.session_state.df        = df.copy()
        st.session_state.orig_df   = df.copy()
        st.session_state.file_name = up.name

        if int(df["ambiguous"].sum()) > 0 and not st.session_state.get("ambiguous_reviewed", False):
            st.session_state.stage = "ambiguity"
        else:
            st.session_state.stage = "preview"
        st.rerun()

# ── STAGE: AMBIGUITY REVIEW ───────────────────────────────────────────────────
elif st.session_state.stage == "ambiguity":
    df    = st.session_state.df
    amb   = df[df["ambiguous"]].copy()
    amb_cnt = int(amb.shape[0])

    st.warning(f"⚠️ **{amb_cnt} ambiguous date format{'s' if amb_cnt!=1 else ''} detected** — "
               f"the day and month could not be determined with certainty (both ≤ 12). "
               f"Please confirm the correct dates below.")

    if amb_cnt == 0:
        st.session_state.ambiguous_reviewed = True
        st.session_state.stage = "preview"
        st.rerun()

    with st.container(border=True):
        st.markdown("**Bulk Edit Ambiguous Dates**")
        st.caption("Edit the corrected_date column, then click **Apply all corrections**.")

        amb = amb.reset_index().rename(columns={"index": "_orig_idx"})
        if "corrected_date" not in amb.columns:
            amb["corrected_date"] = amb["date"]

        editor = st.data_editor(
            amb[["_orig_idx","date_raw","corrected_date","sales","bilik_sold"]],
            hide_index=True, use_container_width=True, num_rows="fixed",
            column_config={
                "_orig_idx":      st.column_config.Column("Row", disabled=True),
                "date_raw":       st.column_config.Column("Original Date", disabled=True),
                "corrected_date": st.column_config.DateColumn("Corrected Date", format="DD-MM-YYYY"),
                "sales":          st.column_config.NumberColumn("Sales", format="%.2f", disabled=True),
                "bilik_sold":     st.column_config.NumberColumn("Bilik Sold", format="%d", disabled=True),
            },
            key="bulk_amb_editor",
        )

        c1, c2 = st.columns(2)
        if c1.button("← Back to Upload", use_container_width=True):
            reset_to_upload(); st.rerun()

        apply_all = c2.button("Apply all corrections →", type="primary", use_container_width=True)

        still_empty = editor[editor["corrected_date"].isna()]
        if not still_empty.empty:
            st.warning(f"{len(still_empty)} row(s) still missing a corrected date.")

        if apply_all:
            if editor["corrected_date"].isna().any():
                st.error("Please fill all corrected dates before applying.")
                st.stop()
            edited = editor.copy()
            edited["corrected_date"] = pd.to_datetime(edited["corrected_date"], errors="coerce")
            if edited["corrected_date"].isna().any():
                st.error("Some corrected dates are invalid. Please fix them.")
                st.stop()
            for _, r in edited.iterrows():
                idx = int(r["_orig_idx"])
                nd  = pd.Timestamp(r["corrected_date"])
                st.session_state.df.at[idx, "date"]     = nd
                st.session_state.df.at[idx, "date_raw"] = nd.strftime("%d-%m-%Y")
                st.session_state.df.at[idx, "ambiguous"]= False
            st.session_state.df = st.session_state.df.sort_values("date").reset_index(drop=True)
            st.session_state.ambiguous_reviewed = True
            st.session_state.stage = "preview"
            st.rerun()

# ── STAGE: PREVIEW ────────────────────────────────────────────────────────────
elif st.session_state.stage == "preview":
    df = st.session_state.df
    st.subheader("📋 Preview & Select Hotel")

    if df["date"].isna().any():
        st.error("Detected invalid dates. Please edit and fix before continuing.")
    else:
        st.caption(f"File: **{st.session_state.file_name}** — {len(df)} records")
        show = df.copy()
        show["date"] = show["date"].dt.strftime("%d-%m-%Y")
        st.dataframe(show[["date","sales","bilik_sold"]], use_container_width=True, hide_index=True)

        colA, colB = st.columns([3, 1], vertical_alignment="top")
        with colB:
            st.session_state.hotel = st.selectbox("Select Hotel", HOTEL_CODES, index=0)
            disabled = (st.session_state.hotel == "Choose Hotel")
            if st.button("Confirm & Generate Graph →", type="primary",
                         disabled=disabled, use_container_width=True):
                st.session_state.stage = "graph"; st.rerun()

        c1, c2 = st.columns(2)
        if c1.button("✏️ Edit Data", use_container_width=True):
            st.session_state.edit_buffer_df = df[["date","sales","bilik_sold"]].copy()
            st.session_state.stage = "edit"; st.rerun()
        if c2.button("🔄 Start Over", use_container_width=True):
            reset_to_upload(); st.rerun()

# ── STAGE: EDIT ───────────────────────────────────────────────────────────────
elif st.session_state.stage == "edit":
    st.subheader("✏️ Edit Data")
    if st.session_state.edit_buffer_df is None:
        st.session_state.edit_buffer_df = st.session_state.df[["date","sales","bilik_sold"]].copy()

    editor = st.data_editor(
        st.session_state.edit_buffer_df, use_container_width=True,
        hide_index=True, num_rows="fixed",
        column_config={
            "date":       st.column_config.DateColumn("Date", format="DD-MM-YYYY"),
            "sales":      st.column_config.NumberColumn("Sales", step=0.01, format="%.2f", min_value=0.0),
            "bilik_sold": st.column_config.NumberColumn("Bilik Sold", step=1, format="%d", min_value=0),
        },
        key="date_number_editor",
    )

    c1, c2, c3 = st.columns(3)
    if c1.button("✅ Apply changes", type="primary", use_container_width=True):
        new = editor.copy()
        new["date"]       = pd.to_datetime(new["date"], errors="coerce")
        new["sales"]      = pd.to_numeric(new["sales"],      errors="coerce")
        new["bilik_sold"] = pd.to_numeric(new["bilik_sold"], errors="coerce")
        if new.isna().any().any():
            st.error("Some values are invalid. Please correct them before applying.")
        else:
            st.session_state.df[["date","sales","bilik_sold"]] = new[["date","sales","bilik_sold"]]
            st.session_state.df["ambiguous"] = False
            st.session_state.df["date_raw"]  = st.session_state.df["date"].dt.strftime("%d-%m-%Y")
            st.session_state.df = st.session_state.df.sort_values("date").reset_index(drop=True)
            st.session_state.edit_buffer_df  = None
            st.session_state.stage = "preview"; st.rerun()

    if c2.button("↩️ Reset to original", use_container_width=True):
        base = st.session_state.orig_df
        st.session_state.edit_buffer_df = base[["date","sales","bilik_sold"]].copy()

    if c3.button("✖ Cancel", use_container_width=True):
        st.session_state.edit_buffer_df = None
        st.session_state.stage = "preview"; st.rerun()

# ── STAGE: GRAPH ──────────────────────────────────────────────────────────────
elif st.session_state.stage == "graph":
    base_df = st.session_state.df
    hotel   = st.session_state.hotel

    st.subheader(f"📈 {hotel} — Sales Report")

    k = kpis(base_df)
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([2,2,2,1], vertical_alignment="center")
        c1.metric("Period",           k["period"])
        c2.metric("Total Rooms Sold", f"{k['total_rooms']:,}")
        c3.metric("Total Sales (RM)", f"RM {k['total_sales']:,.2f}")
        with c4:
            gran = st.selectbox("Granularity", ["Daily","Weekly","Monthly"], index=0)

    st.markdown("")
    gdf = aggregate(base_df, granularity=gran)
    fig = draw_dual_axis(gdf, hotel, granularity=gran)

    # Download on top
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    dl_col1, dl_col2, dl_col3 = st.columns([1,1,1])
    dl_col1.download_button("⬇️ Download Chart", buf,
                             file_name=f"{hotel}_sales.png", mime="image/png",
                             use_container_width=True)
    if dl_col2.button("✏️ Edit Data", use_container_width=True):
        st.session_state.edit_buffer_df = base_df[["date","sales","bilik_sold"]].copy()
        st.session_state.stage = "edit"; st.rerun()
    if dl_col3.button("🔄 Start Over", use_container_width=True):
        reset_to_upload(); st.rerun()

    st.pyplot(fig)

    # Data table
    with st.expander("📋 View data table"):
        show = gdf.copy()
        show["period"] = pd.to_datetime(show["period"]).dt.strftime("%d/%m/%Y")
        show = show.rename(columns={"period":"Date","sales":"Sales (RM)","bilik_sold":"Bilik Sold"})
        st.dataframe(show.style.format({"Sales (RM)":"RM {:.2f}","Bilik Sold":"{:.0f}"}),
                     use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray; font-size:0.85rem;'>"
    "Ikhlas Hotel Management System &nbsp;|&nbsp; Internal Use Only &nbsp;|&nbsp; Developed by AimynKifli &nbsp;|&nbsp; 2026"
    "</p>",
    unsafe_allow_html=True
)
