import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ===== App setup =====
st.set_page_config(page_title="Ikhlas Hotel — Statistic", layout="wide")

USERS = {}
try:
    USERS = dict(st.secrets["auth"]["users"])
except Exception:
    st.error(
        "Auth is not configured. Add credentials in **Settings → Secrets** as:\n\n"
        "[auth]\nusers = {\"admin\":\"your-password\"}"
    )
    st.stop()

HOTEL_CODES = ["Choose Hotel","BI","MF","KZ","ST","PN","PL","JJ","ZI","NC","PJ","NL"]


# Hide sidebar while logged out
if not st.session_state.get("authed"):
    st.markdown("""
    <style>
      [data-testid="stSidebar"]{display:none;}
      [data-testid="stAppViewContainer"]{margin-left:0px;}
    </style>""", unsafe_allow_html=True)

# ===== Session state =====
def init_state():
    ss = st.session_state
    ss.setdefault("authed", False)
    ss.setdefault("username", None)
    ss.setdefault("stage", "upload")          
    ss.setdefault("df", None)
    ss.setdefault("orig_df", None)
    ss.setdefault("file_name", None)
    ss.setdefault("hotel", "Choose Hotel")
    ss.setdefault("uploader_key", 0)
    ss.setdefault("edit_buffer_df", None)
    ss.setdefault("ambiguous_reviewed", False)
    ss.setdefault("edit_row", None)           

init_state()

# ===== CSS =====
st.markdown("""
<style>
div[data-testid="stContainer"][aria-live="polite"][role="region"] > div:has(> div[data-testid="stFileUploader"]) {
  background:#fff;border:1px solid #dbece3;border-radius:14px;box-shadow:0 8px 24px rgba(3,84,63,.06);padding:18px;
}
.u-head{display:flex;align-items:center;gap:10px;font-weight:800;margin-bottom:6px;}
.u-icon{width:28px;height:28px;border-radius:8px;background:#ffe2e2;display:inline-flex;align-items:center;justify-content:center;}
.u-sub{color:#667085;margin-bottom:12px;font-size:13px;}
[data-testid="stFileUploader"]{background:transparent;}
[data-testid="stFileUploaderDropzone"]{
  position:relative;border:2px dashed #ff4b4b !important;border-radius:12px !important;
  background:#ffe2e2 !important;padding:60px 36px 40px !important;text-align:center;
}
[data-testid="stFileUploaderDropzone"] svg,[data-testid="stFileUploaderDropzone"] > div:first-child{display:none !important;}
#login-card [data-testid="stFormSubmitButton"] > button{
  width:100% !important;background:var(--primary-color) !important;color:#fff !important;
  font-weight:700 !important;border:none !important;border-radius:10px !important;padding:0.9rem 1rem !important;
}
[data-testid="stSidebar"] > div:first-child{display:flex;flex-direction:column;height:100%;}
.brand-wrap{text-align:center;padding:6px 6px 2px 6px;}
.brand-icon{font-size:22px;line-height:1;}
.brand-title{font-size:26px;font-weight:800;margin:6px 0 0 0;}
.brand-sub{color:#ff4b4b;font-size:18px;margin-top:2px;font-weight:600;}
.sb-disabled{padding:8px 12px;border:1px solid #e5e7eb;border-radius:8px;background:#f9fafb;color:#6b7280;text-align:center;}
.sb-spacer{flex:1 1 auto;}
</style>
""", unsafe_allow_html=True)

# ===== Helpers =====
AMBIGUOUS_REGEX = re.compile(r"^\s*(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\s*$")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]
    expected = {"date","sales","bilik_sold"}
    if set(df.columns) != expected:
        raise ValueError("Uploaded file must contain exactly these columns: date, sales, bilik_sold")

    df["date_raw"] = df["date"].astype(str).str.strip()
    dt_dayfirst   = pd.to_datetime(df["date_raw"], dayfirst=True,  errors="coerce", infer_datetime_format=True)
    dt_monthfirst = pd.to_datetime(df["date_raw"], dayfirst=False, errors="coerce", infer_datetime_format=True)
    parsed = dt_dayfirst.copy().fillna(dt_monthfirst)

    ambiguous = (dt_dayfirst.notna() & dt_monthfirst.notna() &
                 (dt_dayfirst.dt.normalize() != dt_monthfirst.dt.normalize()))

    def looks_ambiguous(s):
        m = AMBIGUOUS_REGEX.match(str(s))
        if not m: return False
        d, mth = int(m.group(1)), int(m.group(2))
        return d <= 12 and mth <= 12

    df["date"] = parsed
    df["ambiguous"] = ambiguous | df["date_raw"].map(looks_ambiguous)
    df["sales"] = pd.to_numeric(df["sales"], errors="raise")
    df["bilik_sold"] = pd.to_numeric(df["bilik_sold"], errors="raise")
    return df.sort_values("date").reset_index(drop=True)

def reset_to_upload():
    authed = st.session_state.get("authed", False)
    username = st.session_state.get("username")
    st.session_state.update(dict(
        stage="upload", df=None, orig_df=None, file_name=None, hotel="Choose Hotel",
        uploader_key=st.session_state.get("uploader_key", 0)+1,
        edit_buffer_df=None, ambiguous_reviewed=False, edit_row=None,
        authed=authed, username=username
    ))

def make_template_df():
    today = pd.Timestamp.today().normalize()
    dates = pd.date_range(end=today, periods=7, freq="D")
    return pd.DataFrame({"date": dates.strftime("%d-%m-%Y"), "sales":[0]*7, "bilik_sold":[0]*7})

def aggregate(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    if granularity == "Daily":
        return df.copy()
    rule = "W" if granularity == "Weekly" else "M"
    return (df.set_index("date").resample(rule).sum(numeric_only=True)[["sales","bilik_sold"]].reset_index())

def kpis(df: pd.DataFrame) -> dict:
    total_sales = float(df["sales"].sum()); total_rooms = int(df["bilik_sold"].sum())
    if len(df):
        start = df["date"].min().strftime("%d-%m-%Y")
        end   = df["date"].max().strftime("%d-%m-%Y")
        idx_s = df["sales"].idxmax(); best_sales_day = df.loc[idx_s,"date"].strftime("%d-%m-%Y"); 
        best_sales_value = float(df.loc[idx_s,"sales"])
        idx_r = df["bilik_sold"].idxmax(); best_rooms_day = df.loc[idx_r,"date"].strftime("%d-%m-%Y"); 
        best_rooms_value = int(df.loc[idx_r,"bilik_sold"])
    else:
        start=end=best_sales_day=best_rooms_day="-"; best_sales_value=best_rooms_value=float("nan")
    return dict(total_sales=total_sales,total_rooms=total_rooms,period=f"{start} → {end}",
                best_sales_day=best_sales_day,best_sales_value=best_sales_value,
                best_rooms_day=best_rooms_day,best_rooms_value=best_rooms_value)

def draw_dual_axis_with_labels(df: pd.DataFrame, hotel_code: str, granularity: str):
    labels = (df["date"].dt.strftime("%d-%m-%Y") if granularity=="Daily"
              else df["date"].dt.strftime("W%V %Y") if granularity=="Weekly"
              else df["date"].dt.strftime("%b %Y"))
    rooms = df["bilik_sold"].astype(float).values
    sales = df["sales"].astype(float).values

    fig, ax1 = plt.subplots(figsize=(11,6))
    sns.set(style="whitegrid", rc={"grid.alpha": 0.3})

    bars = ax1.bar(labels, rooms, color="#ffc7c7", label="Room Sold")
    ax1.set_ylabel("Rooms Sold (units)")
    ax1.tick_params(axis="x", rotation=45)
    for r, v in zip(bars, rooms):
        ax1.text(r.get_x() + r.get_width()/2, r.get_height(), f"{int(v)} Unit",
                 ha="center", va="bottom", fontsize=10)

    ax2 = ax1.twinx()
    ax2.plot(labels, sales, linestyle="--", linewidth=1, marker="o",
             label="Sales (RM)", color="#5b2020")
    ax2.set_ylabel("Sales (RM)")
    for x, y in zip(range(len(labels)), sales):
        ax2.text(x + 0.1, y, f"RM{y:,.2f}", ha="left", va="top",
                 fontsize=10, color="#5b2020")


    if len(sales) >= 2:
        ax2.plot([0, len(labels)-1], [sales[0], sales[-1]],
                 linestyle="-", color="#9c8bff", linewidth=1.5, label="Sales Trend")

    t1 = df["date"].min().strftime("%d-%m-%Y")
    t2 = df["date"].max().strftime("%d-%m-%Y")
    ax1.set_title(f"Rooms vs Sales — {hotel_code} \n{t1} to {t2}")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")
    fig.tight_layout()
    return fig, f"Rooms_vs_Sales_{hotel_code}_{granularity}_{t1}_to_{t2}"


# ===== Login page =====
def render_login():
    st.markdown("<div style='height:7vh'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;">
      <div style="width:64px;height:64px;border-radius:999px;background:#ffe2e2;display:inline-flex;align-items:center;justify-content:center;">
        <span style="font-size:28px;">🏨</span>
      </div>
      <h1 style="margin:10px 0 6px;font-size:28px;font-weight:800;">Welcome Buddy!</h1>
      <p style="color:#6b7280;margin:0;">Sign in to access the Ikhlas Hotel Management Statistics Team dashboard</p>
    </div>""", unsafe_allow_html=True)
    _, mid, _ = st.columns([1.5,2,1.5])
    with mid:
        st.markdown('<div id="login-card">', unsafe_allow_html=True)
        with st.form("login_form"):
            st.markdown('<div style="display:flex;align-items:center;gap:8px;font-weight:700;color:#ff3232;margin-bottom:4px;">🛡️ <span>Secure Login</span></div><p style="margin-top:0;color:#6b7280;font-size:13px;">Please enter your credentials below.</p>', unsafe_allow_html=True)
            u = st.text_input("Username", placeholder="Enter your username")
            p = st.text_input("Password", placeholder="Enter your password", type="password")
            if st.form_submit_button("Sign In", width='stretch', type="primary"):
                if u in USERS and USERS[u] == p:
                    st.session_state.authed = True; st.session_state.username = u; st.rerun()
                else:
                    st.error("Invalid credentials.")
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ===== Sidebar =====
with st.sidebar:
    if st.session_state.get("authed"):
        st.markdown("""<div class="brand-wrap"><div class="brand-icon">🏨</div>
        <div class="brand-title">Ikhlas Hotel Management</div><div class="brand-sub">Statistics Tools</div></div>""", unsafe_allow_html=True)
        st.markdown("---")
        tmpl = make_template_df()
        st.subheader("Download Template")
        st.caption("Columns: **date (DD-MM-YYYY)**, **sales**, **bilik_sold**")
        st.download_button("CSV template", tmpl.to_csv(index=False).encode("utf-8"),
                           "hotel_stats_template.csv", "text/csv", width='stretch')
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as wr: tmpl.to_excel(wr, index=False, sheet_name="template")
        bio.seek(0)
        st.download_button("Excel template", bio.getvalue(),
                           "hotel_stats_template.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           width='stretch')
        st.markdown("---"); st.subheader("Other Apps")
        st.markdown('<div class="sb-disabled">Daily Statistics (coming soon)</div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-spacer"></div>', unsafe_allow_html=True)
        st.markdown("---")
        if st.button("Logout", width='stretch'):
            for k in ["authed","username","stage","df","orig_df","file_name","hotel","edit_buffer_df","ambiguous_reviewed","edit_row"]:
                st.session_state.pop(k, None)
            st.session_state["uploader_key"] = st.session_state.get("uploader_key",0)+1
            st.rerun()

# ===== Main flow =====
if not st.session_state.authed:
    render_login()

st.title("Statistic Insight")

# ---- UPLOAD
if st.session_state.stage == "upload":
    st.markdown('<div id="upload-card-marker"></div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("""
        <div class="u-head">
          <span class="u-icon">
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#ff4b4b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M12 21V9"/><path d="M8 13l4-4 4 4"/><path d="M4 24h16"/>
            </svg>
          </span>
          <div>Upload CSV File</div>
        </div>
        <div class="u-sub">Select your daily sales report file to begin the analysis process</div>
        """, unsafe_allow_html=True)

        up = st.file_uploader(
            "Drop your CSV/Excel file here or click to browse",
            type=["csv","xlsx"], accept_multiple_files=False,
            label_visibility="collapsed", key=f"uploader_{st.session_state.uploader_key}"
        )
        st.markdown('<div id="proceed-marker"></div>', unsafe_allow_html=True)
        proceed = st.button("Proceed to Data Review", 
                            width='stretch',
                              type="primary",
                              icon= ":material/arrow_forward:",
                              disabled=(up is None))

if proceed:
    try:
        # Save file contents into session_state right away
        file_bytes = up.getvalue()
        st.session_state["uploaded_file_bytes"] = file_bytes
        st.session_state["uploaded_file_name"] = up.name

        # Load DataFrame from the saved bytes
        if up.name.lower().endswith(".xlsx"):
            df_raw = pd.read_excel(io.BytesIO(file_bytes))
        else:
            df_raw = pd.read_csv(io.BytesIO(file_bytes))

        df = normalize_columns(df_raw)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    st.session_state.df = df.copy()
    st.session_state.orig_df = df.copy()
    st.session_state.file_name = up.name

    if int(df["ambiguous"].sum()) > 0 and not st.session_state.get("ambiguous_reviewed", False):
        st.session_state.stage = "ambiguity"
    else:
        st.session_state.stage = "preview"
    st.rerun()

# ---- AMBIGUITY REVIEW (inline editor instead of st.modal)
elif st.session_state.stage == "ambiguity":
    df = st.session_state.df
    amb = df[df["ambiguous"]].copy()
    amb_cnt = int(amb.shape[0])

    st.markdown(f"""
    <div style="background:#FFF8E1;border:1px solid #F6D98A;border-radius:10px;
                padding:12px 14px;margin:6px 0 12px 0;display:flex;gap:10px;align-items:center;">
      <span style="font-size:18px;">⚠️</span>
      <div>
        <div style="font-weight:700;">{amb_cnt} ambiguous date format{'s' if amb_cnt!=1 else ''} detected</div>
        <div style="color:#6b7280;">Please review and confirm the corrections below.</div>
      </div>
    </div>""", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("**Data Corrections Required**")
        st.caption("Review the detected issues and confirm the suggested corrections.")

        # Header
        h = st.columns([2.0, 2.0, 1.2, 1.0, 2.2, 0.9])
        h[0].markdown("**Original Date**"); h[1].markdown("**Corrected Date**")
        h[2].markdown("**Sales**"); h[3].markdown("**Items Sold**")
        h[4].markdown("**Issue**"); h[5].markdown("**Action**")

        # Rows
        for i, row in amb.iterrows():
            c = st.columns([2.0, 2.0, 1.2, 1.0, 2.2, 0.9])
            c[0].write(row["date_raw"])
            c[1].write(row["date"].strftime("%d-%m-%Y") if pd.notna(row["date"]) else "-")
            c[2].write(f"RM{float(row['sales']):,.2f}")
            c[3].write(int(row["bilik_sold"]))
            c[4].write("Ambiguous date (DD/MM vs MM/DD)")
            if c[5].button("Edit", key=f"edit_{i}", width='stretch'):
                st.session_state.edit_row = int(i); st.rerun()

        st.markdown("---")

        # Quick stats and actions
        total_records = len(df)
        date_rng = f"{df['date'].min().strftime('%d %b %Y')} – {df['date'].max().strftime('%d %b %Y')}"
        c1,c2,c3 = st.columns(3)
        with c1:
            st.markdown("**Total Records**")
            st.markdown(f"<div style='font-size:28px;font-weight:800'>{total_records:,}</div>", unsafe_allow_html=True)
            st.caption(f"🟢 {amb_cnt} require correction")
        with c2:
            st.markdown("**Date Range**")
            st.markdown(f"<div style='font-size:28px;font-weight:800'>{date_rng}</div>", unsafe_allow_html=True)
        with c3:
            st.markdown("**Data Quality**")
            st.markdown("<div style='font-size:28px;font-weight:800'>Needs review</div>", unsafe_allow_html=True)

        left, right = st.columns([1,2])
        with left:
            if st.button("← Back to Upload", width='stretch'):
                reset_to_upload(); st.rerun()
        with right:
            if st.button("Confirm & Continue →", type="primary", width='stretch'):
                st.session_state.ambiguous_reviewed = True
                st.session_state.stage = "preview"; st.rerun()

    # INLINE EDITOR (no st.modal)
    edit_idx = st.session_state.get("edit_row")
    if edit_idx is not None:
        with st.container(border=True):
            st.subheader("Edit ambiguous date")
            row = st.session_state.df.loc[edit_idx]
            st.markdown(f"**Original string:** `{row['date_raw']}`")
            current = pd.to_datetime(row["date"]).date() if pd.notna(row["date"]) else pd.Timestamp.today().date()
            with st.form(f"edit_form_{edit_idx}", clear_on_submit=False):
                new_date = st.date_input("Choose the correct date", value=current, format="DD-MM-YYYY", key=f"d_{edit_idx}")
                col_a, col_b = st.columns(2)
                save = col_a.form_submit_button("Save", type="primary")
                cancel = col_b.form_submit_button("Cancel")
                if save:
                    nd = pd.Timestamp(new_date)
                    st.session_state.df.at[edit_idx, "date"] = nd
                    st.session_state.df.at[edit_idx, "date_raw"] = nd.strftime("%d-%m-%Y")
                    st.session_state.df.at[edit_idx, "ambiguous"] = False
                    st.session_state.df = st.session_state.df.sort_values("date").reset_index(drop=True)
                    st.session_state.edit_row = None
                    if int(st.session_state.df["ambiguous"].sum()) == 0:
                        st.session_state.ambiguous_reviewed = True
                        st.session_state.stage = "preview"
                    st.rerun()
                if cancel:
                    st.session_state.edit_row = None; st.rerun()

# ---- PREVIEW
elif st.session_state.stage == "preview":
    df = st.session_state.df
    st.subheader("Preview & Select Hotel")

    if df["date"].isna().any():
        st.error("Detected invalid dates (NaT). Please fix before continuing.")
    else:
        st.caption(f"File: **{st.session_state.file_name}** — {len(df)} records")
        show = df.copy(); show["date"] = show["date"].dt.strftime("%d-%m-%Y")
        st.dataframe(show[["date","sales","bilik_sold"]], width='stretch', hide_index=True)

        colA, colB = st.columns([3,1], vertical_alignment="top")
        with colB:
            st.session_state.hotel = st.selectbox("Select Hotel", HOTEL_CODES, index=0)
            disabled = (st.session_state.hotel == "Choose Hotel")
            if st.button("Confirm & Generate Graph", type="primary", disabled=disabled, width='stretch'):
                st.session_state.stage = "graph"; st.rerun()

        if st.button("Edit Data"):
            st.session_state.edit_buffer_df = df[["date","sales","bilik_sold"]].copy()
            st.session_state.stage = "edit"; st.rerun()

        if st.button("Start Over"):
            reset_to_upload(); st.rerun()

# ---- EDIT
elif st.session_state.stage == "edit":
    st.subheader("Edita Date Data")
    if st.session_state.edit_buffer_df is None:
        st.session_state.edit_buffer_df = st.session_state.df[["date","sales","bilik_sold"]].copy()
    editor = st.data_editor(
        st.session_state.edit_buffer_df, width='stretch', hide_index=True, num_rows="fixed",
        column_config={
            "date": st.column_config.DateColumn("date", format="DD-MM-YYYY"),
            "sales": st.column_config.NumberColumn("sales", step=0.01, format="%.2f", min_value=0.0),
            "bilik_sold": st.column_config.NumberColumn("bilik_sold", step=1, format="%d", min_value=0),
        },
        key="date_number_editor",
    )
    c1,c2,c3 = st.columns(3)
    if c1.button("Apply changes", type="primary"):
        new = editor.copy()
        new["date"] = pd.to_datetime(new["date"], errors="coerce")
        new["sales"] = pd.to_numeric(new["sales"], errors="coerce")
        new["bilik_sold"] = pd.to_numeric(new["bilik_sold"], errors="coerce")
        if new.isna().any().any():
            st.error("Some values are invalid (empty/NaN). Please correct them before applying.")
        else:
            st.session_state.df[["date","sales","bilik_sold"]] = new[["date","sales","bilik_sold"]]
            st.session_state.df["ambiguous"] = False
            st.session_state.df["date_raw"] = st.session_state.df["date"].dt.strftime("%d-%m-%Y")
            st.session_state.df = st.session_state.df.sort_values("date").reset_index(drop=True)
            st.session_state.edit_buffer_df = None
            st.success("Changes applied.")
            st.session_state.stage = "preview"; st.rerun()
    if c2.button("↩️ Reset to original (from file)"):
        base = st.session_state.orig_df; st.session_state.edit_buffer_df = base[["date","sales","bilik_sold"]].copy()
    if c3.button("Cancel (back to preview)"):
        st.session_state.edit_buffer_df = None; st.session_state.stage = "preview"; st.rerun()

# ---- GRAPH
elif st.session_state.stage == "graph":
    base_df = st.session_state.df; hotel = st.session_state.hotel
    st.subheader("Graph & Data")
    k = kpis(base_df)
    c1,c2,c3 = st.columns(3)
    c1.metric("Total Sales (RM)", f"{k['total_sales']:,.2f}")
    c2.metric("Total Rooms", f"{k['total_rooms']:,}")
    c3.metric("Period", k["period"])
    with st.expander("Highlights", expanded=False):
        sv = "-" if pd.isna(k["best_sales_value"]) else f"RM{k['best_sales_value']:,.2f}"
        rv = "-" if pd.isna(k["best_rooms_value"]) else f"{int(k['best_rooms_value']):,}"
        st.write(f"• Highest Sales: **{sv}** on **{k['best_sales_day']}**")
        st.write(f"• Most Rooms Sold: **{rv}** on **{k['best_rooms_day']}**")

    gran = st.selectbox("Granularity", ["Daily","Weekly","Monthly"], index=0)
    gdf = aggregate(base_df, granularity=gran)
    fig, title = draw_dual_axis_with_labels(gdf, hotel, granularity=gran)
    st.pyplot(fig)

# --- Download & Start Over ---
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
    buf.seek(0)

    col1, col2 = st.columns([1,1])
    with col1:
        clicked = st.download_button(
            "Download Chart (PNG)",
            data=buf.getvalue(),
            file_name=f"{title}.png",
            mime="image/png",
            type="primary",
            key="dl_chart",
            help="After download, you'll be returned to the Upload step.",
            icon=':material/download:'
        )

    with col2:
        start_over = st.button("Start Over", key="btn_start_over", type="secondary")

    if clicked:
        reset_to_upload()
        st.rerun()

    if start_over:
        reset_to_upload()
        st.rerun()


