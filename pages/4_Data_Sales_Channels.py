import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import io

st.set_page_config(
    page_title="Sales Channels Report — Puan Yasmin",
    page_icon="🏩",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ── Colour map ────────────────────────────────────────────────────────────────
COLORS = {
    "WALK IN":  "#ff9900",
    "AGODA":    "#cc3399",
    "BOOKING":  "#3399cc",
    "EXPEDIA":  "#466900",
}

# Column name → display label → colour key
# Bar channels (grouped, 4 OTA/walkin channels)
BAR_COLS = [
    ("WALK IN",  "Walk In",   "WALK IN"),
    ("AGODA",    "Agoda",     "AGODA"),
    ("B.COM",    "Booking",   "BOOKING"),
    ("EXPEDIA",  "Expedia",   "EXPEDIA"),
]
# Line channels — S/D % for each (same colour as bar)
LINE_COLS = [
    ("S/D % Walk In",      "S/D% Walk In",    "WALK IN"),
    ("S/D % Agoda",        "S/D% Agoda",      "AGODA"),
    ("S/D % Booking.Com",  "S/D% Booking",    "BOOKING"),
    ("S/D % Expedia",      "S/D% Expedia",    "EXPEDIA"),
]


HOTEL_ORDER = ["ZI","KZ","BI","NC","MF","ST","JJ","PN","PL","NL","PJ"]

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {"py_stage":"upload", "py_df":None, "py_file":"", "py_uploader_key":0}.items():
    if k not in st.session_state:
        st.session_state[k] = v

def reset():
    st.session_state.py_stage        = "upload"
    st.session_state.py_df           = None
    st.session_state.py_file         = ""
    st.session_state.py_uploader_key += 1

def find_col(df, candidates):
    up = {c.strip().upper(): c for c in df.columns}
    for c in candidates:
        if c.strip().upper() in up:
            return up[c.strip().upper()]
    return None

def load_df(file_bytes, fname):
    if fname.lower().endswith(".xlsx"):
        raw = pd.read_excel(io.BytesIO(file_bytes))
    else:
        raw = pd.read_csv(io.BytesIO(file_bytes))

    # Normalise column names (strip whitespace)
    raw.columns = [str(c).strip() for c in raw.columns]

    # Verify required columns exist
    required = ["HOTEL","SALES","WALK IN","AGODA","B.COM","EXPEDIA",
                "S/D % Sales","S/D % Walk In","S/D % Agoda","S/D % Booking.Com","S/D % Expedia"]
    missing  = [r for r in required if find_col(raw, [r]) is None]
    if missing:
        st.error(f"Missing columns: **{', '.join(missing)}**. Found: {list(raw.columns)}")
        st.stop()

    # Coerce numerics
    for c in raw.columns:
        if c != "HOTEL":
            raw[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0)

    raw["HOTEL"] = raw["HOTEL"].astype(str).str.strip().str.upper()

    # Sort by fixed hotel order
    order_map = {h: i for i, h in enumerate(HOTEL_ORDER)}
    raw["_sort"] = raw["HOTEL"].map(order_map).fillna(99)
    raw = raw.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)

    return raw

# ── Chart ─────────────────────────────────────────────────────────────────────
def draw_chart(df: pd.DataFrame):
    hotels  = df["HOTEL"].tolist()
    n       = len(hotels)
    n_bars  = len(BAR_COLS)
    grp_w   = 0.75
    bar_w   = grp_w / n_bars
    offsets = np.linspace(-(grp_w - bar_w) / 2, (grp_w - bar_w) / 2, n_bars)
    x       = np.arange(n)

    fig_w = max(13, n * n_bars * 0.22 + 3)

    # GridSpec: chart (top, 70%) + table (bottom, 30%) with breathing room
    fig = plt.figure(figsize=(fig_w, 10))
    fig.patch.set_facecolor("white")
    gs  = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.2)
    ax1 = fig.add_subplot(gs[0])
    ax_tbl = fig.add_subplot(gs[1])

    ax1.set_facecolor("#f9f9f9")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#ddd")
    ax1.grid(axis="y", color="#e8e8e8", linewidth=0.5, zorder=0)

    # ── Grouped bars ──────────────────────────────────────────────────────────
    bar_handles = []
    for i, (col, label, ck) in enumerate(BAR_COLS):
        vals  = df[col].values.astype(float)
        color = COLORS[ck]
        bars  = ax1.bar(x + offsets[i], vals, bar_w * 0.9,
                        color=color, label=label, zorder=3,
                        edgecolor="white", linewidth=0.5)
        bar_handles.append(mpatches.Patch(color=color, label=label))
        for bar, val in zip(bars, vals):
            if val > 0:
                ax1.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 200,
                         f"RM{val:,.0f}",
                         ha="center", va="bottom",
                         fontsize=6.5, color="#333", fontweight="bold", rotation=90)

    max_sales = max((df[c].max() for c, _, _ in BAR_COLS), default=1)
    ax1.set_ylim(0, max_sales * 1.40)
    ax1.set_ylabel("Sales (RM)", fontsize=10, color="#333")
    ax1.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"RM {v/1000:,.0f}k" if v >= 1000 else f"RM {v:,.0f}")
    )
    ax1.tick_params(axis="y", labelcolor="#333", labelsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(hotels, fontsize=10, color="#222")
    ax1.tick_params(axis="x", pad=6)
    ax1.set_xlim(-0.6, n - 0.4)

    # ── S/D % lines (right axis) ──────────────────────────────────────────────
    ax2 = ax1.twinx()
    ax2.set_facecolor("none")
    ax2.set_ylabel("S/D %", fontsize=10, color="#555")
    ax2.tick_params(axis="y", labelcolor="#555", labelsize=8)
    ax2.axhline(0, color="#bbb", linewidth=0.8, linestyle=":", zorder=2)

    line_handles = []
    for col, label, ck in LINE_COLS:
        vals  = df[col].values.astype(float)
        color = COLORS[ck]
        ax2.plot(x, vals, color=color, linewidth=1,
                 marker="o", markersize=2, zorder=5, linestyle="--")
        line_handles.append(
            mlines.Line2D([], [], color=color, linewidth=1,
                          linestyle="--", marker="o", markersize=2, label=label)
        )

    sd_all = np.concatenate([df[c].values.astype(float) for c, _, _ in LINE_COLS])
    sd_max = max(abs(sd_all.max()), abs(sd_all.min()), 20)
    ax2.set_ylim(-sd_max * 1.5, sd_max * 1.5)

    # ── Legend ────────────────────────────────────────────────────────────────
    ax1.legend(
        handles=bar_handles + line_handles,
        loc="upper right", fontsize=8,
        facecolor="white", edgecolor="#ccc",
        ncol=2,
        title="  Bars (Sales RM)   Lines (S/D %)",
        title_fontsize=7.5
    )
    ax1.set_title("Sales by Channel & S/D % — All Hotels",
                  fontsize=12, fontweight="bold", color="#111", pad=10)

    # ── S/D % table in its own axes ───────────────────────────────────────────
    ax_tbl.axis("off")
    table_rows    = []
    row_labels    = []
    row_colors_tbl = []
    cell_colors   = []

    for col, label, ck in LINE_COLS:
        vals = df[col].values.astype(float)
        table_rows.append([f"{v:+.1f}%" for v in vals])
        row_labels.append(label)
        row_colors_tbl.append(COLORS[ck])
        cell_colors.append([
            "#d4f5d4" if v > 0 else "#fdd5d5" if v < 0 else "#f0f0f0"
            for v in vals
        ])

    tbl = ax_tbl.table(
        cellText=table_rows,
        rowLabels=row_labels,
        colLabels=hotels,
        cellColours=cell_colors,
        rowColours=row_colors_tbl,
        rowLoc="right",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor("#e8e8e8")
            cell.set_text_props(fontweight="bold", color="#222", fontsize=9)
        if c == -1 and r > 0:
            cell.set_facecolor(row_colors_tbl[r - 1])
            cell.set_text_props(fontweight="bold", color="white", fontsize=8)
        if c == -1 and r == 0:
            cell.set_facecolor("#e8e8e8")

    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE
# ═══════════════════════════════════════════════════════════════════════════════
st.title("🏩 Sales Channels Report — Puan Yasmin")
st.caption("Upload the sales channel report to view stacked bar chart by hotel with S/D % lines.")
st.markdown("---")

# ── STAGE: UPLOAD ─────────────────────────────────────────────────────────────
if st.session_state.py_stage == "upload":
    with st.container(border=True):
        st.markdown("#### 📂 Upload Report File")
        st.caption("Expected columns: Hotel, Walk In, Agoda, B.Com, Expedia, S/D % Walk In, S/D % Agoda, S/D % Booking.Com, S/D % Expedia")

        up = st.file_uploader(
            "Drop your CSV/Excel file here",
            type=["csv","xlsx"], accept_multiple_files=False,
            label_visibility="collapsed",
            key=f"py_uploader_{st.session_state.py_uploader_key}"
        )
        proceed = st.button("Proceed →", type="primary",
                            disabled=(up is None), use_container_width=True)

    if proceed:
        try:
            file_bytes = up.getvalue()
            df = load_df(file_bytes, up.name)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()
        st.session_state.py_df    = df
        st.session_state.py_file  = up.name
        st.session_state.py_stage = "graph"
        st.rerun()

# ── STAGE: GRAPH ──────────────────────────────────────────────────────────────
elif st.session_state.py_stage == "graph":
    df = st.session_state.py_df
    st.caption(f"File: **{st.session_state.py_file}** — {len(df)} hotels")

    # KPIs
    with st.container(border=True):
        cols = st.columns(5)
        cols[0].metric("Total Sales",    f"RM {df['SALES'].sum():,.2f}")
        cols[1].metric("Total Walk In",  f"RM {df['WALK IN'].sum():,.2f}")
        cols[2].metric("Total Agoda",    f"RM {df['AGODA'].sum():,.2f}")
        cols[3].metric("Total Booking",  f"RM {df['B.COM'].sum():,.2f}")
        cols[4].metric("Total Expedia",  f"RM {df['EXPEDIA'].sum():,.2f}")

    st.markdown("")

    # Chart
    fig = draw_chart(df)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    buf.seek(0)

    dl1, dl2 = st.columns([1, 1])
    dl1.download_button("⬇️ Download Chart", buf,
                        file_name="puan_yasmin_report.png", mime="image/png",
                        use_container_width=True)
    if dl2.button("🔄 Upload New File", use_container_width=True):
        reset(); st.rerun()

    st.pyplot(fig)

    # Copyable text summary — single iframe so JS+buttons share same context
    with st.expander("📋 Copy-able Text Summary", expanded=False):
        import json, streamlit.components.v1 as components

        def sd_word(val):
            return f"increased by {val:.2f}%" if val >= 0 else f"decreased by {abs(val):.2f}%"

        channels = [("Walk-ins","WALK IN","S/D % Walk In"),("Agoda","AGODA","S/D % Agoda"),("Booking.com","B.COM","S/D % Booking.Com"),("Expedia","EXPEDIA","S/D % Expedia")]

        def make_block(row):
            lines = [f"{row['HOTEL']} – Sales {sd_word(row['S/D % Sales'])} compared to last month"]
            for idx, (name, col, sd_col) in enumerate(channels, start=1):
                if row[col] == 0:
                    lines.append(f"{idx}.  No reservation in {name}")
                else:
                    lines.append(f"{idx}.  {name} {sd_word(row[sd_col])}")
            return "\n".join(lines)

        all_blocks = [make_block(row) for _, row in df.iterrows()]
        all_text   = "\n\n".join(all_blocks)
        t_json     = json.dumps(all_blocks)
        a_json     = json.dumps(all_text)

        cards_html = ""
        for i, block in enumerate(all_blocks):
            disp = block.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            cards_html += (
                f'<div class="copy-card">{disp}\n'
                f'<button class="copy-btn" onclick="copyHotel(this,{i})">📋 Copy</button></div>\n'
            )

        html = f"""<!DOCTYPE html><html><head><style>
        body{{margin:0;font-family:sans-serif;}}
        .copy-card{{background:#f8f9fa;border:1px solid #e0e0e0;border-radius:10px;
                   padding:14px 16px 10px;margin-bottom:10px;font-family:monospace;
                   font-size:13px;line-height:1.7;white-space:pre-wrap;}}
        .copy-btn{{display:inline-flex;align-items:center;gap:5px;margin-top:8px;
                  padding:5px 14px;background:#4e79a7;color:white;border:none;
                  border-radius:6px;font-size:12px;font-weight:600;cursor:pointer;transition:background 0.2s;}}
        .copy-btn:hover{{background:#3a5f8a;}}
        .copy-btn.copied{{background:#2e9e5b;}}
        .copy-all-btn{{display:inline-flex;align-items:center;gap:6px;padding:9px 22px;
                      margin-bottom:16px;background:#2e9e5b;color:white;border:none;
                      border-radius:8px;font-size:13px;font-weight:700;cursor:pointer;transition:background 0.2s;}}
        .copy-all-btn:hover{{background:#236e40;}}
        .copy-all-btn.copied{{background:#1a5230;}}
        </style></head><body>
        <script>
        const _texts = {t_json};
        const _all   = {a_json};
        function legacyCopy(text){{
            const ta=document.createElement("textarea");
            ta.value=text;
            ta.setAttribute("readonly","");
            ta.style="position:absolute;left:-9999px;";
            document.body.appendChild(ta);
            ta.select();
            document.execCommand("copy");
            document.body.removeChild(ta);
        }}
        function copyHotel(btn,idx){{
            legacyCopy(_texts[idx]);
            btn.classList.add("copied");
            btn.innerHTML="✅ Copied!";
            setTimeout(()=>{{btn.classList.remove("copied");btn.innerHTML="📋 Copy";}},10000);
        }}
        function copyAll(btn){{
            legacyCopy(_all);
            btn.classList.add("copied");
            btn.innerHTML="✅ All Copied!";
            setTimeout(()=>{{btn.classList.remove("copied");btn.innerHTML="📋 Copy All Hotels";}},10000);
        }}
        </script>
        <button class="copy-all-btn" onclick="copyAll(this)">📋 Copy All Hotels</button>
        {cards_html}
        </body></html>"""

        components.html(html, height=len(all_blocks)*155+80, scrolling=True)

    # Data table
    with st.expander("📋 View data table"):
        show = df.copy()
        fmt  = {c: "RM {:.2f}" for c in ["SALES","WALK IN","AGODA","B.COM","EXPEDIA"]}
        fmt.update({c: "{:.2f}%" for c in ["S/D % Sales","S/D % Walk In","S/D % Agoda","S/D % Booking.Com","S/D % Expedia"]})
        st.dataframe(show.style.format(fmt), use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray; font-size:0.85rem;'>"
    "Ikhlas Hotel Management System &nbsp;|&nbsp; Internal Use Only &nbsp;|&nbsp; Developed by AimynKifli &nbsp;|&nbsp; 2026"
    "</p>",
    unsafe_allow_html=True
)

