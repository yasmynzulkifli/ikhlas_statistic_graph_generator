import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="Ikhlas Hotel Management",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# with st.sidebar:
#     st.image("https://img.icons8.com/fluency/96/hotel.png", width=60)
#     st.markdown("## 🏨 Ikhlas Hotel")
#     st.markdown("---")
#     st.page_link("pages/home.py",                    label="🏠 Home")
#     st.page_link("pages/data_sales_single.py",       label="📊 Data Sales Individual")
#     st.page_link("pages/data_sales_all.py",          label="📈 Data Sales All Hotel")
#     st.page_link("pages/puan_yasmin.py",             label="📉 Data Sales Channels")
#     st.page_link("pages/5_Data_Payment_Channels.py", label="💳 Data Payment Channels")

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:2rem 0 0.5rem 0;'>
    <h1 style='font-size:2.6rem; font-weight:800; margin-bottom:0.3rem;'>
        🏨 Ikhlas Hotel Management
    </h1>
    <p style='color:gray; font-size:1.05rem;'>
        Central dashboard for hotel analytics & reporting
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Page cards — 2×2 grid ─────────────────────────────────────────────────────
CARDS = [
    {
        "icon": "📊",
        "title": "Data Sales Individual",
        "desc": "Upload a single hotel's daily sales CSV (Date, Sales, Bilik Sold) and generate a dual-axis chart — bars for Bilik Sold, line for Sales — with Daily / Weekly / Monthly granularity.",
        "bg": "#f0f4ff", "border": "#d0d9f5",
        "btn": "Go to Data Sales Individual →",
        "page": "pages/2_Data_Sales_Individual.py",
    },
    {
        "icon": "📈",
        "title": "Data Sales All Hotel",
        "desc": "Upload a combined CSV with all hotels (Date, Hotel, Sales, Bilik Sold) and compare performance side-by-side with grouped bar charts or a colour-coded heatmap.",
        "bg": "#f0fff4", "border": "#b7e4c7",
        "btn": "Go to Data Sales All Hotel →",
        "page": "pages/3_Data_Sales_All_Hotel.py",
    },
    {
        "icon": "📉",
        "title": "Data Sales Channels",
        "desc": "Upload a channel breakdown report (Walk In, Agoda, Booking.com, Expedia) to view grouped bars with S/D % trend lines and auto-generated copyable text per hotel.",
        "bg": "#fff8f0", "border": "#f5d0a9",
        "btn": "Go to Data Sales Channels →",
        "page": "pages/4_Data_Sales_Channels.py",
    },
    {
        "icon": "💳",
        "title": "Data Payment Channels",
        "desc": "Upload a payment method report (Cash, OTA, CC, QR) to view grouped bars with S/D % trend lines and auto-generated copyable text summaries per hotel.",
        "bg": "#f0f0ff", "border": "#c8c8f0",
        "btn": "Go to Data Payment Channels →",
        "page": "pages/5_Data_Payment_Channels.py",
    },
]

row1 = st.columns(2, gap="large")
row2 = st.columns(2, gap="large")
cols = [row1[0], row1[1], row2[0], row2[1]]

for col, card in zip(cols, CARDS):
    with col:
        st.markdown(f"""
        <div style='background:{card["bg"]}; border-radius:16px; padding:1.4rem 1.5rem 0.8rem 1.5rem;
                    border:1px solid {card["border"]}; min-height:155px;'>
            <span style='font-size:1.8rem;'>{card["icon"]}</span>
            <h3 style='margin:0.3rem 0 0.4rem 0; font-size:1.1rem;'>{card["title"]}</h3>
            <p style='color:#555; font-size:0.88rem; margin:0; line-height:1.5;'>{card["desc"]}</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button(card["btn"], key=f"btn_{card['title']}", use_container_width=True):
            st.switch_page(card["page"])

st.markdown("---")

# ── Template downloads ────────────────────────────────────────────────────────
st.markdown("### 📥 Download CSV Templates")
st.caption("Use these templates to prepare your data in the correct format before uploading.")

TEMPLATES = [
    {
        "label":    "📊 Data Sales Individual",
        "file":     "templates/template_sales_individual.csv",
        "fname":    "template_sales_individual.csv",
        "desc":     "Columns: Date, Sales, Bilik Sold",
    },
    {
        "label":    "📈 Data Sales All Hotel",
        "file":     "templates/template_sales_all_hotels.csv",
        "fname":    "template_sales_all_hotels.csv",
        "desc":     "Columns: Date, Hotel, Sales, Bilik Sold",
    },
    {
        "label":    "📉 Data Sales Channels",
        "file":     "templates/template_sales_channels.csv",
        "fname":    "template_sales_channels.csv",
        "desc":     "Columns: HOTEL, SALES, WALK IN, AGODA, B.COM, EXPEDIA, S/D % ...",
    },
    {
        "label":    "💳 Data Payment Channels",
        "file":     "templates/template_payment_channels.csv",
        "fname":    "template_payment_channels.csv",
        "desc":     "Columns: HOTEL, SALES, CASH, OTA, CC, QR, S/D % ...",
    },
]

BASE = os.path.dirname(__file__)

tcols = st.columns(4, gap="medium")
for tcol, tmpl in zip(tcols, TEMPLATES):
    with tcol:
        fpath = os.path.join(BASE, "..", tmpl["file"])
        try:
            with open(fpath, "rb") as f:
                data = f.read()
            st.download_button(
                label=tmpl["label"],
                data=data,
                file_name=tmpl["fname"],
                mime="text/csv",
                use_container_width=True,
                key=f"dl_{tmpl['fname']}",
            )
            st.caption(tmpl["desc"])
        except FileNotFoundError:
            st.warning(f"Template not found: {tmpl['file']}")

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray; font-size:0.85rem;'>"
    "Ikhlas Hotel Management System &nbsp;|&nbsp; Internal Use Only |&nbsp; Developed by AimynKifli |&nbsp; 2026"
    "</p>",
    unsafe_allow_html=True
)
