# Ikhlas Hotel — Statistic Dashboard

This Streamlit app is designed for **Ikhlas Hotel Management** to upload, review, and visualize daily hotel sales and room data.

---

## 🚀 How to Use

1. **Login**
   - Enter your username and password (provided by the admin).

2. **Upload Data**
   - Upload your daily sales report in **CSV** or **Excel** format.
   - File must contain exactly these columns:
     - `date` (DD-MM-YYYY)
     - `sales`
     - `bilik_sold`

3. **Preview**
   - Review the uploaded data.
   - If there are ambiguous date formats (e.g., DD/MM vs MM/DD), you’ll be asked to correct them.

4. **Confirm Hotel Name**
   - Select the hotel code from the dropdown list.

5. **Generate Graph**
   - Preview sales and room statistics in interactive charts.
   - Choose between **Daily**, **Weekly**, or **Monthly** granularity.

6. **Download**
   - Download the generated chart in **PNG** format for reporting or presentation.

---

## 📂 Template File

You can also download a template file (CSV or Excel) from the sidebar to ensure the correct format before uploading.

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) for the app
- [Pandas](https://pandas.pydata.org/) for data handling
- [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/) for visualization

---

## 📧 Support

If you encounter any problems or have questions, please contact:  
**nuryasmynzulkifli@gmail.com**

---
