import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="Digital Payments Dashboard", layout="wide")

st.title("Digital Payments & Currency Analytics")
st.markdown(
    "This dashboard presents a professional analytics view of digital payment volumes and their impact on currency in circulation. "
    "Upload your dataset, explore trends, inspect correlations, and predict future circulation values."
)

# Sidebar dataset control
st.sidebar.header("Dataset Upload")
file_upload = st.sidebar.file_uploader(
    "Upload dataset (Excel or CSV)", type=["xlsx", "xls", "csv"]
)

default_file_path = "ABA FINAL PROJECT.xlsx"

@st.cache_data
def load_dataset(uploaded_file):
    if uploaded_file is None:
        return pd.read_excel(default_file_path)
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)

try:
    df = load_dataset(file_upload)
except Exception as exc:
    st.error(f"Unable to load dataset: {exc}")
    st.stop()

required_columns = [
    "UPI_Volume",
    "DebitCard_Volume",
    "CreditCard_Volume",
    "Currency_in_Circulation",
]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    st.error("Dataset is missing required columns: " + ", ".join(missing_columns))
    st.write("Your dataset columns:", df.columns.tolist())
    st.stop()

st.sidebar.markdown("---")
st.sidebar.write(f"Rows: {len(df)}")
st.sidebar.write(f"Columns: {len(df.columns)}")

# Prepare date/year-based trends if available
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.strftime("%b")
else:
    df["Year"] = np.nan
    df["Month"] = ""

# Train regression model without changing existing logic
X = df[["UPI_Volume", "DebitCard_Volume", "CreditCard_Volume"]]
y = df["Currency_in_Circulation"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# Build dashboard tabs
tab1, tab2, tab3 = st.tabs([
    "Overview",
    "Trend Analysis",
    "Prediction Model",
])

with tab1:
    st.header("Overview 🔎")
    st.write(
        "The Overview tab summarizes the dataset, key payment volume metrics, and average currency in circulation. "
        "This section is ideal for quick executive-level insights."
    )

    total_upi = df["UPI_Volume"].sum()
    total_debit = df["DebitCard_Volume"].sum()
    total_credit = df["CreditCard_Volume"].sum()
    avg_cic = df["Currency_in_Circulation"].mean()

    year_2024_df = df[df["Year"] == 2024]
    total_upi_2024 = year_2024_df["UPI_Volume"].sum()
    total_other_2024 = year_2024_df["DebitCard_Volume"].sum() + year_2024_df["CreditCard_Volume"].sum()

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Total UPI Volume", f"{total_upi:,.0f}")
    metric_col2.metric("Total Debit Card Volume", f"{total_debit:,.0f}")
    metric_col3.metric("Total Credit Card Volume", f"{total_credit:,.0f}")
    metric_col4.metric("Average Currency in Circulation", f"{avg_cic:,.2f}")

    st.write("#### 2024 Volume Highlights")
    highlight_col1, highlight_col2 = st.columns(2)
    highlight_col1.metric(
        "UPI Volume in 2024",
        f"{total_upi_2024:,.0f}" if not year_2024_df.empty else "N/A",
    )
    highlight_col2.metric(
        "Other Volume in 2024",
        f"{total_other_2024:,.0f}" if not year_2024_df.empty else "N/A",
    )

    st.divider()
    st.subheader("Dataset Summary")
    st.dataframe(df[required_columns].describe().T)
    st.divider()
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

with tab2:
    st.header("Trend Analysis 📈")
    st.write(
        "Analyze digital payment trajectories over time and explore UPI growth patterns for selected years."
    )

    if df["Year"].notna().any():
        year_summary = (
            df.groupby("Year")[
                ["UPI_Volume", "Currency_in_Circulation"]
            ]
            .sum()
            .reset_index()
            .sort_values("Year")
        )

        st.divider()
        selected_year = st.selectbox(
            "Choose year for UPI trend",
            year_summary["Year"].astype(int).tolist(),
            index=0,
        )

        year_df = df[df["Year"] == selected_year].sort_values("Date")
        if year_df.empty:
            st.warning(f"No data available for {selected_year}.")
        else:
            st.subheader(f"Monthly UPI Trend for {selected_year}")
            st.line_chart(
                year_df.set_index("Date")["UPI_Volume"],
                width="stretch",
                height=300,
            )

            year_df["UPI_Growth_Pct"] = year_df["UPI_Volume"].pct_change() * 100
            st.subheader("Monthly UPI Growth Percentage")
            st.bar_chart(
                year_df.set_index("Month")["UPI_Growth_Pct"].fillna(0),
                width="stretch",
                height=350,
            )

            annual_growth = year_summary.copy()
            annual_growth["UPI_Growth_Pct"] = annual_growth["UPI_Volume"].pct_change() * 100
            max_growth_year = int(annual_growth.loc[annual_growth["UPI_Growth_Pct"].idxmax(), "Year"])
            max_growth_value = annual_growth["UPI_Growth_Pct"].max()

            st.divider()
            st.subheader("Annual UPI Growth")
            st.line_chart(
                annual_growth.set_index("Year")["UPI_Growth_Pct"].fillna(0),
                width="stretch",
                height=300,
            )
            st.success(
                f"Maximum UPI growth was in {max_growth_year} with {max_growth_value:.2f}% growth."
            )
    else:
        st.info("Year-based trend analysis is unavailable because the dataset does not include a parsable Date column.")

with tab3:
    st.header("Prediction Model 🧠")
    st.write(
        "Enter payment volume inputs to predict currency in circulation, then review model quality metrics for the trained regression model."
    )

    st.divider()
    input_col1, input_col2, input_col3 = st.columns(3)
    upi_value = input_col1.number_input(
        "UPI Volume",
        min_value=0,
        value=int(X["UPI_Volume"].median()),
        step=1,
    )
    debit_value = input_col2.number_input(
        "Debit Card Volume",
        min_value=0,
        value=int(X["DebitCard_Volume"].median()),
        step=1,
    )
    credit_value = input_col3.number_input(
        "Credit Card Volume",
        min_value=0,
        value=int(X["CreditCard_Volume"].median()),
        step=1,
    )

    if upi_value < 0 or debit_value < 0 or credit_value < 0:
        st.error("Please enter non-negative values for all volumes.")
    else:
        predict_col1, predict_col2 = st.columns([1, 2])
        predict_button = predict_col1.button("Predict Currency")

        if predict_button:
            user_input = np.array([[upi_value, debit_value, credit_value]])
            predicted_value = model.predict(user_input)[0]
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)

            result_col1, result_col2, result_col3 = st.columns(3)
            result_col1.metric("Predicted Currency", f"{predicted_value:.2f}")
            result_col2.metric("Model R² Score", f"{r2:.4f}")
            result_col3.metric("Mean Squared Error", f"{mse:.2f}")

            st.success(
                "If digital payment volumes increase, currency circulation is expected to rise proportionally based on historical trends. "
                "This model provides an estimate of that relationship."
            )

            forecast_df = pd.DataFrame(
                {
                    "UPI_Volume": [upi_value],
                    "DebitCard_Volume": [debit_value],
                    "CreditCard_Volume": [credit_value],
                    "Predicted_Currency_in_Circulation": [predicted_value],
                }
            )
            csv_data = forecast_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Prediction Results as CSV",
                data=csv_data,
                file_name="currency_prediction.csv",
                mime="text/csv",
            )
