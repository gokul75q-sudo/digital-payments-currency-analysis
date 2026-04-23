import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Title
st.title("Impact of Digital Payments on Currency in Circulation")

# Load Excel dataset
file_path = "ABA FINAL PROJECT.xlsx"

df = pd.read_excel(file_path)

st.subheader("Dataset Preview")
st.write(df.head())

# Correlation Matrix
st.subheader("Correlation Matrix")

fig1, ax1 = plt.subplots()
sns.heatmap(df.corr(numeric_only=True), annot=True, ax=ax1)
st.pyplot(fig1)

# Define Variables
X = df[['UPI_Volume', 'DebitCard_Volume', 'CreditCard_Volume']]
y = df['Currency_in_Circulation']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

st.success("Model Training Completed")

# Predictions
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

st.subheader("Model Performance")

st.write("R2 Score:", r2)
st.write("Mean Squared Error:", mse)

# Coefficients
coef_df = pd.DataFrame({
    "Variable": X.columns,
    "Coefficient": model.coef_
})

st.subheader("Model Coefficients")
st.write(coef_df)

# Plot Actual vs Predicted
st.subheader("Actual vs Predicted")

fig2, ax2 = plt.subplots()
ax2.scatter(y_test, y_pred)
ax2.set_xlabel("Actual")
ax2.set_ylabel("Predicted")
ax2.set_title("Actual vs Predicted")

st.pyplot(fig2)

# Save Predictions
df["Predicted_CIC"] = model.predict(X)

df.to_excel("Predicted_Output.xlsx", index=False)

st.success("Predicted_Output.xlsx saved successfully")