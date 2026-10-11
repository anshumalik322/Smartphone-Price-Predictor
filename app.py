import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


# ============================================
# PAGE SETTINGS
# ============================================

st.set_page_config(
    page_title="Smartphone Price Predictor",
    page_icon="📱",
    layout="centered"
)


# ============================================
# TITLE
# ============================================

st.title("📱 SmartPhone Price Predictor")
st.write("Predict the estimated price of a smartphone using Machine Learning.")


# ============================================
# LOAD DATA
# ============================================

@st.cache_data
def load_data():
    return pd.read_csv("smartphone_data_100_filled.csv")
    


df = load_data()


# ============================================
# PREPARE DATA
# ============================================

X = df[["Brand", "RAM", "Storage", "Age", "Condition"]]
y = df["Price"]

categorical_features = ["Brand", "Condition"]
numerical_features = ["RAM", "Storage", "Age"]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# ============================================
# MACHINE LEARNING MODEL
# ============================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=200,
                random_state=42
            )
        )
    ]
)


# Train model
model.fit(X, y)


# ============================================
# USER INPUT
# ============================================

st.subheader("Enter Smartphone Details")


brand = st.selectbox(
    "📱 Brand",
    sorted(df["Brand"].unique())
)


ram = st.number_input(
    "🧠 RAM (GB)",
    min_value=1,
    max_value=64,
    value=8,
    step=1
)


storage = st.number_input(
    "💾 Storage (GB)",
    min_value=16,
    max_value=2048,
    value=256,
    step=16
)


age = st.number_input(
    "📅 Age (years)",
    min_value=0,
    max_value=10,
    value=1,
    step=1
)


condition = st.selectbox(
    "⭐ Condition",
    sorted(df["Condition"].unique())
)


# ============================================
# PREDICTION
# ============================================

if st.button("🔮 Predict Price"):

    new_phone = pd.DataFrame({
        "Brand": [brand],
        "RAM": [ram],
        "Storage": [storage],
        "Age": [age],
        "Condition": [condition]
    })

    prediction = model.predict(new_phone)

    predicted_price = prediction[0]

    st.success(
        f"💰 Estimated Smartphone Price: ₹{predicted_price:,.2f}"
    )


# ============================================
# DATASET INFORMATION
# ============================================

with st.expander("📊 View Dataset"):

    st.dataframe(df)
