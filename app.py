import streamlit as st
import pandas as pd
import numpy as np
import os
import glob

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="SmartPhone Price Predictor",
    page_icon="📱",
    layout="wide"
)


# ==========================================
# CUSTOM DESIGN
# ==========================================

st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}

h1 {
    color: #1769aa;
}

h2, h3 {
    color: #243b53;
}

.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 45px;
    background-color: #1769aa;
    color: white;
    font-weight: bold;
}

div[data-testid="stMetric"] {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# HEADER
# ==========================================

st.title("📱 SmartPhone Price Predictor")

st.markdown("""
Predict the estimated price of a smartphone using
Machine Learning.

Enter your smartphone specifications below to get
a price estimate based on your dataset.
""")

st.divider()


# ==========================================
# LOAD DATASET AUTOMATICALLY
# ==========================================

@st.cache_data
def load_data():
    """
    Find a CSV file in the project directory.
    """

    possible_files = [
        "smartphone_data.csv",
        "smartphone_data_100_filled.csv",
        "smartphone_data_100_filled (1).csv"
    ]

    for filename in possible_files:
        if os.path.isfile(filename):
            return pd.read_csv(filename), filename

    csv_files = glob.glob("*.csv")

    if len(csv_files) == 1:
        return pd.read_csv(csv_files[0]), csv_files[0]

    if len(csv_files) == 0:
        raise FileNotFoundError(
            "No CSV file found. Upload your smartphone dataset "
            "to the same GitHub repository as app.py."
        )

    raise FileNotFoundError(
        "Multiple CSV files found. Keep one intended dataset "
        "in the repository or update the filename list in app.py."
    )


try:
    df, dataset_name = load_data()

except Exception as error:
    st.error(f"Dataset loading failed: {error}")
    st.stop()


# ==========================================
# CLEAN DATASET
# ==========================================

df.columns = df.columns.astype(str).str.strip()

# Remove completely empty rows and columns.
df = df.dropna(how="all")
df = df.dropna(axis=1, how="all")

# Remove duplicate column names.
df = df.loc[:, ~df.columns.duplicated()]

if df.empty:
    st.error("Your CSV dataset is empty.")
    st.stop()


# ==========================================
# FIND THE PRICE COLUMN
# ==========================================

def find_price_column(columns):
    """
    Identify the target column using common price names.
    """

    normalized = {
        str(column).strip().lower().replace(" ", "_"): column
        for column in columns
    }

    preferred_names = [
        "price",
        "resale_price",
        "selling_price",
        "predicted_price",
        "estimated_price",
        "smartphone_price",
        "target_price"
    ]

    for name in preferred_names:
        if name in normalized:
            return normalized[name]

    # Look for another column containing 'price'.
    for column in columns:
        name = str(column).lower()

        if "price" in name:
            return column

    return None


price_column = find_price_column(df.columns)

if price_column is None:
    st.error(
        "Could not identify the target price column. "
        "Rename your dataset's target column to Price."
    )
    st.write("Available columns:", df.columns.tolist())
    st.stop()


# Convert the target to numeric values.
df[price_column] = pd.to_numeric(
    df[price_column]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.replace("₹", "", regex=False)
    .str.strip(),
    errors="coerce"
)

# Keep only valid, positive target prices.
df = df[
    df[price_column].notna()
    & (df[price_column] > 0)
].copy()

if len(df) < 5:
    st.error(
        "At least five valid smartphone price records are "
        "needed to train this demonstration model."
    )
    st.stop()


# ==========================================
# EXTRACT ALL AVAILABLE FEATURES
# ==========================================

# Every dataset column except the target price becomes
# a candidate input feature.

feature_columns = [
    column for column in df.columns
    if column != price_column
]

# Remove features that have no useful variation.
feature_columns = [
    column for column in feature_columns
    if df[column].nunique(dropna=True) > 1
]

if not feature_columns:
    st.error(
        "No usable smartphone specification columns were found."
    )
    st.stop()


# Convert columns that contain numeric values into numeric
# features when appropriate.

for column in feature_columns:
    if (
        pd.api.types.is_object_dtype(df[column])
        or pd.api.types.is_string_dtype(df[column])
    ):
        original = df[column].dropna().astype(str).str.strip()

        cleaned = (
            original
            .str.replace(",", "", regex=False)
            .str.replace("₹", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.replace(r"\s+", "", regex=True)
        )

        numeric_values = pd.to_numeric(
            cleaned,
            errors="coerce"
        )

        # Convert only if almost all non-empty values
        # can be interpreted as numbers.
        if (
            len(original) > 0
            and numeric_values.notna().mean() >= 0.95
        ):
            converted = pd.to_numeric(
                df[column]
                .astype(str)
                .str.strip()
                .str.replace(",", "", regex=False)
                .str.replace("₹", "", regex=False)
                .str.replace("%", "", regex=False)
                .str.replace(r"\s+", "", regex=True),
                errors="coerce"
            )

            df[column] = converted


# ==========================================
# PREPARE FEATURES AND TARGET
# ==========================================

X = df[feature_columns].copy()
y = df[price_column].copy()

numeric_columns = X.select_dtypes(
    include=["number", "bool"]
).columns.tolist()

categorical_columns = [
    column for column in X.columns
    if column not in numeric_columns
]


# ==========================================
# BUILD MACHINE LEARNING PIPELINE
# ==========================================

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    )
])

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    )
])

transformers = []

if numeric_columns:
    transformers.append((
        "numeric",
        numeric_pipeline,
        numeric_columns
    ))

if categorical_columns:
    transformers.append((
        "categorical",
        categorical_pipeline,
        categorical_columns
    ))

preprocessor = ColumnTransformer(
    transformers=transformers
)

model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "regressor",
        RandomForestRegressor(
            n_estimators=150,
            random_state=42,
            min_samples_leaf=1
        )
    )
])


# ==========================================
# TRAIN AND EVALUATE MODEL
# ==========================================

@st.cache_resource
def train_model(data, features, target):
    """
    Train the model and calculate a held-out MAE.
    """

    X_data = data[features].copy()
    y_data = data[target].copy()

    numeric = X_data.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical = [
        column for column in X_data.columns
        if column not in numeric
    ]

    transformers = []

    if numeric:
        transformers.append((
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ]),
            numeric
        ))

    if categorical:
        transformers.append((
            "categorical",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore")
                )
            ]),
            categorical
        ))

    preprocessing = ColumnTransformer(
        transformers=transformers
    )

    trained_model = Pipeline([
        ("preprocessor", preprocessing),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=150,
                random_state=42,
                min_samples_leaf=1
            )
        )
    ])

    evaluation_mae = None

    if len(X_data) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X_data,
            y_data,
            test_size=0.2,
            random_state=42
        )

        evaluation_model = Pipeline([
            ("preprocessor", preprocessing),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=150,
                    random_state=42,
                    min_samples_leaf=1
                )
            )
        ])

        evaluation_model.fit(X_train, y_train)

        predictions = evaluation_model.predict(X_test)

        evaluation_mae = mean_absolute_error(
            y_test,
            predictions
        )

    # Train the final model on all available records.
    trained_model.fit(X_data, y_data)

    return trained_model, evaluation_mae


try:
    model, evaluation_mae = train_model(
        df,
        feature_columns,
        price_column
    )

except Exception as error:
    st.error(f"Model training failed: {error}")
    st.stop()


# ==========================================
# DATASET INFORMATION
# ==========================================

with st.expander("📊 View Dataset Information"):
    col1, col2, col3 = st.columns(3)

    col1.metric("Smartphone Records", len(df))
    col2.metric("Specification Columns", len(feature_columns))
    col3.metric("Price Column", str(price_column))

    st.write("**Dataset file:**", dataset_name)

    st.write("**Detected specification columns:**")
    st.write(feature_columns)

    st.dataframe(
        df,
        use_container_width=True
    )

    if evaluation_mae is not None:
        st.metric(
            "Test Mean Absolute Error",
            f"₹{evaluation_mae:,.2f}"
        )

        st.caption(
            "This is the average absolute prediction error "
            "on a held-out test split. It is not a guarantee "
            "of accuracy on future smartphones."
        )


# ==========================================
# SMARTPHONE INPUT FORM
# ==========================================

st.header("📱 Enter Smartphone Details")

st.write(
    "The fields below are generated automatically "
    "from your CSV dataset."
)

user_values = {}

# Display inputs in two columns.
left_column, right_column = st.columns(2)

for index, column in enumerate(feature_columns):

    current_column = left_column if index % 2 == 0 else right_column

    series = df[column]
    non_null = series.dropna()

    with current_column:

        # Numeric specification.
        if (
            pd.api.types.is_numeric_dtype(series)
            or pd.api.types.is_bool_dtype(series)
        ):

            if non_null.empty:
                user_values[column] = 0
                st.number_input(
                    column,
                    value=0,
                    key=f"empty_{index}",
                    disabled=True
                )

            else:
                numeric_series = pd.to_numeric(
                    non_null,
                    errors="coerce"
                ).dropna()

                if numeric_series.empty:
                    user_values[column] = 0

                else:
                    minimum = float(numeric_series.min())
                    maximum = float(numeric_series.max())
                    default = float(numeric_series.median())

                    if minimum == maximum:
                        user_values[column] = minimum

                        st.number_input(
                            column,
                            value=minimum,
                            disabled=True,
                            key=f"fixed_{index}"
                        )

                    else:
                        # Use a number input so large storage,
                        # camera and battery values remain editable.
                        step = (
                            1.0
                            if numeric_series.nunique() > 30
                            else 1.0
                        )

                        user_values[column] = st.number_input(
                            column,
                            min_value=minimum,
                            max_value=maximum,
                            value=min(
                                max(default, minimum),
                                maximum
                            ),
                            step=step,
                            key=f"number_{index}"
                        )

        # Categorical specification.
        else:
            options = sorted(
                non_null.astype(str).unique().tolist()
            )

            if not options:
                options = ["Unknown"]

            user_values[column] = st.selectbox(
                column,
                options=options,
                key=f"select_{index}"
            )


# ==========================================
# PRICE PREDICTION
# ==========================================

st.divider()

if st.button("🚀 Predict Smartphone Price"):

    input_data = pd.DataFrame(
        [user_values],
        columns=feature_columns
    )

    # Restore original numeric feature types.
    for column in numeric_columns:
        input_data[column] = pd.to_numeric(
            input_data[column],
            errors="coerce"
        )

    try:
        predicted_price = float(
            model.predict(input_data)[0]
        )

        st.success("Prediction completed successfully!")

        st.subheader("💰 Estimated Smartphone Price")

        st.metric(
            label="Predicted Price",
            value=f"₹{predicted_price:,.2f}"
        )

        st.caption(
            "This is a machine-learning estimate based on "
            "your dataset, not a verified live market price."
        )

        with st.expander("🔍 Review Entered Specifications"):
            st.dataframe(
                input_data,
                use_container_width=True
            )

    except Exception as error:
        st.error(f"Prediction failed: {error}")


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "SmartPhone Price Predictor | "
    "Powered by Python, Pandas, Scikit-learn and Streamlit"
)
