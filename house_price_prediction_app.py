import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="House Price Prediction App",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("House Price Prediction App")

# ---------------------------------------------------------
# Data Loading
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "house_price_regression_dataset.csv"

if not DATASET_PATH.exists():
    st.error(
        "Dataset file not found. Please keep "
        "'house_price_regression_dataset.csv' in the same folder "
        "as this Python file."
    )
    st.stop()


@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

df = load_data(DATASET_PATH)

# Target and feature columns from the inspected dataset
target_column = "House_Price"
feature_columns = [
    "Square_Footage",
    "Num_Bedrooms",
    "Num_Bathrooms",
    "Year_Built",
    "Lot_Size",
    "Garage_Size",
    "Neighborhood_Quality",
]

# Validate required columns before continuing
required_columns = feature_columns + [target_column]
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error(f"Required columns are missing from the dataset: {missing_columns}")
    st.stop()

# Keep the project focused on the known target and input features
project_df = df[required_columns].copy()

num_col_eda = project_df.select_dtypes(include=np.number).columns.tolist()
cat_col_eda = project_df.select_dtypes(exclude=np.number).columns.tolist()

# ---------------------------------------------------------
# Sidebar: Navigation ONLY
# ---------------------------------------------------------
st.sidebar.header("Navigation")

if "active_section" not in st.session_state:
    st.session_state.active_section = "Data Summary"

if st.sidebar.button("Show Data Summary", use_container_width=True):
    st.session_state.active_section = "Data Summary"

if st.sidebar.button("EDA - Exploratory Data Analysis", use_container_width=True):
    st.session_state.active_section = "EDA"

if st.sidebar.button("Show Train-Test Split", use_container_width=True):
    st.session_state.active_section = "Train-Test Split"

if st.sidebar.button("Show Model Selection", use_container_width=True):
    st.session_state.active_section = "Model Selection"

if st.sidebar.button("Prediction", use_container_width=True):
    st.session_state.active_section = "Prediction"

active_section = st.session_state.active_section

# ---------------------------------------------------------
# Model List and Session State
# ---------------------------------------------------------
model_list = [
    "Decision Tree Regressor",
    "Random Forest Regressor",
    "Gradient Boosting Regressor",
    "Linear Regression",
    "Ridge Regression",
    "Lasso Regression",
    "K-Nearest Neighbors Regressor",
    "AdaBoost Regressor",
    "Support Vector Regressor",
    "Polynomial Regression",
]

if "selected_model" not in st.session_state:
    st.session_state.selected_model = model_list[0]

if "trained_model" not in st.session_state:
    st.session_state.trained_model = None

if "trained_model_name" not in st.session_state:
    st.session_state.trained_model_name = None

# ---------------------------------------------------------
# Main Screen: Section-specific Settings
# ---------------------------------------------------------
test_size = st.session_state.get("test_size", 0.2)
random_state = st.session_state.get("random_state", 3)
model_choice = st.session_state.selected_model
degree = st.session_state.get("polynomial_degree", 2)

analysis_type = None
col_name = None
col1_name = None
col2_name = None
multi_columns = None

if active_section == "Train-Test Split":
    st.header("Train-Test Split Settings")

    test_size = st.slider(
        "Select Test Size",
        min_value=0.1,
        max_value=0.9,
        value=float(test_size),
        step=0.1,
        key="test_size",
    )

    random_state = st.number_input(
        "Select Random State",
        min_value=0,
        max_value=100,
        value=int(random_state),
        step=1,
        key="random_state",
    )

elif active_section == "EDA":
    st.header("EDA - Exploratory Data Analysis")

    analysis_type = st.selectbox(
        "Select Analysis Type",
        [
            "Univariate Analysis",
            "Bivariate Analysis",
            "Multivariate Analysis",
        ],
        key="eda_analysis_type",
    )

    if analysis_type == "Univariate Analysis":
        col_name = st.selectbox(
            "Select the column for Univariate Analysis",
            project_df.columns.tolist(),
            key="eda_univariate_column",
        )

    elif analysis_type == "Bivariate Analysis":
        col1_name = st.selectbox(
            "Select the X-axis column",
            feature_columns,
            key="eda_bivariate_x",
        )
        col2_name = st.selectbox(
            "Select the Y-axis column",
            [col for col in project_df.columns if col != col1_name],
            key="eda_bivariate_y",
        )

    elif analysis_type == "Multivariate Analysis":
        multi_columns = st.multiselect(
            "Select columns for Multivariate Analysis",
            project_df.columns.tolist(),
            default=feature_columns[:3] + [target_column],
            key="eda_multivariate_columns",
        )

elif active_section == "Model Selection":
    st.header("Model Selection")

    model_choice = st.selectbox(
        "Select a model for training",
        model_list,
        index=model_list.index(st.session_state.selected_model),
        key="model_choice",
    )
    st.session_state.selected_model = model_choice

    if model_choice == "Polynomial Regression":
        degree = st.slider(
            "Select the degree for Polynomial Regression",
            min_value=2,
            max_value=5,
            value=int(degree),
            step=1,
            key="polynomial_degree",
        )

# ---------------------------------------------------------
# Train-Test Split
# ---------------------------------------------------------
x = project_df[feature_columns]
y = project_df[target_column]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=float(test_size),
    random_state=int(random_state),
)

# ---------------------------------------------------------
# Model Creation
# All preprocessing is inside the Pipeline to avoid leakage.
# ---------------------------------------------------------
def create_model(selected_name, polynomial_degree=2):
    if selected_name == "Decision Tree Regressor":
        estimator = DecisionTreeRegressor(random_state=3)
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", estimator),
        ])

    elif selected_name == "Random Forest Regressor":
        estimator = RandomForestRegressor(
            n_estimators=100,
            random_state=3,
        )
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", estimator),
        ])

    elif selected_name == "Gradient Boosting Regressor":
        estimator = GradientBoostingRegressor(random_state=3)
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", estimator),
        ])

    elif selected_name == "Linear Regression":
        estimator = LinearRegression()
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", estimator),
        ])

    elif selected_name == "Ridge Regression":
        estimator = Ridge()
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", estimator),
        ])

    elif selected_name == "Lasso Regression":
        estimator = Lasso(max_iter=10000)
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", estimator),
        ])

    elif selected_name == "K-Nearest Neighbors Regressor":
        estimator = KNeighborsRegressor()
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", estimator),
        ])

    elif selected_name == "AdaBoost Regressor":
        estimator = AdaBoostRegressor(random_state=3)
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", estimator),
        ])

    elif selected_name == "Support Vector Regressor":
        estimator = SVR()
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", estimator),
        ])

    elif selected_name == "Polynomial Regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("polynomial", PolynomialFeatures(degree=polynomial_degree)),
            ("model", LinearRegression()),
        ])

    else:
        raise ValueError("Please select a valid regression model.")


# Fit the currently selected model. The same pipeline is used for prediction.
current_model = create_model(model_choice, int(degree))
current_model.fit(x_train, y_train)

# Store the trained model so Prediction can use the same selected model.
st.session_state.trained_model = current_model
st.session_state.trained_model_name = model_choice

# Prepare model download (model + feature order + target name)
model_package = {
    "model": current_model,
    "feature_columns": feature_columns,
    "target_column": target_column,
    "model_name": model_choice,
}
model_bytes = pickle.dumps(model_package)

# ---------------------------------------------------------
# Main Screen: Section Output
# ---------------------------------------------------------

# 1. Data Summary
if active_section == "Data Summary":
    st.header("Data Summary")

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Total Records", f"{project_df.shape[0]:,}")
    metric2.metric("Total Columns", f"{project_df.shape[1]}")
    metric3.metric("Missing Values", f"{project_df.isnull().sum().sum()}")

    st.subheader("Dataset Records")
    st.dataframe(project_df, use_container_width=True)

    st.subheader("Dataset Information")
    summary_df = pd.DataFrame({
        "Column": project_df.columns,
        "Data Type": [str(project_df[col].dtype) for col in project_df.columns],
        "Non-Null Count": [project_df[col].notna().sum() for col in project_df.columns],
        "Missing Values": [project_df[col].isna().sum() for col in project_df.columns],
        "Unique Values": [project_df[col].nunique() for col in project_df.columns],
    })
    st.dataframe(summary_df, use_container_width=True)

    st.subheader("Statistical Summary")
    st.dataframe(project_df.describe().T, use_container_width=True)

# 2. EDA
elif active_section == "EDA":
    col1, col2 = st.columns(2)

    if analysis_type == "Univariate Analysis":
        if col_name in num_col_eda:
            chart1 = px.histogram(
                project_df,
                x=col_name,
                title=f"Distribution of {col_name}",
                marginal="rug",
            )
            col1.plotly_chart(chart1, use_container_width=True)

            chart2 = px.box(
                project_df,
                y=col_name,
                title=f"Box Plot of {col_name}",
            )
            col2.plotly_chart(chart2, use_container_width=True)

            chart3 = px.violin(
                project_df,
                y=col_name,
                box=True,
                title=f"Violin Plot of {col_name}",
            )
            col1.plotly_chart(chart3, use_container_width=True)

        elif col_name in cat_col_eda:
            category_counts = (
                project_df[col_name]
                .value_counts()
                .reset_index()
            )
            category_counts.columns = [col_name, "Count"]

            chart1 = px.bar(
                category_counts,
                x=col_name,
                y="Count",
                title=f"Count by {col_name}",
            )
            col1.plotly_chart(chart1, use_container_width=True)

            chart2 = px.pie(
                category_counts,
                names=col_name,
                values="Count",
                title=f"Distribution by {col_name}",
            )
            col2.plotly_chart(chart2, use_container_width=True)

    elif analysis_type == "Bivariate Analysis":
        if col1_name in project_df.columns and col2_name in project_df.columns:
            chart1 = px.scatter(
                project_df,
                x=col1_name,
                y=col2_name,
                trendline="ols",
                title=f"{col1_name} vs {col2_name}",
            )
            col1.plotly_chart(chart1, use_container_width=True)

            chart2 = px.density_heatmap(
                project_df,
                x=col1_name,
                y=col2_name,
                title=f"Density Heatmap: {col1_name} vs {col2_name}",
            )
            col2.plotly_chart(chart2, use_container_width=True)

            if project_df[col1_name].nunique() <= 20:
                grouped_df = (
                    project_df.groupby(col1_name, as_index=False)[col2_name]
                    .mean()
                )
                chart3 = px.bar(
                    grouped_df,
                    x=col1_name,
                    y=col2_name,
                    title=f"Average {col2_name} by {col1_name}",
                )
                st.plotly_chart(chart3, use_container_width=True)

    elif analysis_type == "Multivariate Analysis":
        if multi_columns and len(multi_columns) >= 2:
            selected_data = project_df[multi_columns]

            correlation = selected_data.corr(numeric_only=True)
            chart1 = px.imshow(
                correlation,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                title="Correlation Heatmap",
                aspect="auto",
            )
            col1.plotly_chart(chart1, use_container_width=True)

            if len(multi_columns) >= 2:
                chart2 = px.scatter_matrix(
                    selected_data,
                    dimensions=multi_columns,
                    title="Multivariate Scatter Matrix",
                )
                col2.plotly_chart(chart2, use_container_width=True)
        else:
            st.info("Please select at least two columns for Multivariate Analysis.")

# 3. Train-Test Split Summary
elif active_section == "Train-Test Split":
    st.header("Train-Test Split Summary")

    st.write(f"X Train Shape: {x_train.shape}")
    st.write(f"X Test Shape: {x_test.shape}")
    st.write(f"Y Train Shape: {y_train.shape}")
    st.write(f"Y Test Shape: {y_test.shape}")

# 4. Model Selection
elif active_section == "Model Selection":
    st.write(f"Selected Model: **{model_choice}**")

    st.subheader("Model Performance")
    y_pred = current_model.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metric1, metric2 = st.columns(2)
    metric1.metric("Mean Squared Error (MSE)", f"{mse:,.2f}")
    metric2.metric("Root Mean Squared Error (RMSE)", f"{rmse:,.2f}")

    metric3, metric4 = st.columns(2)
    metric3.metric("Mean Absolute Error (MAE)", f"{mae:,.2f}")
    metric4.metric("R-squared (R²)", f"{r2:.4f}")

    st.download_button(
        label="Download Model",
        data=model_bytes,
        file_name="house_price_model.pkl",
        mime="application/octet-stream",
    )

# 5. Prediction
elif active_section == "Prediction":
    st.header("House Price Prediction")
    st.write("Enter House Details to Predict the Price")

    st.download_button(
        label="Download Model",
        data=model_bytes,
        file_name="house_price_model.pkl",
        mime="application/octet-stream",
    )

    with st.form("house_price_prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            square_footage = st.number_input(
                "Square Footage",
                min_value=1,
                value=int(project_df["Square_Footage"].median()),
                step=10,
            )
            num_bedrooms = st.number_input(
                "Number of Bedrooms",
                min_value=0,
                value=int(project_df["Num_Bedrooms"].median()),
                step=1,
            )
            num_bathrooms = st.number_input(
                "Number of Bathrooms",
                min_value=0,
                value=int(project_df["Num_Bathrooms"].median()),
                step=1,
            )

        with col2:
            year_built = st.number_input(
                "Year Built",
                min_value=1800,
                max_value=2100,
                value=int(project_df["Year_Built"].median()),
                step=1,
            )
            lot_size = st.number_input(
                "Lot Size",
                min_value=0.0,
                value=float(project_df["Lot_Size"].median()),
                step=0.1,
                format="%.2f",
            )

        with col3:
            garage_size = st.number_input(
                "Garage Size",
                min_value=0,
                value=int(project_df["Garage_Size"].median()),
                step=1,
            )
            neighborhood_quality = st.number_input(
                "Neighborhood Quality",
                min_value=1,
                max_value=10,
                value=int(project_df["Neighborhood_Quality"].median()),
                step=1,
            )

        predict_button = st.form_submit_button("Predict House Price")

    if predict_button:
        input_data = pd.DataFrame([{
            "Square_Footage": square_footage,
            "Num_Bedrooms": num_bedrooms,
            "Num_Bathrooms": num_bathrooms,
            "Year_Built": year_built,
            "Lot_Size": lot_size,
            "Garage_Size": garage_size,
            "Neighborhood_Quality": neighborhood_quality,
        }], columns=feature_columns)

        prediction = current_model.predict(input_data)

        st.success("Prediction Completed Successfully!")
        st.metric(
            "Predicted House Price",
            f"{prediction[0]:,.2f}",
        )
