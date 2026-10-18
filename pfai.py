import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import MinMaxScaler

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Industrial Energy Consumption Analysis",
    layout="wide"
)

st.title("⚡ Industrial Energy Consumption Analysis & Prediction")

st.markdown("---")

# ==================================================
# FILE UPLOAD
# ==================================================

uploaded_file = st.file_uploader(
    "Upload Energy Dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("Dataset Loaded Successfully!")

    # ==================================================
    # DATASET OVERVIEW
    # ==================================================

    st.header("📊 Dataset Overview")

    st.dataframe(df)

    st.write("Dataset Shape:", df.shape)

    # ==================================================
    # DATA TYPES
    # ==================================================

    st.header("📝 Data Types")

    st.write(df.dtypes)

    # ==================================================
    # MISSING VALUES
    # ==================================================

    st.header("🔍 Missing Values")

    st.write(df.isnull().sum())

    # ==================================================
    # DUPLICATE VALUES
    # ==================================================

    st.header("📌 Duplicate Values")

    st.write("Total Duplicates:", df.duplicated().sum())

    # ==================================================
    # PREPROCESSING
    # ==================================================

    st.header("🧹 Data Preprocessing")

    clean_df = df.drop_duplicates()

    st.write("Shape After Removing Duplicates:")

    st.write(clean_df.shape)

    # ==================================================
    # NORMALIZATION
    # ==================================================

    st.header("⚙️ Normalization")

    numeric_cols = clean_df.select_dtypes(
        include='number'
    ).columns

    scaler = MinMaxScaler()

    normalized_df = clean_df.copy()

    normalized_df[numeric_cols] = scaler.fit_transform(
        clean_df[numeric_cols]
    )

    st.write("Normalized Dataset")

    st.dataframe(normalized_df.head())

    # ==================================================
    # DISTRIBUTION ANALYSIS
    # ==================================================

    st.header("📈 Distribution Analysis")

    selected_feature = st.selectbox(
        "Select Feature",
        numeric_cols
    )

    fig, ax = plt.subplots(figsize=(10,5))

    sns.histplot(
        clean_df[selected_feature],
        kde=True,
        color="orange",
        edgecolor="black"
    )

    plt.title(
        f"Distribution of {selected_feature}"
    )

    plt.grid(True)

    st.pyplot(fig)

    # ==================================================
    # CORRELATION HEATMAP
    # ==================================================

    st.header("🔥 Correlation Heatmap")

    corr = clean_df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(12,8))

    sns.heatmap(
        corr,
        annot=True,
        cmap="RdYlGn",
        linewidths=0.5,
        fmt=".2f"
    )

    plt.title("Correlation Heatmap")

    st.pyplot(fig)

    # ==================================================
    # FEATURE SELECTION
    # ==================================================

    st.header("🎯 Feature Selection")

    target = "Industrial Energy Use (%)"

    if target in corr.columns:

        corr_target = corr[target].drop(target)

        fig, ax = plt.subplots(figsize=(10,6))

        corr_target.sort_values().plot(
            kind="barh",
            color=[
                'red',
                'blue',
                'green',
                'orange',
                'purple',
                'cyan',
                'brown',
                'pink'
            ],
            ax=ax
        )

        plt.title(
            "Feature Importance Based on Correlation"
        )

        st.pyplot(fig)

        st.write(corr_target.sort_values())

        # ==================================================
        # FEATURE DEPENDENCY ANALYSIS
        # ==================================================

        st.header("🔗 Feature Dependency Analysis")

        feature_x = st.selectbox(
            "Select Feature for Dependency Analysis",
            corr_target.index
        )

        fig, ax = plt.subplots(figsize=(10,6))

        sns.regplot(
            x=clean_df[feature_x],
            y=clean_df[target],
            scatter_kws={"color":"blue"},
            line_kws={"color":"red"}
        )

        plt.xlabel(feature_x)
        plt.ylabel(target)

        plt.title(
            f"{feature_x} vs {target}"
        )

        st.pyplot(fig)

    # ==================================================
    # FEATURE COMPARISON
    # ==================================================

    st.header("⚖️ Feature Comparison")

    feature1 = st.selectbox(
        "Feature 1",
        numeric_cols,
        key="feature1"
    )

    feature2 = st.selectbox(
        "Feature 2",
        numeric_cols,
        key="feature2"
    )

    fig, ax = plt.subplots(figsize=(10,6))

    sns.scatterplot(
        x=clean_df[feature1],
        y=clean_df[feature2],
        color="green"
    )

    plt.xlabel(feature1)
    plt.ylabel(feature2)

    plt.title(
        f"{feature1} vs {feature2}"
    )

    st.pyplot(fig)

    # ==================================================
    # PAIRPLOT ANALYSIS
    # ==================================================

    st.header("📉 Feature Relationship Analysis")

    selected_cols = [
        'Total Energy Consumption (TWh)',
        'Renewable Energy Share (%)',
        'Fossil Fuel Dependency (%)',
        'Carbon Emissions (Million Tons)',
        'Industrial Energy Use (%)'
    ]

    available_cols = [
        col for col in selected_cols
        if col in clean_df.columns
    ]

    if len(available_cols) > 1:

        pairplot = sns.pairplot(
            clean_df[available_cols]
        )

        st.pyplot(pairplot.fig)

    # ==================================================
    # PREDICTION SECTION
    # ==================================================
# ==================================================
# INDUSTRIAL ENERGY PREDICTION
# ==================================================

st.header("🤖 Industrial Energy Prediction")

try:

    model = joblib.load("fuel_model.pkl")

    col1, col2 = st.columns(2)

    with col1:

        year = st.number_input(
            "Year",
            min_value=2000,
            max_value=2050,
            value=2024
        )

        total_energy = st.number_input(
            "Total Energy Consumption (TWh)",
            min_value=0.0,
            value=5000.0
        )

    with col2:

        fossil = st.number_input(
            "Fossil Fuel Dependency (%)",
            min_value=0.0,
            max_value=100.0,
            value=50.0
        )

        renewable = st.number_input(
            "Renewable Energy Share (%)",
            min_value=0.0,
            max_value=100.0,
            value=30.0
        )

    if st.button("Predict Industrial Energy Use"):

        input_df = pd.DataFrame({
            'Year':[year],
            'Total Energy Consumption (TWh)':[total_energy],
            'Fossil Fuel Dependency (%)':[fossil],
            'Renewable Energy Share (%)':[renewable]
        })

        prediction = model.predict(input_df)

        st.success(
            f"Predicted Industrial Energy Use = {prediction[0]:.2f}%"
        )

        st.metric(
            label="Predicted Industrial Energy Use (%)",
            value=f"{prediction[0]:.2f}"
        )

except Exception as e:

    st.warning(
        "fuel_model.pkl not found in project folder."
    )

    st.write(e)
else:

    st.info("Please upload a CSV dataset to begin.")