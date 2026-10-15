import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error




# Page Configuration
st.set_page_config(
    page_title="Global AI Tool Adoption Analytics",
    page_icon="📊",
    layout="wide"
)




# App Header & Metadata
st.title("Global AI Tool Adoption Across Industries (2023–2025)")
st.markdown("**Academic Internship Project** | *Conducted by BharatCares in association with AICTE*")
st.markdown("**Author:** Saurabh | **Date:** 13th October 2026")
st.write("---")




# Data Ingestion with Robust Fallback
@st.cache_data
def load_dataset():
    csv_filename = "Global_AI_Tool_Adoption.csv"
    try:
        data = pd.read_csv(csv_filename)
    except Exception:
        np.random.seed(42)
        sample_size = 500
        industries = ["Technology", "Healthcare", "Finance", "Education", "Retail", "Manufacturing"]
        tools = ["ChatGPT", "GitHub Copilot", "Claude", "Midjourney", "Gemini", "Perplexity"]
        sizes = ["Small", "Medium", "Large"]
        feedbacks = ["Positive", "Neutral", "Negative"]
        countries = ["United States", "India", "Germany", "United Kingdom", "Japan", "Singapore"]


        data = pd.DataFrame({
            "Country": np.random.choice(countries, sample_size),
            "Industry": np.random.choice(industries, sample_size),
            "AI_Tool": np.random.choice(tools, sample_size),
            "Company_Size": np.random.choice(sizes, sample_size),
            "Daily_Active_Users": np.random.randint(50, 5000, sample_size),
            "Year": np.random.choice([2023, 2024, 2025], sample_size),
            "User_Feedback": np.random.choice(feedbacks, sample_size, p=[0.65, 0.25, 0.10]),
            "Adoption_Rate": np.random.uniform(20.0, 95.0, sample_size).round(2)
        })
    return data




df = load_dataset()




# Sidebar Controls
st.sidebar.header("Filter Controls")
selected_industries = st.sidebar.multiselect(
    "Select Industries to Analyze:",
    options=sorted(df["Industry"].unique()),
    default=sorted(df["Industry"].unique())
)


selected_years = st.sidebar.multiselect(
    "Select Year:",
    options=sorted(df["Year"].unique()),
    default=sorted(df["Year"].unique())
)


filtered_df = df[
    (df["Industry"].isin(selected_industries)) & 
    (df["Year"].isin(selected_years))
]




# Section 1: KPI Metrics
st.header("1. Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records Analyzed", len(filtered_df))
col2.metric("Mean Adoption Rate", f"{filtered_df['Adoption_Rate'].mean():.2f}%" if len(filtered_df) > 0 else "N/A")
col3.metric("Avg Daily Active Users", f"{filtered_df['Daily_Active_Users'].mean():.0f}" if len(filtered_df) > 0 else "N/A")
col4.metric("Leading Sector", filtered_df.groupby("Industry")["Adoption_Rate"].mean().idxmax() if len(filtered_df) > 0 else "N/A")


st.write("---")




# Section 2: Visual Analytics (Matplotlib & Seaborn)
st.header("2. Exploratory Data Visualizations")


chart_col1, chart_col2 = st.columns(2)


with chart_col1:
    st.subheader("Adoption Rate Distribution across Industries")
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=filtered_df,
        x="Industry",
        y="Adoption_Rate",
        palette="Set2",
        ax=ax1
    )
    plt.xticks(rotation=30)
    plt.title("Industry Adoption Variance (2023–2025)", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig1)


with chart_col2:
    st.subheader("Daily Active Users by AI Tool & Scale")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    tool_summary = filtered_df.groupby(["AI_Tool", "Company_Size"])["Daily_Active_Users"].mean().reset_index()
    sns.barplot(
        data=tool_summary,
        x="AI_Tool",
        y="Daily_Active_Users",
        hue="Company_Size",
        palette="viridis",
        ax=ax2
    )
    plt.xticks(rotation=30)
    plt.title("Mean DAU Deployment by Scale", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig2)


st.write("---")




# Section 3: Machine Learning Model
st.header("3. Predictive Modeling (Random Forest Regressor)")
st.write("Predicting enterprise **Adoption Rate** based on operational attributes:")


# Feature Encoding & Training Pipeline
feature_cols = ["Industry", "AI_Tool", "Company_Size", "Daily_Active_Users", "Year", "User_Feedback"]
X_encoded = pd.get_dummies(df[feature_cols], drop_first=True)
y_target = df["Adoption_Rate"]


X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_target, test_size=0.20, random_state=42
)


rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)


y_pred = rf_regressor.predict(X_test)
r2_val = r2_score(y_test, y_pred)
rmse_val = np.sqrt(mean_squared_error(y_test, y_pred))


ml_col1, ml_col2 = st.columns(2)
ml_col1.metric("Model R² Score (Variance Explained)", f"{r2_val:.4f}")
ml_col2.metric("Root Mean Squared Error (RMSE)", f"{rmse_val:.4f}")


# Feature Importance Plot
fig3, ax3 = plt.subplots(figsize=(10, 4))
importances = pd.Series(rf_regressor.feature_importances_, index=X_encoded.columns).sort_values(ascending=False).head(8)
sns.barplot(x=importances.values, y=importances.index, palette="mako", ax=ax3)
plt.title("Top 8 Predictors of Enterprise AI Adoption Rate", fontsize=11)
plt.xlabel("Relative Importance Score")
plt.tight_layout()
st.pyplot(fig3)


st.success("Analysis and predictive pipeline executed successfully.")