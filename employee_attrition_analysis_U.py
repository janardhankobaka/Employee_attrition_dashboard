"""
Employee Attrition Dashboard using Streamlit
Author: Bava
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import joblib
import logging
import os
import time

# ---------------------------------------------
# ✅ SET UP LOGGING
# ---------------------------------------------
logging.basicConfig(
    filename="employee_attrition_debug.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.debug("Starting Employee Attrition Dashboard application.")

# ---------------------------------------------
# ✅ PAGE CONFIGURATION
# ---------------------------------------------
try:
    st.set_page_config(page_title="Employee Attrition Dashboard", layout="wide")
    logging.debug("Page config set successfully.")
except Exception as e:
    logging.error(f"Error setting page config: {str(e)}")
    st.error(f"Error setting page config: {str(e)}")

# ---------------------------------------------
# ✅ DATA LOADING FUNCTION
# ---------------------------------------------
@st.cache_data(hash_funcs={pd.DataFrame: lambda _: time.time()})
def load_data(_=time.time()):
    try:
        local_path = "HR-Employee-Attrition.csv"
        if os.path.exists(local_path):
            df = pd.read_csv(local_path)
            logging.debug("Loaded local dataset.")
        else:
            dataset_url = "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv" 
            df = pd.read_csv(dataset_url)
            logging.debug(f"Loaded dataset from {dataset_url}.")
        if 'Attrition' in df.columns:
            df['AttritionFlag'] = df['Attrition'].map({'Yes': 1, 'No': 0})
        return df
    except Exception as e:
        logging.error(f"Error loading dataset: {str(e)}")
        return None

@st.cache_data(hash_funcs={pd.DataFrame: lambda _: time.time()})
def load_uploaded_data(file):
    try:
        df = pd.read_csv(file)
        if 'Attrition' in df.columns:
            df['AttritionFlag'] = df['Attrition'].map({'Yes': 1, 'No': 0})
        logging.debug("Uploaded dataset loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Error loading uploaded dataset: {str(e)}")
        return None

# Load data with fallback
try:
    with st.sidebar:
        st.subheader("📂 Data Upload")
        uploaded_file = st.file_uploader("Upload CSV (e.g., HR-Employee-Attrition.csv)", type="csv")
        if uploaded_file is not None:
            with st.spinner("Loading your data..."):
                df = load_uploaded_data(uploaded_file)
            if df is not None and not df.empty:
                st.success("✅ Uploaded file loaded!")
            else:
                st.warning("Failed to load uploaded dataset. Using default dataset.")
                df = load_data()
        else:
            df = load_data()

    required_cols = {'Age', 'JobRole', 'DistanceFromHome', 'Attrition', 'AttritionFlag'}
    if df is None or df.empty or not required_cols.issubset(df.columns):
        st.error("Invalid dataset. Ensure it has 'Age', 'JobRole', 'DistanceFromHome', 'Attrition' columns.")
        df = pd.DataFrame({'Age': [], 'JobRole': [], 'DistanceFromHome': [], 'Attrition': [], 'AttritionFlag': []})
        logging.warning("Using empty DataFrame as fallback.")
except Exception as e:
    logging.error(f"Error in data loading: {str(e)}")
    st.error(f"Data loading error: {str(e)}.")
    df = pd.DataFrame({'Age': [], 'JobRole': [], 'DistanceFromHome': [], 'Attrition': [], 'AttritionFlag': []})

# ---------------------------------------------
# ✅ SESSION STATE FOR THEME
# ---------------------------------------------
if 'theme_option' not in st.session_state or st.session_state.theme_option not in ["Light ☀️", "Dark 🌙"]:
    st.session_state.theme_option = "Light ☀️"
    logging.debug("Theme initialized/reset to Light mode.")

# ---------------------------------------------
# ✅ SIDEBAR OPTIONS
# ---------------------------------------------
try:
    with st.sidebar:
        st.subheader("🎨 Theme & Options")
        def toggle_theme():
            st.session_state.theme_option = "Dark 🌙" if st.session_state.theme_option == "Light ☀️" else "Light ☀️"
            logging.debug(f"Theme switched to {st.session_state.theme_option}.")
        st.button(f"Switch to {'Dark 🌙' if st.session_state.theme_option == 'Light ☀️' else 'Light ☀️'}", on_click=toggle_theme)

        chart_color = st.color_picker("Choose Chart Color", "#636EFA")
        st.markdown("---")
        st.subheader("🔎 Filter")
        if 'JobRole' in df.columns and not df.empty:
            job_role_options = ["All"] + sorted(list(df['JobRole'].unique()))
            selected_job_role = st.selectbox("Filter by Job Role", job_role_options)
            if selected_job_role != "All":
                df = df[df['JobRole'] == selected_job_role]
                logging.debug(f"Filtered by Job Role: {selected_job_role}")
        if st.button("Clear Cache"):
            st.cache_data.clear()
            st.success("Cache cleared! Refresh to update.")
            logging.debug("Cache cleared by user.")
except Exception as e:
    logging.error(f"Error in sidebar: {str(e)}")
    st.error(f"Sidebar error: {str(e)}")

# ---------------------------------------------
# ✅ CUSTOM CSS BASED ON THEME
# ---------------------------------------------
light_css = """
<style>
.stApp { background-color: #f5f7fa; }
section[data-testid="stSidebar"] { background: linear-gradient(to bottom, #e0f2fe, #bfdbfe); }
h1, h2, h3, .stMarkdown { color: #0f172a !important; }
.stButton > button { background-color: #3b82f6; color: white; border-radius: 8px; }
.stButton > button:hover { background-color: #2563eb; }
div[role="tablist"] button { color: #0f172a !important; font-weight: bold; }
div[role="tablist"] button:hover { color: #2563eb !important; }
.plotly .modebar { fill: #0f172a !important; }
</style>
"""

dark_css = """
<style>
.stApp { background-color: #111827; }
section[data-testid="stSidebar"] { background: linear-gradient(to bottom, #1f2937, #374151); }
h1, h2, h3, .stMarkdown { color: #f3f4f6 !important; }
.stButton > button { background-color: #3b82f6; color: white; border-radius: 8px; }
.stButton > button:hover { background-color: #2563eb; }
div[role="tablist"] button { color: #f3f4f6 !important; font-weight: bold; }
div[role="tablist"] button:hover { color: #60a5fa !important; }
.plotly .modebar { fill: #f3f4f6 !important; }
.plotly .hoverlayer text { fill: #ffffff !important; }
</style>
"""

st.markdown(light_css if st.session_state.theme_option == "Light ☀️" else dark_css, unsafe_allow_html=True)
st.sidebar.markdown(f"**Current Theme:** {st.session_state.theme_option}")

# ---------------------------------------------
# ✅ HEADER
# ---------------------------------------------
st.markdown("""
<div style='padding: 1rem; background: linear-gradient(to right, #3b82f6, #06b6d4); color: white; border-radius: 8px;'>
    <h1 style='margin: 0;'>👩‍💼 Employee Attrition Dashboard</h1>
</div>
<br>
""", unsafe_allow_html=True)

# ---------------------------------------------
# ✅ TABS
# ---------------------------------------------
try:
    tab1, tab2 = st.tabs(["Before Prediction", "After Prediction"])
    logging.debug("Tabs created successfully.")
except Exception as e:
    logging.error(f"Error creating tabs: {str(e)}")
    st.error(f"Tab creation error: {str(e)}")
    st.stop()

# ---------------------------------------------
# ✅ BEFORE PREDICTION TAB
# ---------------------------------------------
with tab1:
    try:
        st.subheader("📊 Attrition Analysis")
        if 'Attrition' in df.columns and not df.empty:
            col1, col2 = st.columns(2)
            with col1:
                fig1 = px.histogram(df, x='Age', color='Attrition', barmode='overlay',
                                   title="Attrition by Age",
                                   color_discrete_sequence=[chart_color, '#D97706'])
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                fig2 = px.histogram(df, x='JobRole', color='Attrition',
                                   title="Attrition by Job Role",
                                   color_discrete_sequence=[chart_color, '#D97706'])
                st.plotly_chart(fig2, use_container_width=True)

            distance_bins = pd.cut(df['DistanceFromHome'], bins=[0, 5, 10, 15, 20, 25, 30], include_lowest=True)
            df['DistanceBin'] = distance_bins.astype(str)
            distance_attrition = df.groupby('DistanceBin', observed=True)['AttritionFlag'].mean().reset_index()
            fig3 = px.bar(distance_attrition, x='DistanceBin', y='AttritionFlag',
                         title="Attrition Rate by Distance from Home",
                         labels={'AttritionFlag': 'Attrition Rate'},
                         color_discrete_sequence=[chart_color],
                         text='AttritionFlag')
            fig3.update_traces(texttemplate='%{text:.2f}', textposition='auto')
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Upload a dataset with 'Attrition', 'Age', 'JobRole', 'DistanceFromHome' columns to view charts.")
    except Exception as e:
        logging.error(f"Error in Before Prediction tab: {str(e)}")
        st.error(f"Error in Before Prediction tab: {str(e)}")

# ---------------------------------------------
# ✅ AFTER PREDICTION TAB
# ---------------------------------------------
with tab2:
    try:
        st.subheader("🧠 Attrition Prediction")
        if df is None or df.empty or 'Attrition' not in df.columns:
            st.error("Invalid dataset for training. Upload a valid CSV with 'Attrition', 'Age', 'JobRole', 'DistanceFromHome'.")
            logging.error("Invalid dataset for prediction.")
        else:
            # Encode categorical columns
            df_encoded = df.copy()
            le_dict = {}
            for col in df_encoded.select_dtypes(include='object'):
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                le_dict[col] = le

            # Scale numerical columns
            scaler = StandardScaler()
            numerical_cols = df_encoded.select_dtypes(include=['int64', 'float64']).columns
            numerical_cols = [col for col in numerical_cols if col not in ['AttritionFlag']]
            df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])

            # Prepare data
            X = df_encoded.drop(['Attrition', 'AttritionFlag'], axis=1, errors='ignore')
            y = df_encoded['AttritionFlag']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Handle class imbalance
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

            # Define models with tuned hyperparameters
            models = {
                "XGBoost": XGBClassifier(
                    use_label_encoder=False, eval_metric='logloss', random_state=42,
                    max_depth=5, learning_rate=0.1, n_estimators=200, scale_pos_weight=1
                ),
                "RandomForest": RandomForestClassifier(
                    random_state=42, max_depth=10, n_estimators=200, min_samples_split=5, class_weight='balanced'
                ),
                "LightGBM": LGBMClassifier(
                    random_state=42, max_depth=5, learning_rate=0.1, n_estimators=200, class_weight='balanced'
                )
            }

            # Compare models and display all accuracies
            st.markdown("### Model Comparison")
            model_accuracies = {}
            report_data = []
            for model_name, model in models.items():
                model.fit(X_train_resampled, y_train_resampled)
                y_pred_test = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred_test)
                model_accuracies[model_name] = (model, accuracy)
                report = classification_report(y_test, y_pred_test, output_dict=True)
                report_data.append({
                    'Model': model_name,
                    'Accuracy': f"{accuracy:.2f}",
                    'Precision (No Attrition)': f"{report['0']['precision']:.2f}",
                    'Recall (No Attrition)': f"{report['0']['recall']:.2f}",
                    'F1-Score (No Attrition)': f"{report['0']['f1-score']:.2f}",
                    'Precision (Attrition)': f"{report['1']['precision']:.2f}",
                    'Recall (Attrition)': f"{report['1']['recall']:.2f}",
                    'F1-Score (Attrition)': f"{report['1']['f1-score']:.2f}"
                })
                st.write(f"**{model_name} Accuracy**: {accuracy:.2f}")

            # Select best model
            best_model_name = max(model_accuracies, key=lambda k: model_accuracies[k][1])
            best_model, best_accuracy = model_accuracies[best_model_name]
            st.markdown(f"### Best Model: {best_model_name}")
            st.write(f"**Accuracy**: {best_accuracy:.2f}")
            y_pred_test = best_model.predict(X_test)
            st.text(f"Classification Report:\n{classification_report(y_test, y_pred_test)}")

            # Save best model
            joblib.dump(best_model, f"best_model_{best_model_name.lower()}.pkl")
            logging.debug(f"Saved best model: {best_model_name} with accuracy {best_accuracy:.2f}")
            st.info(f"Best model ({best_model_name}, Accuracy: {best_accuracy:.2f}) saved as 'best_model_{best_model_name.lower()}.pkl'.")

            # Downloadable report
            report_df = pd.DataFrame(report_data)
            report_csv = report_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Model Performance Report",
                data=report_csv,
                file_name="model_performance_report.csv",
                mime="text/csv"
            )

            # Predict on input data
            X_input = df_encoded[X_train.columns]
            y_pred = best_model.predict(X_input)
            df['PredictedAttrition'] = y_pred
            st.success(f"✅ Predictions made with {best_model_name} (Accuracy: {best_accuracy:.2f})!")
            st.dataframe(df[['Age', 'JobRole', 'DistanceFromHome', 'PredictedAttrition']].head())

            pred_count = df['PredictedAttrition'].value_counts().reset_index()
            pred_count.columns = ['Attrition (Predicted)', 'Count']
            fig4 = px.bar(pred_count, x='Attrition (Predicted)', y='Count',
                         title=f"Predicted Attrition Distribution ({best_model_name})",
                         text='Count', color_discrete_sequence=[chart_color])
            fig4.update_traces(texttemplate='%{text}', textposition='auto')
            fig4.update_layout(
                xaxis=dict(tickvals=[0, 1], ticktext=['No Attrition', 'Attrition']),
                hoverlabel=dict(bgcolor='#ffffff')
            )
            st.plotly_chart(fig4, use_container_width=True)

            st.write("**Predicted Attrition Counts:**")
            counts_df = pred_count.set_index('Attrition (Predicted)')
            counts_df.index = counts_df.index.map({0: 'No Attrition', 1: 'Attrition'})
            st.write(counts_df)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"📥 Download {best_model_name} Predictions",
                data=csv,
                file_name=f"predictions_{best_model_name.lower()}.csv",
                mime="text/csv"
            )
    except Exception as e:
        logging.error(f"Error in After Prediction tab: {str(e)}")
        st.error(f"Error in After Prediction tab: {str(e)}")

# ---------------------------------------------
# ✅ FOOTER
# ---------------------------------------------
try:
    st.markdown("""
        <hr style='border: 1px solid #3b82f6;'>
        <p style='text-align: center; color: #9ca3af;'>Made with ❤️ by Bava</p>
    """, unsafe_allow_html=True)
    logging.debug("Footer rendered successfully.")
except Exception as e:
    logging.error(f"Error rendering footer: {str(e)}")
    st.error(f"Footer error: {str(e)}")