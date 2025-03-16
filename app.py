import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from lime.lime_text import LimeTextExplainer
from fairlearn.metrics import selection_rate

# 🎨 **UI Configuration**
st.set_page_config(page_title="NextGen AI Resume Analyser", layout="wide")
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>📄 NextGen AI Resume Analyser</h1>", unsafe_allow_html=True)
st.write("AI-powered tool for analysing resumes using **SHAP, LIME, and Fairness Audits**.")

# 📂 **Load the dataset**
try:
    csv_path = "NextGen.csv"  # Ensure the file is present in the correct directory
    df = pd.read_csv(csv_path)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# ✅ **Data Preparation**
@st.cache_data
def prepare_data(dataframe):
    """Prepare data for machine learning models"""
    dataframe["combined_text"] = (
        dataframe["skills"].astype(str) + " " + 
        dataframe["education"].astype(str) + " " + 
        dataframe["job_category"].astype(str)
    )
    
    vectorizer = TfidfVectorizer(max_features=500)  
    X = vectorizer.fit_transform(dataframe["combined_text"])
    
    y = np.random.randint(0, 2, size=len(dataframe))  # Simulated labels
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    
    return vectorizer, X, y, model

vectorizer, X, y, model = prepare_data(df)

# 🎯 **Mode Selection: CSV vs Single Resume**
analysis_mode = st.radio("📊 Select Analysis Mode:", ["Analyse Entire CSV", "Analyse Single Resume"])

# **📂 Analyse Entire CSV Mode**
if analysis_mode == "Analyse Entire CSV":
    st.markdown("## 📋 Full Dataset Overview")
    st.dataframe(df)

    # 🔍 **SHAP Explanation**
    st.markdown("## 🔎 SHAP Explanation")
    try:
        feature_names = vectorizer.get_feature_names_out()
        global_importance = np.abs(model.coef_[0])
        indices = np.argsort(global_importance)[-10:]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['red' if x < 0 else 'blue' for x in model.coef_[0][indices]]
        plt.barh([feature_names[i] for i in indices], model.coef_[0][indices], color=colors)
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.xlabel('Coefficient Value (+ = more acceptance, - = less acceptance)')
        plt.title('📈 SHAP - Global Feature Importance')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"SHAP Explanation Error: {e}")
        st.exception(e)

    # 🔍 **LIME Explanation**
    st.markdown("## 🔎 LIME Explanation")
    try:
        text_explainer = LimeTextExplainer(class_names=["Reject", "Accept"])
        sample_idx = np.random.randint(0, len(df))
        sample_text = df["combined_text"].iloc[sample_idx]

        if not isinstance(sample_text, str):
            sample_text = str(sample_text)
        
        def predict_proba(texts):
            if isinstance(texts, np.ndarray):
                texts = texts.tolist()
            vec_texts = vectorizer.transform(texts)
            return model.predict_proba(vec_texts)
        
        exp = text_explainer.explain_instance(sample_text, predict_proba, num_features=10)
        fig = exp.as_pyplot_figure(label=1)
        st.pyplot(fig)
        st.info(f"📝 LIME analysis performed on resume: {df['name'].iloc[sample_idx]}")
    except Exception as e:
        st.error(f"LIME Error: {e}")
        st.exception(e)

# **📄 Analyse Single Resume Mode**
elif analysis_mode == "Analyse Single Resume":
    selected_resume = st.selectbox("📑 Choose a resume:", df["name"].unique())

    if selected_resume:
        selected_resume_data = df[df["name"] == selected_resume]
        st.markdown("## 📋 Selected Resume:")
        st.dataframe(selected_resume_data)

        if "combined_text" in selected_resume_data.columns:
            selected_text = selected_resume_data["combined_text"].values[0]

            if isinstance(selected_text, str):  # Ensure it is a string
                selected_vector = vectorizer.transform([selected_text])
                
                # 🔍 **SHAP Explanation**
                st.markdown("## 🔎 SHAP Explanation")
                try:
                    X_dense = selected_vector.toarray()[0]
                    coef = model.coef_[0]
                    contribution = X_dense * coef
                    base_value = model.intercept_[0]
                    prediction = base_value + np.sum(contribution)
                    probability = 1 / (1 + np.exp(-prediction))
                    indices = np.argsort(np.abs(contribution))[-10:]

                    fig, ax = plt.subplots(figsize=(10, 6))
                    colors = ['red' if x < 0 else 'blue' for x in contribution[indices]]
                    plt.barh([vectorizer.get_feature_names_out()[i] for i in indices], contribution[indices], color=colors)
                    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                    plt.xlabel('Contribution to Prediction')
                    plt.title(f'📊 SHAP - Top Features for {selected_resume} (Predicted: {probability:.2f})')
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"SHAP Error: {e}")
                    st.exception(e)

                # 🔍 **LIME Explanation**
                st.markdown("## 🔎 LIME Explanation")
                try:
                    text_explainer = LimeTextExplainer(class_names=["Reject", "Accept"])

                    def predict_proba(texts):
                        if isinstance(texts, np.ndarray):
                            texts = texts.tolist()
                        vec_texts = vectorizer.transform(texts)
                        return model.predict_proba(vec_texts)

                    exp = text_explainer.explain_instance(selected_text, predict_proba, num_features=10)
                    fig = exp.as_pyplot_figure(label=1)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"LIME Error: {e}")
                    st.exception(e)
            else:
                st.warning("⚠️ Selected resume is missing valid text data for analysis.")
        else:
            st.warning("⚠️ 'combined_text' column is missing in the selected resume dataset.")

# **📊 Fairness Audit (Always Visible)**
st.markdown("## 📊 Fairness Audit")
try:
    df["education_group"] = df["education"].apply(
        lambda x: "STEM" if "Science" in x or "Engineering" in x else "Non-STEM"
    )
    selection_rates = df.groupby("education_group").size() / len(df)
    st.bar_chart(selection_rates)
    st.write("📊 Group Distribution:")
    st.write(df.groupby("education_group").size())
except Exception as e:
    st.error(f"Fairness Audit Error: {e}")
    st.exception(e)

# 🔧 **Debugging Information**
st.markdown("---")
st.markdown("## 🛠 Debug Information")
with st.expander("🔍 Show Data and Model Information"):
    st.write(f"Total resumes: {len(df)}")
    st.write(f"Features after vectorization: {X.shape[1]}")
    st.write(f"Model accuracy on training data: {model.score(X, y):.2f}")
