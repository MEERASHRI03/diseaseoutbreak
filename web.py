import os
import pickle  # Pretrained models
import streamlit as st  # Web application
from streamlit_option_menu import option_menu  # Styling

st.set_page_config(
    page_title="Prediction of Disease Outbreaks",
    layout="wide",
    page_icon="🩺"
)

# Load models
diabetes_model = pickle.load(open(r"C:\Users\MEERA SHRI\OneDrive\Desktop\Disease Pro\training_sessions\diabetes_model.sav", "rb"))
heart_disease_model = pickle.load(open(r"C:\Users\MEERA SHRI\OneDrive\Desktop\Disease Pro\training_sessions\heart_disease_model.sav", "rb"))
parkinsons_model = pickle.load(open(r"C:\Users\MEERA SHRI\OneDrive\Desktop\Disease Pro\training_sessions\parkinsons_model.sav", "rb"))

# Sidebar navigation
with st.sidebar:
    selected = option_menu(
        "Prediction of disease outbreak system",
        ["Diabetes Prediction", "Heart Disease Predictions", "Parkinsons Predictions"],
        menu_icon="hospital fill",
        icons=["activity", "heart", "person"],
        default_index=0
    )

# ========================== DIABETES PREDICTION ==========================
if selected == "Diabetes Prediction":
    st.title("Diabetes Prediction using ML")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input("Number of Pregnancies")
    with col2:
        Glucose = st.text_input("Glucose level")
    with col3:
        Bloodpressure = st.text_input("Blood Pressure value")
    with col1:
        SkinThickness = st.text_input("Skin Thickness value")
    with col2:
        Insulin = st.text_input("Insulin level")
    with col3:
        BMI = st.text_input("BMI value")
    with col1:
        DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function value")
    with col2:
        Age = st.text_input("Age of the person")

    if st.button("Diabetes Test Result"):
        try:
            user_input = [Pregnancies, Glucose, Bloodpressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
            user_input = [float(x) if x.strip() else 0.0 for x in user_input]
            diab_prediction = diabetes_model.predict([user_input])
            diab_diagnosis = "The person is diabetic" if diab_prediction[0] == 1 else "The person is not diabetic"
            st.success(diab_diagnosis)
        except ValueError:
            st.error("Please enter valid numerical values.")

# ========================== HEART DISEASE PREDICTION ==========================
elif selected == "Heart Disease Predictions":
    st.title("Heart Disease Prediction using ML")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.text_input("Age")
    with col2:
        sex = st.text_input("Sex (1 = Male, 0 = Female)")
    with col3:
        cp = st.text_input("Chest Pain Type (0-3)")
    with col1:
        trestbps = st.text_input("Resting Blood Pressure")
    with col2:
        chol = st.text_input("Cholesterol Level")
    with col3:
        fbs = st.text_input("Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)")
    with col1:
        restecg = st.text_input("Resting Electrocardiographic Results (0-2)")
    with col2:
        thalach = st.text_input("Maximum Heart Rate Achieved")

    if st.button("Heart Disease Test Result"):
        try:
            user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach]
            user_input = [float(x) if x.strip() else 0.0 for x in user_input]
            heart_prediction = heart_disease_model.predict([user_input])
            heart_diagnosis = "The person is likely to have Heart Disease" if heart_prediction[0] == 1 else "The person is not likely to have Heart Disease"
            st.success(heart_diagnosis)
        except ValueError:
            st.error("Please enter valid numerical values.")

# ========================== PARKINSONS PREDICTION ==========================
elif selected == "Parkinsons Predictions":
    st.title("Parkinson's Prediction using ML")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        MDVP_Fo = st.text_input("MDVP: Fo (Hz)")
    with col2:
        MDVP_Fhi = st.text_input("MDVP: Fhi (Hz)")
    with col3:
        MDVP_Flo = st.text_input("MDVP: Flo (Hz)")
    with col1:
        MDVP_Jitter = st.text_input("MDVP: Jitter (%)")
    with col2:
        MDVP_Shimmer = st.text_input("MDVP: Shimmer (%)")
    with col3:
        NHR = st.text_input("NHR (Noise-to-Harmonics Ratio)")
    with col1:
        HNR = st.text_input("HNR (Harmonics-to-Noise Ratio)")
    with col2:
        RPDE = st.text_input("RPDE (Recurrence Period Density Entropy)")

    if st.button("Parkinson’s Test Result"):
        try:
            user_input = [MDVP_Fo, MDVP_Fhi, MDVP_Flo, MDVP_Jitter, MDVP_Shimmer, NHR, HNR, RPDE]
            user_input = [float(x) if x.strip() else 0.0 for x in user_input]
            parkinsons_prediction = parkinsons_model.predict([user_input])
            
            parkinsons_diagnosis = (
                "The person is likely to have Parkinson’s disease"
                if parkinsons_prediction[0] == 1
                else "The person is not likely to have Parkinson’s disease"
            )
            
            st.success(parkinsons_diagnosis)
        except ValueError:
            st.error("Please enter valid numerical values.")
