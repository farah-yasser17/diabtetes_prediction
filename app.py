import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score

# Preprocessing and model setup
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv('diabetes.csv')
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])
    return df

def train_model():
    df = load_and_preprocess_data()
    X = df.drop(['Outcome'], axis=1)
    y = df['Outcome']   

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    # Cross-validation for robust evaluation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return model, sc, accuracy, report, f1, cv_scores.mean()

model, scaler, acc, report, f1, cv_mean = train_model()

# Streamlit app
st.title("🌟 Diabetes Risk Prediction for Females 🌟")
st.markdown(
    """
    This application predicts the risk of diabetes based on user inputs. It uses a dataset from Kaggle and employs an
    **Gradient Boosting Classifier** for prediction.
    """
)

st.sidebar.title("Instructions")
st.sidebar.info(
    """
    - Fill in the input fields below with your data.
    - Click on the 'Predict Risk' button to see the results.
    - This app is for informational purposes only.
    """
)

# Input fields
st.subheader("Enter Your Details")
Pregnancies = st.number_input("Number of Pregnancies", 0, 16, step=1)
Glucose = st.slider("Glucose Level", 74, 200, 100)
BloodPressure = st.slider("Blood Pressure", 30, 130, 80)
SkinThickness = st.slider("Skin Thickness (mm)", 0, 100, 20)
Insulin = st.slider("Insulin Level (IU/mL)", 0, 200, 50)
BMI = st.slider("Body Mass Index (BMI)", 14.0, 60.0, 25.0)
DiabetesPedigreeFunction = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
Age = st.number_input("Age", 10, 100, step=1)

# Prediction logic
inputs = [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]
scaled_inputs = scaler.transform(inputs)

if st.button("Predict Risk"):
    result = model.predict(scaled_inputs)
    if result[0] == 0:
        st.success("🎉 You are at low risk of developing diabetes! Maintain a healthy lifestyle.")
    else:
        st.warning("⚠️ You may be at risk of developing diabetes. Please consult a healthcare professional.")

st.sidebar.title("Model Performance")
st.sidebar.markdown(f"**Accuracy:** {acc:.2f}")
st.sidebar.markdown(f"**F1 Score:** {f1:.2f}")
st.sidebar.markdown(f"**Cross-Validation Accuracy:** {cv_mean:.2f}")

st.markdown("---")
st.caption("App created by ElBod ❤️")
