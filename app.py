import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Employee Salary Prediction", page_icon="💼", layout="centered")
st.title("💼 Employee Salary Prediction")
st.markdown("Predict whether an employee earns **>50K** or **<=50K**")

@st.cache_resource
def train_model():
    url = "https://raw.githubusercontent.com/RahulMahawar310/Employee-Salary-Prediction---Rahul-Mahawar/main/adult%203.csv"

    try:
        data = pd.read_csv(url)
    except:
        st.error("Failed to load dataset!")
        return None, None

    data.occupation.replace({'?': 'others'}, inplace=True)
    data.workclass.replace({'?': 'NotListed'}, inplace=True)
    data = data[data['workclass'] != 'Without-pay']
    data = data[data['workclass'] != 'Never-worked']
    data = data[~data['education'].isin(['5th-6th', '1st-4th', 'Preschool'])]
    data.drop(columns=['education'], inplace=True)
    data = data[(data['age'] <= 75) & (data['age'] >= 17)]
    data = data[(data['educational-num'] <= 16) & (data['educational-num'] >= 5)]

    encoder = LabelEncoder()
    for col in ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']:
        data[col] = encoder.fit_transform(data[col])

    x = data.drop(columns=['income'])
    y = data['income']

    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)

    model = GradientBoostingClassifier()
    model.fit(x_scaled, y)

    return model, scaler

st.info("Loading model — this may take a moment on first run...")
model, scaler = train_model()

if model:
    st.success("Model is ready!")
    st.sidebar.header("Employee Details")

    age = st.sidebar.slider("Age", 17, 75, 30)
    workclass = st.sidebar.selectbox("Workclass", ["Private", "Self-emp-not-inc", "Self-emp-inc", "Local-gov", "State-gov", "Federal-gov", "NotListed"])
    fnlwgt = st.sidebar.number_input("Final Weight", min_value=10000, max_value=1500000, value=200000)
    educational_num = st.sidebar.slider("Education Level (num)", 5, 16, 10)
    marital_status = st.sidebar.selectbox("Marital Status", ["Never-married", "Married-civ-spouse", "Divorced", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"])
    occupation = st.sidebar.selectbox("Occupation", ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "others"])
    relationship = st.sidebar.selectbox("Relationship", ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"])
    race = st.sidebar.selectbox("Race", ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"])
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    capital_gain = st.sidebar.number_input("Capital Gain", min_value=0, max_value=100000, value=0)
    capital_loss = st.sidebar.number_input("Capital Loss", min_value=0, max_value=4000, value=0)
    hours_per_week = st.sidebar.slider("Hours per Week", 1, 99, 40)
    native_country = st.sidebar.selectbox("Native Country", ["United-States", "India", "Mexico", "Philippines", "Germany", "Canada", "Puerto-Rico", "El-Salvador", "Cuba", "Others"])

    workclass_map = {"NotListed":0,"Federal-gov":1,"Local-gov":2,"Private":3,"Self-emp-inc":4,"Self-emp-not-inc":5,"State-gov":6}
    marital_map = {"Divorced":0,"Married-AF-spouse":1,"Married-civ-spouse":2,"Married-spouse-absent":3,"Never-married":4,"Separated":5,"Widowed":6}
    occupation_map = {"Adm-clerical":0,"Armed-Forces":1,"Craft-repair":2,"Exec-managerial":3,"Farming-fishing":4,"Handlers-cleaners":5,"Machine-op-inspct":6,"Other-service":7,"Priv-house-serv":8,"Prof-specialty":9,"Protective-serv":10,"Sales":11,"Tech-support":12,"Transport-moving":13,"others":14}
    relationship_map = {"Husband":0,"Not-in-family":1,"Other-relative":2,"Own-child":3,"Unmarried":4,"Wife":5}
    race_map = {"Amer-Indian-Eskimo":0,"Asian-Pac-Islander":1,"Black":2,"Other":3,"White":4}
    gender_map = {"Female":0,"Male":1}
    country_map = {"Canada":0,"Cuba":1,"El-Salvador":2,"Germany":3,"India":4,"Mexico":5,"Others":6,"Philippines":7,"Puerto-Rico":8,"United-States":9}

    input_data = np.array([[
        age, workclass_map.get(workclass, 3), fnlwgt, educational_num,
        marital_map.get(marital_status, 4), occupation_map.get(occupation, 7),
        relationship_map.get(relationship, 1), race_map.get(race, 4),
        gender_map.get(gender, 1), capital_gain, capital_loss,
        hours_per_week, country_map.get(native_country, 9)
    ]])

    input_scaled = scaler.transform(input_data)

    if st.button("🔍 Predict Salary"):
        prediction = model.predict(input_scaled)
        if ">50K" in prediction[0]:
            st.success("✅ Prediction: Salary > $50,000 per year")
        else:
            st.warning("📊 Prediction: Salary ≤ $50,000 per year")
