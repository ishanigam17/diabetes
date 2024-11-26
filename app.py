import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv(r"./diabetes.csv")

# Header with professional theme
st.markdown("""
    <style>
        /* General Body Styling */
        body {
            background-color: #ffffff; /* Change to a lighter white */
            font-family: 'Verdana', sans-serif;
            color: #333333;
        }

        /* Header Styling */
        .header {
            font-size: 48px;
            font-family: 'Verdana', sans-serif;
            font-weight: 600;
            text-align: center;
            color: #00695c; /* Dark teal for a professional feel */
            margin-bottom: 25px;
        }

        .sub-header {
            font-size: 20px;
            font-family: 'Verdana', sans-serif;
            color: #004d40; /* Muted teal for contrast */
            text-align: center;
            margin-bottom: 15px;
        }

        .highlight {
            font-size: 20px;
            color: #00695c; /* Match header color */
            font-weight: bold;
            text-align: center;
        }

        /* Section Dividers */
        hr {
            border: none;
            border-top: 2px solid #26a69a; /* Softer green tone */
            margin: 20px 0;
        }

        /* Sidebar Styling */
        .sidebar .sidebar-content {
            background-color: #f9f9f9; /* Neutral sidebar background */
            padding: 15px;
        }

        .stSidebar {
            background-color: #e3f2fd; /* Light blue for a calming look */
            padding: 15px;
            border-right: 1px solid #81d4fa; /* Subtle border */
        }

        /* Inputs Styling */
        .stSidebar input, .stSidebar select, .stSidebar button {
            border-radius: 8px;
            border: 1px solid #bdbdbd;
            padding: 10px;
        }

        /* Button Styles */
        .stButton>button {
            background-color: #00796b; /* Deep teal */
            color: white;
            font-size: 18px;
            font-weight: bold;
            padding: 10px 20px;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            transition: background-color 0.3s ease, transform 0.3s ease;
        }

        .stButton>button:hover {
            background-color: #004d40; /* Dark teal on hover */
            transform: scale(1.05);
        }

        /* Prediction Result */
        .highlight {
            text-align: center;
            font-size: 20px;
            padding: 15px;
            border: 1px solid #4caf50; /* Green border for emphasis */
            border-radius: 8px;
            background-color: #e8f5e9; /* Soft green background */
            margin: 10px 0;
        }

        /* Table Styling */
        table {
            margin: auto;
            border-collapse: collapse;
            width: 90%;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        th {
            background-color: #00796b;
            color: white;
            text-align: center;
        }

    </style>
""", unsafe_allow_html=True)


st.sidebar.header("🔍 Enter Your Health Details:")

def get_user_input():
    pregnancies = st.sidebar.number_input('Pregnancies', min_value=0, max_value=17, value=3, step=1)
    bp = st.sidebar.number_input('Blood Pressure (mm Hg)', min_value=0, max_value=122, value=70, step=1)
    bmi = st.sidebar.number_input('BMI (Body Mass Index)', min_value=0.0, max_value=67.0, value=20.0, step=0.1)
    glucose = st.sidebar.number_input('Glucose Level (mg/dL)', min_value=0, max_value=200, value=120, step=1)
    skinthickness = st.sidebar.number_input('Skin Thickness (mm)', min_value=0, max_value=100, value=20, step=1)
    dpf = st.sidebar.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.4, value=0.47, step=0.01)
    insulin = st.sidebar.number_input('Insulin Level (IU/mL)', min_value=0, max_value=846, value=79, step=1)
    age = st.sidebar.slider('Age (years)', min_value=21, max_value=88, value=33)

    user_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skinthickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }

    features = pd.DataFrame(user_data, index=[0])
    return features

user_data = get_user_input()

# Data Summary
st.markdown("<h2 style='color: #0c4b33;'>🔬 Health Data Overview</h2>", unsafe_allow_html=True)
st.table(user_data)  # Display the input data in a table format

# Split the data
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Model training
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

# Button styles and prediction
st.markdown("""
    <style>
        .stButton>button {
            background-color: #0c4b33;
            color: white;
            font-size: 20px;
            padding: 10px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #145c45;
        }
    </style>
""", unsafe_allow_html=True)

# Button for prediction
if st.button('📊 Analyze Risk'):
    st.markdown("<h3 style='text-align: center; color: #4a7c59;'>🔄 Analyzing your health data...</h3>", unsafe_allow_html=True)
    
    progress = st.progress(0)
    for percent in range(100):
        progress.progress(percent + 1)
    
    prediction = rf.predict(user_data)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: #0c4b33;'>📋 Prediction Result</h2>", unsafe_allow_html=True)
    result = 'You are not diabetic.' if prediction[0] == 0 else 'You are **at risk of diabetes.**'
    st.markdown(f"<p class='highlight'>{result}</p>", unsafe_allow_html=True)
    
    # Display model accuracy
    accuracy = accuracy_score(y_test, rf.predict(x_test)) * 100
    st.markdown(f"<p style='color: #4a7c59; font-size: 18px;'>Model Accuracy: {accuracy:.2f}%</p>", unsafe_allow_html=True)

else:
    st.markdown("<h3 style='text-align: center; color: #4a7c59;'>👈 Enter your data and click 'Analyze Risk'</h3>", unsafe_allow_html=True)
