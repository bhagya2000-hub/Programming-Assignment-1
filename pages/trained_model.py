import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import streamlit as st

# Load the dataset
data = pd.read_csv('pages/diabetes.csv')

# Check the columns in the dataset
st.write("### Dataset Columns")
st.write(data.columns)  # Display the column names in the Streamlit app

# Prepare features and target variable
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Save the trained model to a file
with open('trained_model.sav', 'wb') as file:
    pickle.dump(model, file)

# Streamlit App
st.title("Diabetes Prediction System")
st.write(f"Model Accuracy: {accuracy:.2f}")

st.write("## Predict Diabetes")
st.write("Please enter the following details:")

# Streamlit input fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose", min_value=50, max_value=200, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=40, max_value=140, value=80)
skin_thickness = st.number_input("Skin Thickness", min_value=10, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=15, max_value=276, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=20.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
age = st.number_input("Age", min_value=1, max_value=120, value=25)

# Create a DataFrame for user input
user_input = pd.DataFrame({
    'Pregnancies': [pregnancies],
    'Glucose': [glucose],
    'BloodPressure': [blood_pressure],
    'SkinThickness': [skin_thickness],
    'Insulin': [insulin],
    'BMI': [bmi],
    'DiabetesPedigreeFunction': [dpf],
    'Age': [age]
})

# Load the model from the file
with open('trained_model.sav', 'rb') as file:
    loaded_model = pickle.load(file)

# Predict
if st.button("Predict"):
    prediction = loaded_model.predict(user_input)[0]
    st.write("## Prediction Result")
    if prediction == 1:
        st.write("You are a Healthy Man...Continoue your meat plans.")
        st.image("main-qimg-a0f3e574a4b0924aabe8b743c986368d.jpg", caption="Risk of Diabetes", use_column_width=True)  # Optional image
    else:
        st.write("This patient unlikely  have diabetes.Please follow correct meat plans and meet your Doctor..")
        st.image("Suckale08_fig3_glucose_insulin_day.png", caption="Low Risk of Diabetes", use_column_width=True)  # Optional image
