import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import streamlit as st

# Step 1: Load the dataset
data = pd.read_csv('pages/diabetes.csv')

# Step 2: Prepare features and target variable
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Save the trained model to a file
with open('trained_model.sav', 'wb') as file:
    pickle.dump(model, file)

# Step 6: Load the model from the file
with open('trained_model.sav', 'rb') as file:
    loaded_model = pickle.load(file)

# Step 7: Streamlit input form
st.write("## Predict Diabetes")
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose", min_value=50, max_value=200, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=40, max_value=140, value=80)
skin_thickness = st.number_input("Skin Thickness", min_value=10, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=15, max_value=276, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=20.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
age = st.number_input("Age", min_value=1, max_value=120, value=25)

# Step 8: Create the user input DataFrame in the correct order
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

# Step 9: Make a prediction
if st.button("Predict"):
    prediction = loaded_model.predict(user_input)[0]
    if prediction == 1:
        st.write("This patient have diabetes.")
    else:
        st.write("You are a healthy man....You haven't diabetes.")
