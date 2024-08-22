import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Welcome to the Diabetes Predictor🩺")

image_path = "diabetes.jpg"
st.image(image_path)
st.write("This interactive application is designed to help you explore and understand the diabetes dataset from the National Institute of Diabetes and Digestive and Kidney Diseases. The primary aim of this app is to provide a comprehensive overview of the dataset and visualize key aspects, enabling you to gain insights into the factors associated with diabetes.")
st.write("Below diagrams visualize the factots of distribution in the dataset.\n\n")
df = pd.read_csv("pages/diabetes.csv")
#st.write(df)

# Bar Plot for Outcome counts
st.subheader('Distribution of Diabetes Outcomes')
fig, ax = plt.subplots()
sns.countplot(x='Outcome', data=df, ax=ax)
for patch in ax.patches:
    current_width = patch.get_width()
    diff = current_width - 0.3
    # we change the bar width
    patch.set_width(0.3)
    # we recenter the bar
    patch.set_x(patch.get_x() + diff * .5)
ax.set_title('Distribution of Diabetes Outcomes')
ax.set_xlabel('Diabetes Status')
ax.set_ylabel('Number of Individuals')
ax.set_xticklabels(['No Diabetes', 'Diabetes'])
st.pyplot(fig)
st.write("")
st.write("")
st.write("")

# Age Distribution
st.subheader('Age Distribution')
fig, ax = plt.subplots()
sns.histplot(df['Age'], kde=True, bins=20, ax=ax)
ax.set_title('Age Distribution of Individuals')
ax.set_xlabel('Age')
ax.set_ylabel('Frequency')
st.pyplot(fig)
st.write("")
st.write("")
st.write("")

#BMI Distribution
st.subheader('BMI Distribution')
fig, ax = plt.subplots()
sns.histplot(df['BMI'], kde=True, bins=20, ax=ax)
ax.set_title('BMI Distribution of Individuals')
ax.set_xlabel('BMI')
ax.set_ylabel('Frequency')
st.pyplot(fig)
st.write("")
st.write("")
st.write("")

# Glucose Distribution
st.subheader('Glucose Distribution')
fig, ax = plt.subplots()
sns.histplot(df['Glucose'], kde=True, bins=20, ax=ax)
ax.set_title('Glucose Distribution of Individuals')
ax.set_xlabel('Glucose Level')
ax.set_ylabel('Frequency')
st.pyplot(fig)
st.write("")
st.write("")
st.write("")

# Insulin Distribution
st.subheader('Insulin Distribution')
fig, ax = plt.subplots()
sns.histplot(df['Insulin'], kde=True, bins=20, ax=ax)
ax.set_title('Insulin Distribution of Individuals')
ax.set_xlabel('Insulin Level')
ax.set_ylabel('Frequency')
st.pyplot(fig)
st.write("")
st.write("")
st.write("")

# Correlation Heatmap
st.subheader('Correlation Heatmap')
fig, ax = plt.subplots(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
ax.set_title('Correlation Heatmap of Features')
st.pyplot(fig)
st.write("")
st.write("")
st.write("")

# Box Plot for multiple features
st.subheader('Box Plot of Features')
fig, ax = plt.subplots(figsize=(12, 8))
sns.boxplot(data=df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']], ax=ax)
ax.set_title('Box Plot of Selected Features')
ax.set_xlabel('Features')
ax.set_ylabel('Value')
st.pyplot(fig)
st.write("")
st.write("")
st.write("")

st.sidebar.success("Select Pages From Above Menue")
