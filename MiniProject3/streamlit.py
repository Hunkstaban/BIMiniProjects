import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

df = pd.read_csv('./data/Employee-Attrition.csv')

st.write("# Data cleaning")
st.write("Dataframe starting shape")
st.text(df.shape)

st.write(df.describe())

exploredAndCleanedComment = '''
From the exploration above there is no empty, null, or duplicate values in the dataset, but there are values that are redundant/irrelevant, being:
- EmployeeCount, as it's always 1
- StandardHours, as it is always 80
- EmployeeNumber, as it's a unique identifier
- Over18, as it's always yes (Y)

These can be removed as they will offer no value for further processing.
'''
st.write(exploredAndCleanedComment)


st.write("To process further we encode the object values using label encoding")

code = '''cols_to_encode = [
    'Attrition', 'BusinessTravel', 'Department', 'EducationField',
    'Gender', 'JobRole', 'MaritalStatus', 'OverTime'
]

for col in cols_to_encode:
    df_clean[col], _ = pd.factorize(df_clean[col])'''
st.code(code, language="python")

df_clean = df.drop(['EmployeeCount', 'StandardHours', 'EmployeeNumber', 'Over18'], axis=1)

cols_to_encode = [
    'Attrition', 'BusinessTravel', 'Department', 'EducationField',
    'Gender', 'JobRole', 'MaritalStatus', 'OverTime'
]

for col in cols_to_encode:
    df_clean[col], _ = pd.factorize(df_clean[col])

st.write("# Visualizing our target variable (Attrition)")

## attritionFigure, axis = plt.figure(figsize=(6,4))

fig, ax = plt.subplots(figsize=(6,4))

sns.countplot(x='Attrition', data=df_clean, ax=ax)

plt.title('Employee Attrition')
plt.xlabel('Attrition')
plt.ylabel('Count', )
## plt.tight_layout


st.pyplot(fig)

st.write("# Visualizing each feature")
df_clean.hist(figsize=(14, 14))

st.pyplot(plt.gcf())

clean_data_cluster = pd.read_csv("./data/Emp-Attrition-Initial-Clean.csv")

X = clean_data_cluster.drop(columns=['Attrition'])

clean_data_cluster.drop(columns=['Attrition'])
st.write(X.shape)

joblib.load('../model/kmmodel.pkl')
