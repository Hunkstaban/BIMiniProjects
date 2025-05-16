import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load

####### --------------------- EEmil --------------------- #######
import pickle

# Function to load the model
def load_income_model():
    with open("model/income_prediction_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_income_model()

# Example: Predict monthly income for a new employee with given TotalWorkingYears
# You need to input the feature(s) in the same shape as the training data (e.g., as a 2D array)
sample_input = [[5]]  # For example: 5 years of working experience

predicted_income = model.predict(sample_input)

st.write(f"Predicted Monthly Income for {sample_input[0][0]} years of experience: ${predicted_income[0]:.2f}")


## st.sidebar.radio("Choose a model", {'Data Exploration', 'K-Means clustering', 'Classification'})

def display_DataExploration():
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


def display_Mean_shift_clustering(df_clean):

    X = df_clean.drop(columns=['Attrition'])

    X = X.iloc[:, 3:5].values

    model = load('./data/msmodel.pkl')

    st.write("## importing and loading pre trained model to apply on our data")

    loadModel = '''
    X = df_clean.drop(columns=['Attrition'])

    X = X.iloc[:, 3:5].values

    model = load('./data/msmodel.pkl')

    '''

    st.code(loadModel)

    usingModel = '''
    labels = model.labels_

    model.predict(X)

    '''
    st.text("With the model we can run prediction on our data based on the models training" \
    " and extract the labels")

    st.code(usingModel)

    labels = model.labels_
    labels_unique = np.unique(labels)
    

    n_clusters_ = len(labels_unique)
    

    cluster_centers = model.cluster_centers_
    
    
    Y = model.predict(X)
   
    st.write("# Visualisation of Data processed by the pre trained model")

    plt.scatter(X[:,0], X[:,1], c=labels, marker="o", picker=True)
    plt.title(f'Estimated number of clusters = {n_clusters_}')
    plt.xlabel('x')
    plt.ylabel('y')
    st.pyplot(plt)


    st.write("# Visualising clusters in 3D")

    fig = plt.figure()
    plt.title('Discovered Clusters')
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,0], X[:,1],  marker='o', cmap='viridis', c=labels)
    ax.scatter(cluster_centers[:,0], cluster_centers[:,1], marker='x', 
           color='red', s=100, linewidth=3, zorder=10)
    st.pyplot(fig)


clean_data_cluster = pd.read_csv("./data/Emp-Attrition-Initial-Clean.csv")

def display_Classification(df_clean):
    st.write("### Task 3 - Supervised machine learning: classification")

    st.text("Visualizing feature correlations with attrition using correlation matrix")

    corr = df_clean.corr(numeric_only=True)

    plt.figure(figsize=(25, 20))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    st.pyplot(plt)

    
    st.write("### Correlation of features with Attrition")
    st.text("Descending order")
    st.write(corr['Attrition'].sort_values(ascending=False))

    st.write("## Using a pre trained model for descision tree")
    st.text("We use the model on our data and calculate and plot the importances for the different features")
    d_tree_model = load("./models/decision_tree_model.pkl")
    X = df_clean.drop('Attrition', axis=1)

    

    importances = pd.Series(d_tree_model.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)

    importance_fig = plt.figure(figsize=(10, 6))
    importances.plot(kind='bar')
    plt.title('Feature Importances from Decision Tree')
    plt.ylabel('Importance')

    st.pyplot(importance_fig)

    st.write("## Using pre trained model for Random Forest")

    rdf_model = load("./models/random_forest_model.pkl")

    rf_importances = pd.Series(rdf_model.feature_importances_, index=X.columns)
    rf_importances = rf_importances.sort_values(ascending=False)
    rdf_fig = plt.figure(figsize=(10, 6))
    rf_importances.plot(kind='bar')
    plt.title('Feature Importances from Random Forest')
    plt.ylabel('Importance')

    st.pyplot(rdf_fig)



app_mode = st.sidebar.selectbox(label="Models", options={'Data Exploration', 'Mean-shift Clustering', 'Classificaiton'})

if app_mode == "Data Exploration":
    display_DataExploration()
elif app_mode == "Mean-shift Clustering":
    display_Mean_shift_clustering(clean_data_cluster)
elif app_mode == "Classificaiton":
    display_Classification(clean_data_cluster)