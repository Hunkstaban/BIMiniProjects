# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(page_title="Wine Analytics", layout="wide")
st.title("Wine Analytics Interactive Dashboard")

def load_data():
    DATA_PATH = Path("../data")
    redWine = DATA_PATH / "winequality-red.xlsx"
    whiteWine = DATA_PATH / "winequality-white.xlsx"
    df_red = pd.read_excel(redWine, header=1)
    df_white = pd.read_excel(whiteWine, header=1)
    df_red['wine_type'] = 'red'
    df_white['wine_type'] = 'white'
    df = pd.concat([df_red, df_white], ignore_index=True)
    return df

df = load_data()

wine_types = st.sidebar.multiselect(
    "Select wine type(s):", options=df['wine_type'].unique(), default=df['wine_type'].unique()
)
df_filtered = df[df['wine_type'].isin(wine_types)]

st.header("2D Data Visualizations")
fig, ax = plt.subplots()
sns.countplot(x='quality', hue='wine_type', data=df_filtered, ax=ax)
st.pyplot(fig)

fig, ax = plt.subplots()
sns.boxplot(x='quality', y='alcohol', hue='wine_type', data=df_filtered, ax=ax)
st.pyplot(fig)

st.header("3D Visualization: PCA")
features = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
]
X = df_filtered[features].dropna()
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
    X_pca[:,0], X_pca[:,1], X_pca[:,2],
    c=pd.Categorical(df_filtered['quality']).codes[:len(X_pca)],
    cmap='viridis', alpha=0.7
)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.colorbar(scatter, ax=ax, label='Quality')
st.pyplot(fig)
