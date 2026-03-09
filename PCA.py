# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

import random

import seaborn as sns

# %%
df = pd.read_csv("cleaned_data.csv")
df.head()

# %%
df['Operator'] = df['Operator'].map({'Dauda': 0, 'Aidan': 1, 'JD': 2, 'Muizat': 3})
df['Batch'] = df['Batch'].map({'B': 0, 'D': 1, 'E' : 2})

# %%
X = df.drop('Column Name', axis=1)
y = df['Column Name']

# %%
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = X_scaled[:,2:]
print(X_scaled[:2])

# %%
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(np.cumsum(pca.explained_variance_ratio_)) # 91.4% - first 2 components

# %%
np.unique(y)

# %%
def perform_PCA(subset):
    concentration = subset["Column Name"].str.split(" ").str[0] # concentration
    subset["Concentration"] = concentration

    subset["Blank_Or_PFOA"] = np.where(subset["Concentration"].str.contains("Blank"), "Blank", "PFOA")

    pca_df = subset
    pca_df["PC1"] = X_pca[:, 0]
    pca_df["PC2"] = X_pca[:, 1]
    
    sns.scatterplot(data = pca_df, x = "PC1", y = "PC2", hue = concentration, style=subset["Blank_Or_PFOA"])

perform_PCA(df)

# %%
def PCA_no_legend(subset):
    concentration = subset["Column Name"].str.split(" ").str[0] # concentration
    subset["Concentration"] = concentration

    subset["Blank_Or_PFOA"] = np.where(subset["Concentration"].str.contains("Blank"), "Blank", "PFOA")

    pca_df = subset
    pca_df["PC1"] = X_pca[:, 0]
    pca_df["PC2"] = X_pca[:, 1]
    
    sns.scatterplot(data = pca_df, x = "PC1", y = "PC2", hue = concentration, style=subset["Blank_Or_PFOA"], legend = False) # removed legend

PCA_no_legend(df)


