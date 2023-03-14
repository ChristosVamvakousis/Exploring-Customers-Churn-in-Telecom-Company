import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

def FromStringToNumeric(value):
    if value == 'yes' or value == 'Yes' or value == 'DSL':
        return 1
    elif value == 'Fiber optic':
        return 2
    else:
    	return 0
        
########################################################################### GOING TO DATA LOADING AND CLEANING
print('Loading the Data......')
allData = pd.read_csv('telco_churn.csv')
services = allData[['PhoneService','MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','InternetService','Churn']]
services = services.dropna()
services = services.drop_duplicates()
services = services.applymap(FromStringToNumeric)
services.to_csv('Services.csv', index=False)
########################################################################### GOING TO DATA ANALYSIS

TopColumns=services[['MultipleLines','OnlineBackup','DeviceProtection','StreamingTV','InternetService','Churn']]
corr_matrix = TopColumns.corrwith(TopColumns['Churn'])
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix.to_frame(), cmap="Greens",annot=True)
plt.title('Correlation Matrix for customers and services')
plt.show()


"""
fig, axs = plt.subplots(2)
axs[0].boxplot(left)
axs[0].set_title('Customers who left the company services')
axs[1].boxplot(stayed)
axs[1].set_title('Still being customers to the company')
plt.show()
"""
