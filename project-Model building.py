#!/usr/bin/env python
# coding: utf-8

# Project on Model building

# In[ ]:





# 1.Red Wine Quality Prediction Project
# Project Description
# The dataset is related to red and white variants of the Portuguese "Vinho Verde" wine. Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).
# 
# This dataset can be viewed as classification task. The classes are ordered and not balanced (e.g. there are many more normal wines than excellent or poor ones). Also, we are not sure if all input variables are relevant. So it could be interesting to test feature selection methods.
# Attribute Information
# Input variables (based on physicochemical tests):
# 1 - fixed acidity
# 2 - volatile acidity
# 3 - citric acid
# 4 - residual sugar
# 5 - chlorides
# 6 - free sulfur dioxide
# 7 - total sulfur dioxide
# 8 - density
# 9 - pH
# 10 - sulphates
# 11 - alcohol
# Output variable (based on sensory data):
# 12 - quality (score between 0 and 10)
# What might be an interesting thing to do, is to set an arbitrary cutoff for your dependent variable (wine quality) at e.g. 7 or higher getting classified as 'good/1' and the remainder as 'not good/0'.
# This allows you to practice with hyper parameter tuning on e.g. decision tree algorithms looking at the ROC curve and the AUC value.
# You need to build a classification model. 
# Inspiration
# Use machine learning to determine which physiochemical properties make a wine 'good'!
# 
# Dataset Link-
# https://github.com/FlipRoboTechnologies/ML-Datasets/blob/main/Red%20Wine/winequality-red.csv
# 

# Before building the model, let's analyze the dataset to understand the distribution of the variables and identify potential correlations.

# In[33]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns


# In[39]:


df=pd.read_csv("winequality-red.csv")


# In[40]:


print(df)


# In[41]:


df.describe()


# In[42]:


plt.hist(df['quality'], bins=10)
plt.title('Distribution of Quality Scores')
plt.xlabel('Quality Score')
plt.ylabel('Frequency')
plt.show()


# In[48]:


sns.scatterplot(x='fixed acidity', y='quality', data=df)
plt.title('Fixed Acidity vs Quality')
plt.show()

sns.scatterplot(x='volatile acidity', y='quality', data=df)
plt.title('Volatile Acidity vs Quality')
plt.show()

sns.scatterplot(x='citric acid', y='quality', data=df)
plt.title('Citric Acid vs Quality')
plt.show()

sns.scatterplot(x='residual sugar', y='quality', data=df)
plt.title('Residual Sugar vs Quality')
plt.show()


# In[49]:


corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()


# In[50]:


sns.boxplot(x='quality', data=df)
plt.title('Distribution of Quality Scores')
plt.show()


# In[ ]:





# In[43]:


df['good'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)
X = df.drop(['quality', 'good'], axis=1)
y = df['good']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[44]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
lr_model = LogisticRegression()
lr_model.fit(X_train_scaled, y_train)

y_pred_lr = lr_model.predict(X_test_scaled)
print("Logistic Regression:")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Classification Report:")
print(classification_report(y_test, y_pred_lr))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_lr))


# In[45]:


dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_scaled, y_train)

y_pred_dt = dt_model.predict(X_test_scaled)
print("Decision Tree Classifier:")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Classification Report:")
print(classification_report(y_test, y_pred_dt))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_dt))


# In[46]:


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

y_pred_rf = rf_model.predict(X_test_scaled)
print("Random Forest Classifier:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_rf))


# it can be concluded that:
# The Decision Tree Classifier performs poorly.
# The Random Forest Classifier performs the best, with an accuracy of 0.93 and an AUC score of 0.96

# In[ ]:





# In[ ]:





# In[ ]:





# Medical Cost Personal Insurance Project
# Project Description
# Health insurance is a type of insurance that covers medical expenses that arise due to an illness. These expenses could be related to hospitalisation costs, cost of medicines or doctor consultation fees. The main purpose of medical insurance is to receive the best medical care without any strain on your finances. Health insurance plans offer protection against high medical costs. It covers hospitalization expenses, day care procedures, domiciliary expenses, and ambulance charges, besides many others. Based on certain input features such as age , bmi,,no of dependents ,smoker ,region  medical insurance is calculated .
# Columns                                            
# •	age: age of primary beneficiary
# •	sex: insurance contractor gender, female, male
# •	bmi: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height, objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9.
# •	children: Number of children covered by health insurance / Number of dependents
# •	smoker: Smoking
# •	region: the beneficiary's residential area in the US, northeast, southeast, southwest, northwest.
# •	charges: Individual medical costs billed by health insurance
# 
# Predict : Can you accurately predict insurance costs?
# 
# Dataset Link-
# https://github.com/FlipRoboTechnologies/ML-Datasets/blob/main/Medical%20Cost%20Insurance/medical_cost_insurance.csv
# 

# In[24]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns


# In[32]:


df=pd.read_csv("medical_cost_insurance.csv")
df


# In[12]:


print(df.head())  
print(df.info()) 
print(df.describe())  


# In[13]:


plt.hist(df['charges'], bins=10)
plt.xlabel('Medical Costs')
plt.ylabel('Frequency')
plt.title('Distribution of Medical Costs')
plt.show()


# it is clearly seen that the frequency is decresing as per the increase in the medical insuarnce cost

# In[25]:


sns.scatterplot(x='age', y='charges', data=df)
plt.title('Age vs Medical Charges')
plt.show()

sns.scatterplot(x='bmi', y='charges', data=df)
plt.title('BMI vs Medical Charges')
plt.show()

sns.scatterplot(x='children', y='charges', data=df)
plt.title('Number of Children vs Medical Charges')
plt.show()



# In[26]:


corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()


# In[28]:


sns.boxplot(x='smoker', y='charges', data=df)
plt.title('Medical Charges by Smoker Status')
plt.show()


# In[14]:


df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'])


# In[16]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['age', 'bmi', 'children']] = scaler.fit_transform(df[['age', 'bmi', 'children']])


# In[17]:


X = df.drop(['charges'], axis=1)
y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred = lr_model.predict(X_test)

print("Test Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Test R2 Score:", r2_score(y_test, y_pred))


# In[20]:


models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(n_estimators=100)
}


# In[21]:


results = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[model_name] = {'MSE': mse, 'R2': r2}


# In[22]:


for model_name, metrics in results.items():
    print(f"{model_name} - MSE: {metrics['MSE']}, R2: {metrics['R2']}")


# Linear Regression:
# MSE: 33,635,210.43
# R²: 0.7833
# This model explains approximately 78.33% of the variance in the insurance charges.
# 
# Decision Tree:
# MSE: 42,187,156.76
# R²: 0.7283
# This model explains approximately 72.83% of the variance in the insurance charges.
# 
# Random Forest:
# MSE: 20,975,412.60
# R²: 0.8649
# This model explains approximately 86.49% of the variance in the insurance charges.
# 
# 
# The Random Forest model performs the best among the three models, with the lowest MSE and the highest R² value. This indicates that the Random Forest model is the most accurate in predicting insurance costs based on the given features.

# In[ ]:





# In[ ]:




