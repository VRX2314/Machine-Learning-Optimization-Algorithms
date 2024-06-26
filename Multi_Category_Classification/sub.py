# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# %%
# Load data

df = pd.read_csv('./Iris.csv')
df["Species"]=df["Species"].astype('category').cat.codes
df.head()

# %%
df.info()

# %%
df.describe().T

# %%
# Perform EDA on Data

sns.countplot(df, x='Species', hue='Species')
# Data is Equally Distributed

# %%
sns.pairplot(df, hue='Species')
# There are various distinguishing factors present within the data for each species 

# %%
# Making individual category based dataframes
df_1 = df.copy()
df_1[df['Species']!=0] = 0
df_1[df['Species']==0] = 1

X_1 = df_1.drop(['Id', 'Species'], axis=1)
y_1 = df_1['Species']

# %%
X_train, X_test, y_train, y_test = train_test_split(X_1, y_1, test_size=0.2, random_state=42)

svm_1 = SVC(gamma='auto', probability=True)
svm_1.fit(X_train, y_train)
y_pred = svm_1.predict(X_test)
print('SVM 1')
print(classification_report(y_test, y_pred, target_names=["setosa", 'other']))

# %%
df_2 = df.copy()
df_2[df['Species']!=1] = 0
df_2[df['Species']==1] = 1

X_2 = df_2.drop(['Id', 'Species'], axis=1)
y_2 = df_2['Species']

# %%
X_train, X_test, y_train, y_test = train_test_split(X_2, y_2, test_size=0.2, random_state=42)

svm_2 = SVC(gamma='auto', probability=True)
svm_2.fit(X_train, y_train)
y_pred = svm_2.predict(X_test)
print('SVM 2')
print(classification_report(y_test, y_pred, target_names=["versicolor", 'other']))

# %%
df_3 = df.copy()
df_3[df['Species']!=2] = 0
df_3[df['Species']==2] = 1

X_3 = df_3.drop(['Id', 'Species'], axis=1)
y_3 = df_3['Species']

# %%
X_train, X_test, y_train, y_test = train_test_split(X_3, y_3, test_size=0.2, random_state=42)

svm_3 = SVC(gamma='auto', probability=True)
svm_3.fit(X_train, y_train)
y_pred = svm_3.predict(X_test)
print('SVM 3')
print(classification_report(y_test, y_pred, target_names=["virginica", 'other']))

# %%
# Combining all SVMs
svcs=[svm_1, svm_2, svm_3]
X = df.drop('Species', axis=1)
y = df['Species']

# %%
final=[]
final.append(svm_1.predict(X_1))
final.append(svm_2.predict(X_2))
final.append(svm_1.predict(X_3))

final

finalSVMPred=[max(range(len(svcs)), key=lambda i: classifier[i]) for classifier in zip(*final)]
print('SVC Combined')
print(classification_report(y, finalSVMPred))


