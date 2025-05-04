# -*- coding: utf-8 -*-
# IMPORT REQUIRED LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# VALIDATING OUR MODEL

from sklearn.metrics import confusion_matrix,classification_report
from sklearn.cluster import KMeans
from sklearn.preprocessing  import StandardScaler
from sklearn.decomposition import PCA

# LOAD DATASET

iris=load_iris()

df =pd.DataFrame(iris.data,columns=iris.feature_names)

df

iris.target

df['target']= iris.target
df['flower_name']=df['target'].apply(lambda x:iris.target_names[x])

df

df.head(40)

# PREPARING THE FEATURES

x=df.drop(['target','flower_name'],axis=1)
y=df['target']

scaler=StandardScaler()
x_scalled=scaler.fit_transform(x)

# USE THE ELBOW METHOD TO DETERMINE THE THE OPTIMAL NUMBER OF CLUSTER FOR K-means


sse=[]
k_range=range(1,11)
for k in k_range:
    km=KMeans(n_clusters=k,random_state=42)
    km.fit(x_scalled)
    sse.append(km.inertia_)

plt.figure(figsize=(8,5))
plt.plot(k_range,sse,marker='o')
plt.title('Elbow Method  for Optimal k')
plt.xlabel('Number of clusters (K)')
plt.ylabel('sum of the squared distance(sse)')

plt.grid(True)
plt.show()