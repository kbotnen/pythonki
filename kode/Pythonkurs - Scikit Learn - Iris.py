#!/usr/bin/env python
# coding: utf-8

# ## Load and inspect the dataset

# In[ ]:


import warnings
warnings.simplefilter(action='ignore', category=UserWarning)


# In[ ]:


# Load scikit learn and load the dataset
from sklearn.datasets import load_iris # conda install scikit-learn, optional: conda install scikit-learn-intelex
iris_dataset = load_iris()


# In[ ]:


type(iris_dataset)


# In[ ]:


print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))


# In[ ]:


print("First five columns of data:\n{}".format(iris_dataset['data'][:5]))
print("\n")
print("Targets:\n{}".format(iris_dataset['target'][:]))
print("\n")
print("Target names:\n{}".format(iris_dataset['target_names']))
print("\n")
print("Feature names:\n{}".format(iris_dataset['feature_names']))
print("\n")
print("Dataset location:\n{}".format(iris_dataset['filename']))


# In[ ]:


print(iris_dataset['DESCR'])


# ## Prepare data for training the model

# In[ ]:


# We need to create both a training set AND a testing set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
        iris_dataset['data'], 
        iris_dataset['target'], 
        test_size=0.25, 
        random_state=0
)

print("X_train shape: {}".format(X_train.shape)) 
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape)) 
print("y_test shape: {}".format(y_test.shape))


# In[ ]:


# Lets visualize our training dataset.
import pandas as pd
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
grr = pd.plotting.scatter_matrix(
    iris_dataframe,
    c=y_train,
    figsize=(15, 15),
    marker='o',
    hist_kwds={'bins': 20},
    s=60,
    alpha=.8
)


# ## Build the model

# In[ ]:


# Build the model.

# Import one of many classification algorithms.
from sklearn.neighbors import KNeighborsClassifier

n_neighbors=1

# This is a object containing the algorithm that build the model.
knn = KNeighborsClassifier(n_neighbors=1)


# In[ ]:


knn.fit(X_train[:,2:], y_train) # Slicing notation for numpyarray. We only want the two last columns / dimensions.


# ## Visualize our model

# In[ ]:


# Lets visualize our trained model.
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

x_min,x_max = X_train[:,2].min() - 1, X_train[:,2].max()+ 1
y_min,y_max = X_train[:,3].min() - 1, X_train[:,3].max()+ 1
h=0.02
xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])
cmap_light=ListedColormap(['orange', 'cyan', 'cornflowerblue'])

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='gouraud')
for target in iris_dataset.target_names:
    index=np.where(iris_dataset.target_names==target)[0][0]
    ax1.scatter(X_train[:,2][y_train==index],X_train[:,3][y_train==index],
                cmap=cmap_bold,edgecolor='k', s=20, label=target)
ax1.set_xlim(x_min,x_max)
ax1.set_ylim(y_min,y_max)
ax1.legend()
ax1.set_xlabel("petal length (cm)")
ax1.set_ylabel("petal width (cm)")
ax1.set_title("3-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, 'uniform'))
plt.show()


# ## Use the model

# In[ ]:


# Making a prediction.
new_data = np.array([[4,3.5,1.2,0.5]])
prediction = knn.predict(new_data[:,2:])
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(iris_dataset['target_names'][prediction]))


# ## Validate the model

# In[ ]:


# Validate the model using the testdata we prepared earlier.
y_pred = knn.predict(X_test[:,2:])
print("Test set predictions:\n {}".format(y_pred))


# In[ ]:


print("Test set score: {:.2f}".format(knn.score(X_test[:,2:], y_test)))


# ## Oppsummering

# k-Nearest Neighbors is a simple classification algorithm in which predictions a new data point to the closest data points in the training dataset.
# 
# It is not necessary to use all the features in our training dataset. We can use different combinations to try to achive better results.
