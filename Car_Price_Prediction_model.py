#!/usr/bin/env python
# coding: utf-8

# # 1. Introduction
# 
# # 2. Loading Data and Explanation of Features
# 
# # 3. Exploratory Data Analysis (EDA)
# 
# # 4. Applying Regression Models
# 
# # 5. CONCLUSION

# # 1. Introduction
# 
# Hello everyone! In this kernel we will be working on Vehicle dataset from cardekho Dataset . This dataset contains information about used cars listed on www.cardekho.com. We are going to use for finding predictions of price with the use of regression models.
# 
# The datasets consist of several independent variables include:
# 
# Car_Name
# Year
# Selling_Price
# Present_Price
# Kms_Driven
# Fuel_Type
# Seller_Type
# Transmission
# Owner
# We are going to use some of the variables which we need for regression models.

# # 2. Loading Data and Explanation of Features

# In[51]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns #seaborn
import matplotlib.pyplot as plt # matplotlib
import warnings # ignore warnings
warnings.filterwarnings("ignore")
import os


# In[52]:


data=pd.read_csv("C:/Users/lenovo/Desktop/car data.csv")
data.head(10)


# In[53]:


data.info()


# In[54]:


data.isna().any()

#Looks like our data is complete one. There is no NaN values and also feature's types are proper.

#Lets see value counts of the features which are the object type.
# In[55]:


print(data.Fuel_Type_of_Car.value_counts(),"\n")
print(data.Seller_Type_of_Car.value_counts(),"\n")
print(data.Transmission_of_Car.value_counts())

#I am going to chance these object values to numerical values to make it proper for regression models.
# In[56]:


data.Fuel_Type_of_Car.replace(regex={"Petrol":"0","Diesel":"1","CNG":"2"},inplace=True)
data.Seller_Type_of_Car.replace(regex={"Dealer":"0","Individual":"1"},inplace=True)
data.Transmission_of_Car.replace(regex={"Manual":"0","Automatic":"1"},inplace=True)
data[["Fuel_Type_of_Car","Seller_Type_of_Car","Transmission_of_Car"]]=data[["Fuel_Type_of_Car","Seller_Type_of_Car","Transmission_of_Car"]].astype(int)


# # 3. Exploratory Data Analysis (EDA)
# 
# Before applying regression models, lets look at the features and also relationship with each other by visually.

# In[57]:


sns.pairplot(data,diag_kind="kde", diag_kws=dict(shade=True, bw=.05, vertical=False))
plt.show()


# In[58]:


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(16,9))
ax  = fig.gca(projection = "3d")

plot =  ax.scatter(data["Car_Year"],
           data["Present_Price_of_Car"],
           data["Kms_Driven_by_Car"],
           linewidth=1,edgecolor ="k",
           c=data["Selling_Price_of_Car"],s=100,cmap="hot")

ax.set_xlabel("Car_Year")
ax.set_ylabel("Present_Price_of_Car")
ax.set_zlabel("Kms_Driven_by_Car")

lab = fig.colorbar(plot,shrink=.5,aspect=5)
lab.set_label("Selling_Price_of_Car",fontsize = 15)

plt.title("3D plot for Year, Present price and Kms driven",color="red")
plt.show()


# # 4. Applying Regression Models
# 
# Firstly lets separate Selling price from the data and drop unnecessary features.

# In[59]:


y=data.Selling_Price_of_Car
x=data.drop(["Selling_Price_of_Car","Car_Name"],axis=1)


# In[60]:


#Spliting data to train and test sizes.

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
print("x train: ",x_train.shape)
print("x test: ",x_test.shape)
print("y train: ",y_train.shape)
print("y test: ",y_test.shape)


# In[61]:


#Secondly we are going to load libraries that we need calculate scores fo regression models. Than apply function which fit the models, get the scores and plot our predictions .

from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score


# In[62]:


cv=5 # CV value
r_2 = [] # List for r 2 score
CV = [] # list for CV scores mean

# Main function for models
def model(algorithm,x_train_,y_train_,x_test_,y_test_): 
    algorithm.fit(x_train_,y_train_)
    predicts=algorithm.predict(x_test_)
    prediction=pd.DataFrame(predicts)
    R_2=r2_score(y_test_,prediction)
    cross_val=cross_val_score(algorithm,x_train_,y_train_,cv=cv)
    
    # Appending results to Lists 
    r_2.append(R_2)
    CV.append(cross_val.mean())
    
     # Printing results  
    print(algorithm,"\n") 
    print("r_2 score :",R_2,"\n")
    print("CV scores:",cross_val,"\n")
    print("CV scores mean:",cross_val.mean())
    
    # Plot for prediction vs originals
    test_index=y_test_.reset_index()["Selling_Price_of_Car"]
    ax=test_index.plot(label="originals",figsize=(12,6),linewidth=2,color="r")
    ax=prediction[0].plot(label = "predictions",figsize=(12,6),linewidth=2,color="g")
    plt.legend(loc='upper right')
    plt.title("ORIGINALS VS PREDICTIONS")
    plt.xlabel("index")
    plt.ylabel("values")
    plt.show()


# # 1. Linear Regression 

# In[63]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
model(lr,x_train,y_train,x_test,y_test)


# # 2. Lasso
# 
# Before applying Lasso model, I am going to assign a alpha range that effect model and choose the best estimator for model.

# In[64]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

alphas = np.logspace(-3,3,num=14) # range for alpha

grid = GridSearchCV(estimator=Lasso(), param_grid=dict(alpha=alphas))
grid.fit(x_train, y_train)

print(grid.best_score_)
print(grid.best_estimator_.alpha)


# In[65]:


ls = Lasso(alpha = grid.best_estimator_.alpha, normalize = True) # applied the best estimator
model(ls,x_train,y_train,x_test,y_test)


# # 3. Ridge
# 
# We are going to do same operation for Ridge

# In[66]:


from sklearn.linear_model import Ridge

alphas = np.logspace(-3,3,num=14) # range for alpha

grid2 = GridSearchCV(estimator=Ridge(), param_grid=dict(alpha=alphas)) 
grid2.fit(x_train, y_train)

print(grid2.best_score_)
print(grid2.best_estimator_.alpha)


# In[67]:


ridge = Ridge(alpha = 0.01, normalize = True) # applied the best estimator
model(ridge,x_train,y_train,x_test,y_test)


# # 4. Decision Tree Regressor

# In[68]:


from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
model(dtr,x_train,y_train,x_test,y_test)


# # 5. Random Forest Regressor

# In[69]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
model(rf,x_train,y_train,x_test,y_test)


# In[70]:


#Lets see the results together in dataframe

Model = ["LinearRegression","Lasso","Ridge","DecisionTreeRegressor","RandomForestRegressor"]
results=pd.DataFrame({'Model': Model,'R Squared': r_2,'CV score mean': CV})
results


# # 5. CONCLUSION
# 
# We applied couple of regression models on dataset. From the final dataframe, it gives opinion about the score of models and also the plots help us to understand which models is more succesful.

# In[79]:


import pickle

# save the model to disk
filename = 'Car_Price_Prediction_model'
pickle.dump(model, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(x_test, y_test)
print(result)


# In[ ]:




