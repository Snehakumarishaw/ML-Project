#!/usr/bin/env python
# coding: utf-8

# # #xyz pvt .ltd price predictor

# In[1]:


import pandas as pd


# In[2]:


housing = pd.read_csv("data.csv")


# In[3]:


housing.head()


# In[4]:


housing.info()


# In[5]:


housing['CHAS  '].value_counts()


# In[6]:


print(housing.columns)


# In[7]:


housing.describe()


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


#for plotting histogram
#import matplotlib.pyplot as plt
#housing.hist(bins=50, figsize=(20, 15))


# # # Train Test Spliting

# In[10]:


#for learning purpose
import numpy as np
def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    print(shuffled)
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices] 


# In[11]:


train_set, test_set = split_train_test(housing, 0.2)


# In[12]:


print(f"Rows in train set: {len(train_set)}\nRows in ftest set: {len(test_set)}\n")


# In[13]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 42)
print(f"Rows in train set: {len(train_set)}\nRows in ftest set: {len(test_set)}\n")


# In[14]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
for train_index, test_index in split.split(housing, housing['CHAS  ']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[15]:


strat_test_set['CHAS  '].value_counts()


# In[16]:


strat_train_set['CHAS  '].value_counts()


# In[17]:


#95/7


# In[18]:


#376/28


# In[19]:


housing =  strat_train_set.copy()


# In[20]:


strat_test_set.describe()


# # LOOKIN FOR CO RELATION

# In[21]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[22]:


from pandas.plotting import scatter_matrix
attributes =  ['MEDV',' RM ','ZN  ','LSTAT ']
scatter_matrix(housing[attributes], figsize =(12,8))


# In[23]:


housing.plot(kind="scatter", x=' RM ', y='MEDV', alpha=0.8)


# # TRYING OUT ATTRIBUTE COMBINATION

# In[24]:


housing["TAXRM"] = housing['TAX ']/housing[' RM ']


# In[25]:


housing.head()


# In[26]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[27]:


housing.plot(kind="scatter", x='TAXRM', y='MEDV', alpha=0.8)


# In[28]:


housing = strat_train_set.drop('MEDV', axis=1)
housing_labels = strat_train_set['MEDV'].copy()


# # MISSING ATTRIBUTES

# In[29]:


#TO TAKE CARE OF MISSING ATTRIBUTES, WE HAVE THREE OPTIONS :
#1 GET RID OF THE  MISSING DATA POINTS 
#2.GET RID OF THE WHOLE ATTRIBUTES
#3.SET THE VALUE TO SOME VALUE  0 MEAN AND MEDIAN


# In[30]:


a = housing.dropna(subset=[' RM ']) # option 1
a.shape


# In[31]:


housing.drop(' RM ', axis=1).shape #option 2
#Note that there is no RM column and also note that the orignal housing dataframe will remain unchanged


# In[32]:


median= housing[' RM '].median()


# In[33]:


housing[' RM '].fillna(median)# compute median for option 3
#Note that there is no RM column and also note that the orignal housing dataframe will remain unchanged


# In[34]:


housing.shape


# In[35]:


housing.describe()# Before we started filling missing attributes


# In[36]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "median")
imputer.fit(housing)


# In[37]:


imputer.statistics_


# In[38]:


x = imputer.transform(housing)


# In[39]:


housing_tr = pd.DataFrame(x, columns=housing.columns)


# In[40]:


housing_tr.describe()


# # scikit-learn Design

# primarily three type of objects
# 
# 1.Estimators - it estimates some parameters based on a dataset . eg:-Imputer
# it has a fit method or transform method . FIT method - fits the dataset and calculates parameters 
# transform 
# 
# 2.Transformers- take input and returns output based on the learnings from fit(). IT ALSO HAS CONVIENECE  function called fit_transform() which fits and   then transforms.
# 
# 3.Predictors- linear regression model is an example of predictor . fit predict and to common function.
# it also give score function which will evalutae the predictions.

# # FEATURE SCALLING

# TWO TYPES OF FEATURES SCALLING METHOD
# 
# 1.MIN, MAX SCALLING ALSO CALLED (NORMALIZATION)
# - FORMULA:-(VALUE-MIN)/(MAX-MIN)
# 
#  Sklearn provides a class called MinMaxScaler for this 
# 
# 2.STANDARDIZATION:- (value-mean)/standardeviation
# 
# sklearn provides a standard scaler for this 

# # #creating pipeline

# In[41]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    #--- add as many as we want in pipeline
    ('std_scaler',StandardScaler()),
])


# In[42]:


housing_num_tr = my_pipeline.fit_transform(housing_tr)


# In[43]:


housing_num_tr.shape


# # Selecting a  desierd model for xyz company pvt.ltd

# In[44]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble  import RandomForestRegressor
#model = LinearRegression()
#model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)


# In[45]:


some_data = housing.iloc[:5]


# In[46]:


some_labels = housing_labels.iloc[:5]


# In[47]:


prepared_data = my_pipeline.transform(some_data)


# In[48]:


model.predict(prepared_data)


# In[49]:


list(some_labels)


# # EVALUATING THE MODEL

# In[50]:


from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)


# In[51]:


rmse


# # using better evaluation techniques - cross validation

# In[52]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error")
rmse_scores = np.sqrt(-scores)


# In[53]:


rmse_scores


# In[54]:


def print_scores(scores):
    print("scores:", scores)
    print("Mean:", scores.mean())
    print("Standard Deviation:", scores.std())


# In[55]:


print_scores(rmse_scores)


# # convert this notebook into a python file and run the pipeline using VS CODE

# # SAVING THE MODEL

# In[56]:


from joblib import dump, load
dump(model,'XYZ.joblib')


# # TESTING THE MODEL ON TEST DATA 

# In[57]:


x_test = strat_test_set.drop('MEDV', axis=1)
y_test = strat_test_set['MEDV'].copy()
x_test_prepared = my_pipeline.transform(x_test)
final_prediction = model.predict(x_test_prepared)
final_mse = mean_squared_error(y_test, final_prediction)
final_rmse = np.sqrt(final_mse)
#print(final_prediction, list(y_test))


# In[58]:


final_rmse


# In[59]:


prepared_data[0]


# # using the model

# In[60]:


from joblib import  dump, load
import numpy as np
model = load('XYZ.joblib')
features = np.array([[-0.43942006,  3.12628155, -1.12165014, -0.27288841, -1.42262747,
       -0.24141041, -1.31238772,  2.61111401, -1.0016859 , -0.5778192 ,
       -0.97491834,  0.41164221, -0.86091034]])
model.predict(features)


# In[ ]:




