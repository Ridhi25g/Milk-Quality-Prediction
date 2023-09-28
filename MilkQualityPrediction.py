#!/usr/bin/env python
# coding: utf-8

# # Milk Quality Prediction

# Our goal in this project is to design a machine learning model and to predict quality of milk using the data we have, ensuring the safety and excellence of dairy products.

# ## Data Set - Variables

# - pH: This Column defines PH alus of the milk which ranges from 3 to 9.5 max : 6.25 to 6.90
# - Temperature: This Column defines Temprature of the milk which ranges from 34'C to 90'C max : 34'C to 45.20'C
# - Taste: This Column defines Taste of the milk which is categorical data 0 (Bad) or 1 (Good) max : 1 (Good)
# - Odor: This Column defines Odor of the milk which is categorical data 0 (Bad) or 1 (Good) max : 0 (Bad)
# - Fat: This Column defines Fat of the milk which is categorical data 0 (Low) or 1 (High) max : 1 (High)
# - Turbidity: This Column defines Turbidity of the milk which is categorical data 0 (Low) or 1 (High) max : 1 (High)
# - Colour: This Column defines Colour of the milk which ranges from 240 to 255 max : 255
# - Grade: This Column defines Grade (Target) of the milk which is categorical data Where Low(Bad), Medium(Moderate), High(Good)

# ## Import libraries

# In[151]:


import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.simplefilter('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# ## Read the data

# In[152]:


df = pd.read_csv("milkpred2.csv")
df.head()


# In[153]:


df.info()


# In[154]:


df.shape


# # Data preprocessing

# ## Label Encoding

# We have 7 numeric values and 1 categorical value.
# Converting the categorical variable (grade) into numerical.

# In[155]:


# Importing LabelEncoder
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
# Encoding Transmission with the help of Label Encoder
df['Grade']=LE.fit_transform(df['Grade']).astype(int)
df.head()


# ## Graphical Analysis

# In[156]:


# Checking for outliers in numerical variables using boxplot
from scipy.stats import norm
num_var=[var for var in df.columns]

# Plotting Box and Distribution plot 
for var in num_var:
    plt.figure(figsize=(15,6))
    plt.subplot(1,2,1)
    ax=sns.boxplot(data=df[var])
    ax.set_title(f'{var}')
    ax.set_ylabel(var)

    plt.subplot(1,2,2)
    ax=sns.distplot(df[var], fit=norm)
    ax.set_title(f'skewness of {var} : {df[var].skew()}')
    ax.set_xlabel(var)
    print('__'*50)
    plt.show()


# In[157]:


sns.pairplot(df)


# In[158]:


countplot_list = ["Colour","Turbidity","Odor","Taste","pH","Fat","Temperature"]
plt.figure(figsize=(15,10))
x=1
for i in countplot_list:
    plt.subplot(4,2,x)
    sns.barplot(x=i,y='Grade',data=df)
    x = x + 1
plt.tight_layout()


# In[159]:


palette_color = sns.color_palette('bright')
  
# plotting data on chart
df.groupby('Grade').size().plot(kind='pie', autopct='%.2f')
  
# displaying chart
plt.show()


# ## Handling Outliers

# In[160]:


#Checking for outliers value in pH,Colour,Temperature,Taste,Odor,Fat,Turbidity,Grade
att=['pH','Colour','Temperature','Taste','Odor','Fat','Turbidity','Grade']
q1, q3 = np.percentile(df[att], [25, 75])
iqr = q3 - q1
lower_bound = q1 - 1.5*iqr
upper_bound = q3 + 1.5*iqr
print(upper_bound)
outliers = df[(df[att] < lower_bound) | (df[att] > upper_bound)]
outliers


# ## Missing values

# In[161]:


df.isnull().sum()


# In[162]:


# pH
value_to_fill=df['pH'].mean().round(1)
df['pH'].fillna(value_to_fill,inplace=True)


# In[163]:


# Temperature
value_to_fill=df['Temperature'].mean().round(1)
df['Temperature'].fillna(value_to_fill,inplace=True)


# In[164]:


# Odor
value_to_fill=df['Odor'].mean().round(1)
df['Odor'].fillna(value_to_fill,inplace=True)


# In[165]:


# Fat
value_to_fill=df['Fat'].mean().round(1)
df['Fat'].fillna(value_to_fill,inplace=True)


# In[166]:


# Taste
value_to_fill=df['Taste'].mean().round(1)
df['Taste'].fillna(value_to_fill,inplace=True)


# In[167]:


# Turbidity
value_to_fill=df['Turbidity'].mean().round(1)
df['Turbidity'].fillna(value_to_fill,inplace=True)


# In[168]:


# Colour
value_to_fill=df['Colour'].mean().round(1)
df['Colour'].fillna(value_to_fill,inplace=True)


# In[169]:


att=["pH","Temperature","Taste","Odor","Fat","Turbidity","Colour"]
miss_bool = df[att].isnull()
miss_bool


# In[170]:


df.isnull().sum()


# ## Statistical description of the data

# In[171]:


df.describe()


# In[172]:


df.corr()


# In[173]:


plt.figure(figsize=(10,5))
sns.heatmap(df.corr(),annot=True)


# We can see that 'Temperature' and 'Taste' have strong positive correlation.
# 
# And 'Temperature' and 'Grade' have strong negative correlation

# ## Modelling

# In[174]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics


# In[175]:


x=df.iloc[:,:7]
x.head()


# In[176]:


y=df.iloc[:,7]
y.head()


# In[177]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
x_train.head()
y_train.head()


# ## Logistic Regression

# In[178]:


LR = LogisticRegression()
LR.fit(x_train,y_train)


# In[179]:


y_predicts =LR.predict(x_test)
y_predicts


# In[180]:


print(confusion_matrix(y_test, y_predicts)) 


# In[181]:


plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_predicts), annot=True);


# In[182]:


LR.score(x_train,y_train)


# In[183]:


LR.score(x_test,y_test)


# ### Classification report for RF model

# In[184]:


LR_Predict = LR.predict(x_train)
LR_Accuracy = accuracy_score(y_train, LR_Predict)
print("Accuracy: " + str(LR_Accuracy))


# In[185]:


resultLR = classification_report(y_test, y_predicts)
print(resultLR)


# ## Random Forest Classifier

# In[186]:


x_train,x_test,y_train, y_test=train_test_split(x,y,test_size=0.33)
x_train.head()


# In[187]:


RFC = RandomForestClassifier()
RFC.fit(x_train, y_train)


# In[188]:


y_predicts =RFC.predict(x_test)
y_predicts


# In[189]:


print(confusion_matrix(y_test, y_predicts))


# In[190]:


RFC.score(x_train,y_train)


# In[191]:


RFC.score(x_test,y_test)


# ### Classification report for RF model

# In[192]:


resultRFC = classification_report(y_test, y_predicts)
print(resultRFC)


# In[193]:


RFC_Predict = RFC.predict(x_train)
RFC_Accuracy = accuracy_score(y_train, RFC_Predict)
print("Accuracy: " + str(RFC_Accuracy))


# In[194]:


pd.DataFrame({'Actual': y_test,  'Predicted': y_predicts}).head(100)


# In[195]:


model_performance_accuracy = pd.DataFrame({'Model': ['LogisticRegression', 'RandomForestClassifier'], 'Accuracy': [LR_Accuracy, RFC_Accuracy]})
model_performance_accuracy.sort_values(by = "Accuracy", ascending = False)


# After a thorough exploration and evaluation of both machine learning models, it is evident that the Random Forest Classifier stands out as the most efficient and accurate model for predicting milk quality.
