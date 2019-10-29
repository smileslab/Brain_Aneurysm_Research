#!/usr/bin/env python
# coding: utf-8

# #### Import Python Library

# In[1]:


import numpy as np
import numpy.linalg as LA
import pandas as pd
import sklearn as sk
import itertools
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,make_scorer,precision_score,recall_score,roc_auc_score,f1_score
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import svm
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from scipy.spatial import distance
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import mlxtend
import re
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from scipy import stats

import warnings
warnings.filterwarnings('ignore')


# #### Get File Path , features and Status ( Ruptured/Un-Ruptured)

# In[2]:



RCI_loc='C://Users//t8828fa//Desktop//Personal//Higher Study//Phd//Study Material//Research Work//RCI//896 Records_v9.xlsx'
RCI_features= ['Gender','Location',
       'Region', 'Size', 'Size Category (NB)', 'Age', 'Age Category (New 5point)',
       'Diabetes', 'Hypertension', 'Heart Disease', 'COPD','Polycystic Kidney Disease', 'Smoking History',
       'Cigarettes', 'Cigar', 'Smokeless',
       'Multiple Aneurysms', 'Side','Family History','spinning feeling',
       'dizziness', 'diplopia', 'blurred vision', 'speech deficits',
       'motor deficits', 'Sensory deficits']
RCI_label=['Status']


# #### Load data into Dataframe

# In[3]:


df_RCI=pd.read_excel(RCI_loc)


# #### Set Record ID as Index

# In[10]:



df_RCI.set_index('Record ID',inplace=True)


# #### Replcae missing Smoking History attribute

# In[13]:


def impute_Smoking_History(cols):
    Smoking_History = cols[0]
    
    if pd.isnull(Smoking_History):
        return 'NA'
    else:
        return Smoking_History
    
df_RCI['Smoking_History'] = df_RCI[['Smoking_History']] .apply(impute_Smoking_History,axis=1)


# #### Balance the data using randomized under sampling technique

# In[25]:


# Step-1 Get number of records having ruptured status
no_Ruptured = len(df_RCI[df_RCI['Status'] == 'Ruptured'])

# Step-2 Get indices of un-ruptured records
unruptured_indices = df_RCI[df_RCI['Status'] == 'Un-Ruptured'].index

# Step-3 Fetch random indices of un-ruptured records equal to number of ruptured records in he dataset
random_indices = np.random.choice(unruptured_indices,no_Ruptured,replace=False)

# Step-4 Get Ruptured indices
ruptured_indices = df_RCI[df_RCI['Status'] == 'Ruptured'].index

# Step-5 Combine ruptured and un-ruptured indices to form balance date set
under_sample_indices = np.concatenate([ruptured_indices,random_indices])

# Step-6 Create a dataframe for balance dataset
df_RCI_balanced = df_RCI.loc[under_sample_indices]


# #### Separate Features/Predictors(X) and lablel/Status (Y)

# In[27]:


df_RCI_balanced_X=df_RCI_balanced[df_RCI_balanced.columns.drop('Status')]
df_RCI_balanced_Y=df_RCI_balanced['Status']


# #### Label Encoding for features data set
# ##### Some of the Machin learning algorithm ( for example -DTC, RFC) does not work if categorical variable has value type string. This is done to assgn unique integer value to categorival variable values of the features

# In[28]:


# Take a copy of the balanced data set
df_RCI_X_balanced_encoded= df_RCI_balanced_X.copy()

encode = {}
for column in df_RCI_X_balanced_encoded.columns:
        # check if column type is of object ( not integer)
        if df_RCI_X_balanced_encoded.dtypes[column] == np.object:
            
            # Encode the column with integer value
            encode[column] = LabelEncoder()
            
            # assign the value to particular feature value of categorical variable
            df_RCI_X_balanced_encoded[column] = encode[column].fit_transform(df_RCI_X_balanced_encoded[column])


# #### Split the dataset into training and test data

# In[30]:


X_train_b, X_val_b, y_train_b, y_val_b = train_test_split(df_RCI_X_balanced_encoded, 
                                                    df_RCI_balanced_Y, test_size=0.20, 
                                                    random_state=101)


# #### Define K fold and number of splits -

# In[ ]:


n_splits=7
scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score), 
           'f1_score' : make_scorer(f1_score)}

kfold = KFold(n_splits=n_splits, random_state=42)


# #### Algorithm 1-  Decision Tree Classifier(DTC)

# In[ ]:


dtree_b = DecisionTreeClassifier()
results =cross_validate(dtree_b, X_train_b, y_train_b['Status'].map({'Un-Ruptured' :0 ,'Ruptured':1}),cv=kfold,scoring=scoring)
print('Average Accurcay',np.mean(results['test_accuracy']))
print('Average F1 Accurcay',np.mean(results['test_f1_score']))
print('Average Precision ',np.mean(results['test_precision']))
print('Average Recall',np.mean(results['test_recall']))


# #### Algorithm 2 - Random Forest Classifier( RFC)

# In[ ]:


rfc_b = RandomForestClassifier(n_estimators=100,criterion='entropy')
results =cross_validate(rfc_b, X_train_b, y_train_b['Status'].map({'Un-Ruptured' :0 ,'Ruptured':1}),cv=kfold,scoring=scoring)
print('Average Accurcay',np.mean(results['test_accuracy']))
print('Average F1 Accurcay',np.mean(results['test_f1_score']))
print('Average Precision ',np.mean(results['test_precision']))
print('Average Recall',np.mean(results['test_recall']))


# #### Algorithm 3- Adaptive Boosting(AdaBoost)

# In[ ]:


adaboost_b = AdaBoostClassifier(random_state=1)
results =cross_validate(adaboost_b, X_train_b, y_train_b['Status'].map({'Un-Ruptured' :0 ,'Ruptured':1}),cv=kfold,scoring=scoring)
print('Average Accurcay',np.mean(results['test_accuracy']))
print('Average F1 Accurcay',np.mean(results['test_f1_score']))
print('Average Precision ',np.mean(results['test_precision']))
print('Average Recall',np.mean(results['test_recall']))


# #### Algorithm 4- Gradient Boosting ( GBoost)

# In[ ]:


Gboost_b = GradientBoostingClassifier(learning_rate=0.01,random_state=1)
results =cross_validate(Gboost_b, X_train_b, y_train_b['Status'].map({'Un-Ruptured' :0 ,'Ruptured':1}),cv=kfold,scoring=scoring)
print('Average Accurcay',np.mean(results['test_accuracy']))
print('Average F1 Accurcay',np.mean(results['test_f1_score']))
print('Average Precision ',np.mean(results['test_precision']))
print('Average Recall',np.mean(results['test_recall']))


# #### Algorithm 5 Support Vector Machine( SVM)

# In[ ]:


svc_b = svm.SVC(C=1.0, kernel='rbf',probability=True)
results =cross_validate(svc_b, X_train_b, y_train_b['Status'].map({'Un-Ruptured' :0 ,'Ruptured':1}),cv=kfold,scoring=scoring)
print('Average Accurcay',np.mean(results['test_accuracy']))
print('Average F1 Accurcay',np.mean(results['test_f1_score']))
print('Average Precision ',np.mean(results['test_precision']))
print('Average Recall',np.mean(results['test_recall']))


# #### Ensemble Model( DTC,RFC,AdaBoost,GBoost,SBM) using Voting Classifier (Soft Voting)

# In[ ]:


for w1 in range(1,4):
    for w2 in range(1,4):
        for w3 in range(1,4):
            for w4 in range(1,4):
                for w5 in range(1,4):
                    ensemble_model_b = VotingClassifier(estimators=[('dtree_b',dtree_b),('rfc_b', rfc_b),('adaboost_b', adaboost_b),('Gboost_b', Gboost_b),('svc_b', svc_b)], voting='soft',weights=[w1,w2,w3,w4,w5])
                    ensemble_model_b.fit(X_train_b, y_train_b)
                    ensemble_model_pred_b = ensemble_model_b.predict(X_val_b)
                    print('Confusion_Matrix \n',confusion_matrix(y_val_b,ensemble_model_pred_b))
                    print('Classification_Report \n \n',classification_report(y_val_b,ensemble_model_pred_b))
                    print('Accuracy' ,accuracy_score(y_val_b,ensemble_model_pred_b))
                    print('{w1},{w2},{w3},{w4},{w5}'.format(w1=w1,w2=w2,w3=w3,w4=w4,w5=w5))

