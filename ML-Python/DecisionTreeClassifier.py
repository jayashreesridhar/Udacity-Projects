# -*- coding: utf-8 -*-
"""
Created on Tue May 30 13:42:46 2017

@author: JAYASHREE
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 30 08:00:44 2017

@author: JAYASHREE
"""

#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.metrics import accuracy_score
from time import time
from New_feature import computeFraction,combine_feature
from tester import test_classifier
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.feature_selection import f_classif
from sklearn.grid_search import RandomizedSearchCV
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop("TOTAL", 0)
### Task 3: Create new feature(s)
submit_dict = {}
for name in data_dict:

    data_point = data_dict[name]
    #data_point.keys()

    #print
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    #print fraction_from_poi
    data_point["fraction_from_poi"] = fraction_from_poi


    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    #print fraction_to_poi
    submit_dict[name]={"from_poi_to_this_person":fraction_from_poi,
                       "from_this_person_to_poi":fraction_to_poi}
    data_point["fraction_to_poi"] = fraction_to_poi
    salary=data_point["salary"]
    bonus=data_point["bonus"]
    combine_salary_bonus=combine_feature(salary,bonus)
    data_point['combine_salary_bonus']=combine_salary_bonus
    exercised_stock_options=data_point['exercised_stock_options']
    total_stock_value=data_point['total_stock_value']
    data_point['combine_stock_value']=combine_feature(exercised_stock_options,total_stock_value)
    


## updated_features_list

features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',
'to_messages','from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi','fraction_from_poi','fraction_to_poi','combine_salary_bonus','combine_stock_value']


    
### Store to my_dataset for easy export below.
my_dataset = data_dict
#No of items in the dictionary
print "Number of employees detail captured",len(data_dict)
print "Number of features in the dataset",len(data_dict['CAUSEY RICHARD A'])

#To find number of POI'sin the dataset
poi_count=0
for key in data_dict.keys():
    if data_dict[key]["poi"]==1 :
        poi_count+=1
print "Number of POI's in the dataset",poi_count

        

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)



# Correlation Matrix Plot
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white")

df = pd.DataFrame.from_dict(my_dataset, orient='index')
# drop non-numeric features, replace NaN with zero
df = df.drop('email_address', axis=1)
# First replace string `NaN` with numpy nan
df.replace(to_replace='NaN', value=np.nan, inplace=True)
#count number of nan's in columns
print df.isnull().sum()
# then fill in nan
df = df.fillna(0)
#print df.head()
# Compute the correlation matrix
corr = df.corr()


# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(12, 9))

# Draw the heatmap using seaborn
sns.heatmap(corr, vmax=.8, square=True)
## Select features based on k scores
from sklearn.feature_selection import SelectKBest

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
#    # This dataset is way too high-dimensional. Better do PCA:
pca = PCA(n_components=2)

# Maybe some original features where good, too?
selection = SelectKBest(f_classif)
# Build estimator from PCA and Univariate selection:

combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])

# Use combined features to transform dataset:
features_union = combined_features.fit(features,labels).transform(features)
param_grid = {
         'features__pca__n_components':[1, 2, 3],
         'features__univ_select__k':range(1,7),  
        'algorithm__max_features':['sqrt','log2'],
         'algorithm__min_samples_split':[3,4,5,7,9,10,15],
         'algorithm__min_samples_leaf':[2,3,5,7,10],
         'algorithm__max_leaf_nodes':[2,4,6,8,10],
         'algorithm__max_depth':[1,2,3,4,5,6,7,8,9,10],
        'algorithm__criterion': ["gini", "entropy"],
         
          }
scaler = MinMaxScaler()

algo =DecisionTreeClassifier()
#skb = SelectKBest(k='all')

pipeline = Pipeline(steps=[('scaler', scaler), ("features", combined_features),('algorithm', algo)])
cv = StratifiedShuffleSplit(labels, 5, test_size=0.3, random_state=42)

gs = RandomizedSearchCV(pipeline, param_grid, cv=cv, scoring='f1')

gs.fit(features, labels)
print "Best estimator",gs.best_estimator_
clf = gs.best_estimator_
# fit the optimal model
clf.fit(features_train, labels_train)
# predict based on the optimal model
pred = clf.predict(features_test)

#print "predicting  time:", round(time()-t1, 3), "s"
accuracy = accuracy_score(pred,labels_test)
print accuracy
print classification_report(labels_test,pred)
dump_classifier_and_data(clf, my_dataset, features_list)