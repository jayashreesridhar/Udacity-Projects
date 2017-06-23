# -*- coding: utf-8 -*-
"""
Created on Wed May 24 12:19:13 2017

@author: JAYASHREE
"""

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.metrics import accuracy_score
from time import time
from New_feature import computeFraction,combine_feature
from tester import test_classifier
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import classification_report
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.feature_selection import f_classif
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi','salary','bonus','long_term_incentive','exercised_stock_options','from_poi_to_this_person','from_this_person_to_poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop("TOTAL", 0)
### Task 3: Create new feature(s)
submit_dict = {}
for name in data_dict:

    data_point = data_dict[name]

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
#print data_dict['CAUSEY RICHARD A']


## updated_features_list
#features_list = ['poi','salary','bonus','long_term_incentive','exercised_stock_options','fraction_from_poi','fraction_to_poi']
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



from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
# Train a SVM classification model

print "Fitting the classifier to the training set"
# This dataset is way too high-dimensional. Better do PCA:
pca = PCA(n_components=2)

# Maybe some original features where good, too?
selection = SelectKBest(f_classif)
# Build estimator from PCA and Univariate selection:

combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])

# Use combined features to transform dataset:
features_union = combined_features.fit(features,labels).transform(features)
scaler = MinMaxScaler()

abc =SVC(kernel='rbf', class_weight='balanced')
t0 = time()

param_grid = {
          'features__pca__n_components':[1, 2, 3],
         'features__univ_select__k':range(1,7),  
         'abc__C': [10,1e2,1e3,1e4,1e5],
          'abc__gamma': [0.0001,0.0005,0.001,0.005,0.01,0.05],
    
          }
# for sklearn version 0.16 or prior, the class_weight parameter value is 'auto'
pipeline = Pipeline(steps=[('scaler', scaler), ("features", combined_features),('abc', abc)])
cv = StratifiedShuffleSplit(labels, 5, test_size=0.3, random_state=42)

gs = RandomizedSearchCV(pipeline, param_grid, cv=cv, scoring='f1')

gs.fit(features, labels)

print "Best estimator",gs.best_estimator_
clf = gs.best_estimator_
clf.fit(features_train,labels_train)
print "done in %0.3fs" % (time() - t0)




###############################################################################
# Quantitative evaluation of the model quality on the test set


t0 = time()
pred = clf.predict(features_test)
print "done in %0.3fs" % (time() - t0)

print classification_report(labels_test, pred)



dump_classifier_and_data(clf, my_dataset, features_list)