#%% [markdown]
# # Two high-level analyses:
# - Predict performance (accuracy) using a regression
# - Classify users into 'bad' vs 'good'--still need to define this but perhaps the threshold is any user < 70% and any user >= 70%
# 
# For github issues and brainstorming features and analyses, use github:
# - https://github.com/ProjectSidewalk/sidewalk-quality-analysis/issues
#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dateutil import parser as parser
import parse as str_parse
import time
from datetime import datetime, timezone
import csv
from sklearn import svm
import sklearn.feature_selection
from sklearn.feature_selection import SelectFromModel
from sklearn import tree
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold
#%%
users = pd.read_csv('ml-users.csv')
users = users.set_index('user_id')
#%%
# Users w/ accuracies below 65% and are not in neighorhoods without sidewalks
overall_bad_users = ['1353d168-ab49-4474-ae8a-213eb2dafab5', '35872a6c-d171-40d9-8e66-9242b835ea71',
                     '6809bd6e-605f-4861-bc49-32e52c88c675', '939b6faa-0b57-4160-bcc2-d11fd2b69d9f',
                      'f5314ef9-3877-438c-ba65-ee2a2bbbf7f5']
#%%
point_labels = pd.read_csv('sidewalk-seattle-label_point.csv')
point_labels.set_index('label_id', inplace=True)
#%%
label_correctness = pd.read_csv('ml-label-correctness.csv')
#%%
label_correctness.set_index('label_id', inplace=True)
#%%
label_correctness = label_correctness.join(point_labels)
#%%
label_correctness = label_correctness[['user_id', 'label_type', 
    'correct', 'sv_image_x', 'sv_image_y', 'canvas_x', 'canvas_y', 
    'heading', 'pitch', 'zoom', 'lat', 'lng']]
#%%
label_correctness.update(label_correctness['correct'][~pd.isna(label_correctness['correct'])] == 't')
#%%
from sklearn.preprocessing import OrdinalEncoder
#%%
label_type_encoder = OrdinalEncoder()
#%%
label_correctness['label_type'] = label_type_encoder.fit_transform(label_correctness[['label_type']])
#%% [markdown]
# # Classification

#%%
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegressionCV
# from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier, BalancedBaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.base import BaseEstimator

#%%
from sklearn.model_selection import train_test_split, KFold

comparisons = pd.DataFrame()
split_num = 0
for train_index, test_index in KFold(n_splits=5, shuffle=True, random_state=0).split(users.index):
    X_train, X_test = users.index[train_index], users.index[test_index]
    y_train, y_test = users['accuracy'][train_index], users['accuracy'][test_index]
    #%%
    # X_train, X_test, y_train, y_test = train_test_split(users.index, users['accuracy'], random_state=1, test_size=0.2)
    #%%
    train_labels = label_correctness[label_correctness['user_id'].isin(X_train)]
    test_labels = label_correctness[label_correctness['user_id'].isin(X_test)]
    #%%
    test_labels = test_labels.drop(columns='correct')
    #%%
    useful_train = train_labels[~pd.isna(train_labels['correct'])]

    #%%
    clf = BalancedBaggingClassifier(random_state=0)
    features = ['label_type', 'sv_image_x', 'sv_image_y', 'canvas_x', 'canvas_y', 'heading', 'pitch', 'zoom', 'lat', 'lng']
    #%%
    useful_train = useful_train[~pd.isna(useful_train).any(axis=1)]

    #%%
    clf.fit(useful_train[features], useful_train['correct'].astype(int))

    #%%
    # Probabililty correct
    useful_test = test_labels[~pd.isna(test_labels).any(axis=1)]
    useful_test.loc[:, 'prob'] = clf.predict_proba(useful_test[features])[:, 1]

    #%%
    mean_probs = useful_test.groupby('user_id').apply(lambda x: 100 * np.nanmean(x['prob'].values)).rename('predicted')

    #%%
    comparison = pd.DataFrame((mean_probs, y_test, pd.Series(np.full((len(y_test)), split_num), name='split_num', index=y_test.index))).T
    comparisons = comparisons.append(comparison)

    #%%

    split_num += 1

#%%
plt.figure()
plt.xlabel('Predicted accuracy')
plt.ylabel('Actual accuracy')
plt.scatter(comparisons['predicted'], comparisons['accuracy'], c=comparisons['split_num'])
#%%
    