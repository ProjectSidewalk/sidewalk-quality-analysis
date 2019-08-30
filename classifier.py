import sys
import time
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import parse as str_parse
import seaborn as sns
import sklearn.feature_selection
from dateutil import parser as parser
from imblearn.ensemble import (BalancedBaggingClassifier,
                               BalancedRandomForestClassifier,
                               EasyEnsembleClassifier)
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn import linear_model, svm, tree
from sklearn.base import BaseEstimator
from sklearn.ensemble import (BaggingClassifier, BaggingRegressor,
                              ExtraTreesClassifier, ExtraTreesRegressor,
                              RandomForestClassifier, VotingClassifier, RandomForestRegressor)
from sklearn.feature_selection import RFECV, SelectFromModel, VarianceThreshold
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             r2_score, recall_score)
# from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold, cross_val_score, train_test_split, StratifiedKFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import multiprocessing as mp
import warnings
import intersection_proximity
from region_stats import RegionStats
import scipy

from sklearn.base import BaseEstimator

features = ['label_type', 'sv_image_y', 'canvas_x', 'canvas_y', 'heading', 'pitch', 'zoom', 'lat', 'lng', 'proximity_distance', 'proximity_middleness', 'CLASS_DESC', 'ZONEID']

proportion_labels = 0.35
comparisons = pd.DataFrame()
split_num = 0
np.random.seed(0)

def prob_hist(probabilities, n_bins=5):
    return [np.mean(probabilities), np.std(probabilities),
        np.percentile(probabilities, 25), np.percentile(probabilities, 50),
        np.percentile(probabilities, 75), np.mean(probabilities[(probabilities > 0.25) | (probabilities < 0.75)])]

def dearray(array):
    return np.array([list(l) for l in array])

def extract_label_features(point_labels_file, label_correctness_file, population_density_file='data_seattle.geojson', zone_type_file='Zoning_Detailed.geojson'):
    point_labels = pd.read_csv(point_labels_file)
    point_labels.set_index('label_id', inplace=True)
    
    label_correctness = pd.read_csv(label_correctness_file)
    label_correctness = label_correctness.join(point_labels)
    label_correctness = label_correctness[['user_id', 'label_type', 
        'correct', 'sv_image_x', 'sv_image_y', 'canvas_x', 'canvas_y', 
        'heading', 'pitch', 'zoom', 'lat', 'lng']]

    label_correctness.update(label_correctness['correct'][~pd.isna(label_correctness['correct'])] == 't')
    label_type_encoder = OrdinalEncoder()
    label_correctness['label_type'] = label_type_encoder.fit_transform(label_correctness[['label_type']])
    
    ip = intersection_proximity.IntersectionProximity(intersection_proximity.default_settings['seattle'])
    # Intersection Proximity
    def get_proximity_info(label):
        try:
            distance, middleness = ip.compute_proximity(label.lat, label.lng, cache=True)
        except Exception:
            distance = -1
            middleness = -1

        return pd.Series({
            'proximity_distance': distance,
            'proximity_middleness': middleness
        })

    label_correctness = label_correctness.join(label_correctness.apply(get_proximity_info, axis=1))
    
#     # Population Density
#     rs = RegionStats(population_density_file)
#     label_correctness = label_correctness.join(
#         label_correctness.apply(lambda x: pd.Series(rs.get_properties(x.lng, x.lat)), axis=1)
#     )
    
#     # Zone Type
#     rs = RegionStats(zone_type_file)
#     label_correctness = label_correctness.join(
#         label_correctness.apply(lambda x: pd.Series(rs.get_properties(x.lng, x.lat)), axis=1)
#     )
    
    return label_correctness

class UserQualityRegressor(BaseEstimator):
    def __init__(self):
        pass
    
    def fit(self, label_correctness_file, point_labels_file, user_interactions_file, users_file):
        users = pd.read_csv(users_file)
        users = users.set_index('user_id')
        users_for_training = users[users['labels_validated'] > 25].values
        
        label_correctness = extract_label_features(point_labels_file, label_correctness_file)
   
        mask = np.random.permutation(np.arange(len(users_for_training)))
        users_labels_train = users_for_training[mask[:int(proportion_labels * len(mask))]]
        users_labels_test = users_for_training[mask[int(proportion_labels * len(mask)):]]

        train_labels = label_correctness[label_correctness['user_id'].isin(users_for_training)].copy()
        train_labels = train_labels[~pd.isna(train_labels['correct'])]
        train_labels = train_labels[~(pd.isna(train_labels[features]).any(axis=1))]

#         en = OrdinalEncoder()
#         en.fit(pd.concat((train_labels[['CLASS_DESC']], test_labels[['CLASS_DESC']])))
#         train_labels[['CLASS_DESC']] = en.transform(train_labels[['CLASS_DESC']])

        self.rfe_labels = RFECV(estimator=RandomForestClassifier(n_estimators=10), step=1, cv=StratifiedKFold(5),
               scoring='precision')
        self.clf_labels = RandomForestClassifier(random_state=0, n_jobs=-1, n_estimators=30)
        self.clf_accuracy = BaggingRegressor(random_state=0, n_jobs=-1, n_estimators=30)
        self.rfe_accuracy = RFECV(estimator=RandomForestClassifier(n_estimators=10), step=1, cv=StratifiedKFold(5), scoring='f1')

        print('Training label classifier...')
        self.rfe_labels.fit(train_labels[train_labels['user_id'].isin(users_labels_train)][features].values, 
            train_labels[train_labels['user_id'].isin(users_labels_train)]['correct'].astype(int))

        self.clf_labels.fit(train_labels[train_labels['user_id'].isin(users_labels_train)][features].values[:, rfe_labels.support_], 
            train_labels[train_labels['user_id'].isin(users_labels_train)]['correct'].astype(int))

        train_labels = train_labels.join(pd.Series(
            data=clf_labels.predict_proba(train_labels[train_labels['user_id'].isin(users_labels_test)][features].values[:, rfe_labels.support_])[:, 1], 
            index=train_labels[train_labels['user_id'].isin(users_labels_test)].index).rename('prob'), how='outer')

        prob_hist_predictions = pd.DataFrame(train_labels[train_labels['user_id'].isin(users_labels_test)]
            .groupby('user_id').apply(lambda x:\
            prob_hist(x['prob'].values)).rename('prob'))

        prob_hist_predictions = prob_hist_predictions.join(user_quality_features)

        print('Training accuracy classifier...')
        self.rfe_accuracy.fit(np.concatenate((dearray(prob_hist_predictions['prob']), 
            prob_hist_predictions.drop(columns='prob').values), axis=1), 
            y_train.loc[prob_hist_predictions.index] > 65)

        self.clf_accuracy.fit(np.concatenate((dearray(prob_hist_predictions['prob']), 
            prob_hist_predictions.drop(columns='prob').values), axis=1)[:, rfe_accuracy.support_], 
            y_train.loc[prob_hist_predictions.index])

        useful_test = test_labels[~pd.isna(test_labels[features]).any(axis=1)].copy() 
        useful_test.loc[:, 'prob'] = clf_labels.predict_proba(useful_test[features].values[:, rfe_labels.support_])[:, 1]
            
    def predict_accuracy(self, probs, features):
        return self.clf_accuracy.predict([np.concatenate((prob_hist(probs), features))[rfe_accuracy.support_]])[0]

#     def predict(self, )