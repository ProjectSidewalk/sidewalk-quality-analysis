import sys
import time
from datetime import datetime, timezone
import pickle
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
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import multiprocessing as mp
import warnings
import intersection_proximity
# from region_stats import RegionStats
import scipy

from sklearn.base import BaseEstimator

# List of the features using label data
features = ['label_type', 'sv_image_y', 'canvas_x', 'canvas_y', 'heading', 'pitch', 'zoom', 'lat', 'lng']# 'proximity_distance', 'proximity_middleness', 'CLASS_DESC', 'ZONEID']

proportion_labels = 0.35
comparisons = pd.DataFrame()
split_num = 0
np.random.seed(0)

# Calculates the mean, standard deviation, and several percential of the model's prediction
def prob_hist(probabilities, n_bins=5):
    return [np.mean(probabilities), np.std(probabilities),
        np.percentile(probabilities, 25), np.percentile(probabilities, 50),
        np.percentile(probabilities, 75), np.mean(probabilities[(probabilities > 0.25) | (probabilities < 0.75)])]

def dearray(array):
    return np.array([list(l) for l in array])

# Creates the interaction feature of the users inputted to be tested against the model
# counting_features is the title of the interaction data features
# counting_features_actions is the name that the action is logged under by the website and corresponds
# to those in the counting_features list
def extract_user_features(filename, panos_file, user_name):
    df_user = pd.read_csv(filename)
    user_panos = pd.read_csv(panos_file)
    info = df_user.groupby(('mission_id', 'gsv_panorama_id'))
    counting_features_action = ['LowLevelEvent_mousedown', 'ContextMenu_TagAdded',
                        'LabelingCanvas_FinishLabeling', 'RemoveLabel',
                        'LowLevelEvent_keydown', 'Click_ZoomIn',
                        'ContextMenu_TextBoxChange', 'LowLevelEvent_mousemove']
    counting_features = ['Mouse Clicks', 'Tags Added', 'Labels Confirmed', 'Labels Removed',
                 'Key Presses', 'Zoom', 'Comments Written', 'Tags Added']

    users_features = []
    users_features_header = []
    for index, feature in enumerate(counting_features_action):
        by_pano = df_user.groupby(('mission_id', 'gsv_panorama_id')).apply(lambda x: sum(x['action'] == feature))
        pano_mean = by_pano.mean()
#         pano_std = by_pano.std()
        by_mission = df_user.groupby('mission_id').apply(lambda x: sum(x['action'] == feature))
        mission_mean = by_mission.mean()
#         mission_std = by_mission.std()
        users_features.append(pano_mean)
#         users_features.append(pano_std)
        users_features.append(mission_mean)
#         users_features.append(mission_std)
        users_features_header.append(counting_features[index] + 'Mean per Pano')
#         users_features_header.append(counting_features[index] + 'Standard Deviation per Pano')
        users_features_header.append(counting_features[index] + 'Mean per Mission')
#         users_features_header.append(counting_features[index] + 'Standard Deviation per Mission')
    users_features.append(df_user['pitch'].mean())
    users_features_header.append('Pitch')
    df_grouped = df_user.groupby(['gsv_panorama_id'])
    current_heading = []
    for current, group in df_grouped:
        current_heading.append(group['heading'].max() - group['heading'].min())
        full_heading_count = 0
        range = group['heading'].max() - group['heading'].min()
    if range >= 350:
        full_heading_count += 1
    users_features.append(sum(current_heading) / float(len(current_heading))) 
    users_features_header.append('Average Heading Range')
    index = list(user_panos['Unnamed: 0']).index(user_name)
    users_features.append(full_heading_count / float(user_panos['panos seen'][index]))
    users_features_header.append('Panos w/ over 350 Degrees seen')
    
    user_id = df_user.iloc[0]['user_id']
    return users_features, users_features_header, user_id

# Gathers all of the label data from all labels of the inputted users to be tested against the model 
def extract_label_features(point_labels_file, label_correctness_file, population_density_file='data_seattle.geojson', zone_type_file='Zoning_Detailed.geojson'):
    point_labels = pd.read_csv(point_labels_file)
    point_labels.set_index('label_id', inplace=True)
    
    label_correctness = pd.read_csv(label_correctness_file)
    label_correctness = label_correctness.join(point_labels)
    label_correctness = label_correctness[['user_id', 'label_type', 
        'correct', 'sv_image_x', 'sv_image_y', 'canvas_x', 'canvas_y', 
        'heading', 'pitch', 'zoom', 'lat', 'lng']]

    label_correctness.update(label_correctness['correct'][~pd.isna(label_correctness['correct'])] == 't')
    label_type_encoder = OrdinalEncoder()  # TODO make this encoder "static"
    label_correctness['label_type'] = label_type_encoder.fit_transform(label_correctness[['label_type']])
    
#     ip = intersection_proximity.IntersectionProximity(intersection_proximity.default_settings['seattle'])
#     # Intersection Proximity
#     def get_proximity_info(label):
#         try:
#             distance, middleness = ip.compute_proximity(label.lat, label.lng, cache=True)
#         except Exception:
#             distance = -1
#             middleness = -1

#         return pd.Series({
#             'proximity_distance': distance,
#             'proximity_middleness': middleness
#         })

#     label_correctness = label_correctness.join(label_correctness.apply(get_proximity_info, axis=1))
    
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
#   Trains the ml model using the ineraction & label data of all users who have atleast 25 labels validated
    def fit(self, label_correctness_file, point_labels_file, users_file, user_quality_features_file='all_users.csv'):
        users = pd.read_csv(users_file)
        users = users.set_index('user_id')
        y_train = users['accuracy']

        users_for_training = users[users['labels_validated'] > 25].index
        self.label_correctness = extract_label_features(point_labels_file, label_correctness_file)
#  Splits the users into training & testing groups
        user_quality_features = pd.read_csv(user_quality_features_file).set_index('user_id')
        half = int(len(users_for_training) / 2)
        users_labels_train = users_for_training[:half]
        users_labels_test = users_for_training[half:]
   
#         mask = np.random.permutation(np.arange(len(users_for_training)))
#         users_labels_train = users_for_training[mask[:int(proportion_labels * len(mask))]]
#         users_labels_test = users_for_training[mask[int(proportion_labels * len(mask)):]]

        train_labels = self.label_correctness.copy()
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

        self.clf_labels.fit(train_labels[train_labels['user_id'].isin(users_labels_train)][features].values[:, self.rfe_labels.support_], 
            train_labels[train_labels['user_id'].isin(users_labels_train)]['correct'].astype(int))
      
        train_labels = train_labels.join(pd.Series(
            data=self.clf_labels.predict_proba(train_labels[train_labels['user_id'].isin(users_labels_test)][features].values[:, self.rfe_labels.support_])[:, 1], 
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
            prob_hist_predictions.drop(columns='prob').values), axis=1)[:, self.rfe_accuracy.support_], 
            y_train.loc[prob_hist_predictions.index])
   
    
# Creates the prediction of the given user's accuracy based off of their passed in label & interaction data
    def __predict_accuracy(self, probs, user_features):
        return self.clf_accuracy.predict([np.concatenate((prob_hist(probs), user_features))[self.rfe_accuracy.support_]])[0]
# Takes in all the names of the files and creates a prediction of the given user's accuracy
    def predict_one_user(self, filename, label_correctness_file, point_labels_file, panos_file, user_name):
        user_features, user_features_header, user_id = extract_user_features(filename, panos_file, user_name)
        label_correctness = extract_label_features(point_labels_file, label_correctness_file)
        label_correctness = label_correctness[~(pd.isna(label_correctness[features]).any(axis=1))]
        label_correctness = label_correctness[label_correctness['user_id'] == user_id]
        probs = self.clf_labels.predict(label_correctness[features].values[:, self.rfe_labels.support_])
        return self.__predict_accuracy(probs, user_features)
            

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)
    
    @classmethod
    def load(filename):
        return pickle.load(filename)
