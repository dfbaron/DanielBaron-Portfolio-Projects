#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
from tqdm import tqdm
import time

from sklearn.model_selection import train_test_split
from scipy import spatial

import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
from sklearn.feature_selection import chi2

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support
import preprocessing as pp

def transform_business_features(df_business, reprocess):
    if reprocess:
        
        df_business = pd.concat([df_business.drop(['attributes'], axis=1), df_business['attributes'].apply(pd.Series)], axis=1)
        categories = [x.split(',') for x in list(df_business['categories']) if x != None]
        categories = list(set([x.lstrip() for x in [item for sublist in categories for item in sublist]]))

        for category in categories:
            column_name = category.replace(' ', '_').replace('-','_').replace('/','_').lower()
            df_business[category] = df_business['categories'].str.contains(category)

        delete_cols = ['address', 'state', 'postal_code', 'latitude', 'longitude', 'categories', 'hours', 'review_count',                        'is_open', 'city', 'stars']
        df_business = df_business.drop(columns = delete_cols)

        dict_cols = ['BusinessParking','Ambience','GoodForMeal','Music', 'BestNights', 'HairSpecializesIn',                      'DietaryRestrictions']
        new_cols = []
        for dict_col in tqdm(dict_cols):
            new_df = df_business[dict_col].apply(pd.Series)
            new_cols.append(new_df.columns)
            df_business = pd.concat([df_business.drop([dict_col], axis=1), new_df], axis=1)

        unicode_cols = ['Alcohol', 'WiFi', 'RestaurantsAttire', 'NoiseLevel', 'Smoking', 'BYOBCorkage', 'AgesAllowed']
        for unicode_col in tqdm(unicode_cols):
            df_business[unicode_col] = df_business[unicode_col].str.replace('u', '')

        df_business = df_business.replace('True', '1').replace('False', '0').fillna('0')
        df_business = df_business.replace(True, '1').replace(False, '0').replace('None','0')
        df_business = df_business.drop(0, axis=1)

        avoid_cols = ['business_id', 'city'] + dict_cols + unicode_cols
        transform_cols = [x for x in df_business.columns if x not in avoid_cols]
        for col in tqdm(transform_cols):
            df_business[col] = df_business[col].fillna(0)
            df_business[col] = df_business[col].astype(int)

        df_business['WiFi'] = df_business['WiFi'].replace('0', 'no specify')
        df_business['Alcohol'] = df_business['Alcohol'].replace('0', 'no specify')
        df_business['RestaurantsAttire'] = df_business['RestaurantsAttire'].replace(0, 'no specify')
        df_business['NoiseLevel'] = df_business['NoiseLevel'].replace('0', 'no specify')
        df_business['Smoking'] = df_business['Smoking'].replace('0', 'no specify')
        df_business['BYOBCorkage'] = df_business['BYOBCorkage'].replace('0', 'no specify')
        df_business['AgesAllowed'] = df_business['AgesAllowed'].replace('0', 'no specify')

        cat_cols = ['WiFi', 'Alcohol', 'RestaurantsAttire', 'NoiseLevel', 'Smoking', 'BYOBCorkage', 'AgesAllowed']
        df_business = pd.get_dummies(df_business, columns=cat_cols)
    else:
        df_business= pd.read_pickle('df_business_transformed.pkl')
    
    return df_business

def extract_user_features(df_review, df_business, similarity_users, user_id, K_users):
    
    users_id = list(similarity_users.head(K_users+1)['user_id'])
    df_reviews_user, df_non_seen_items = pp.filter_by_user(df_review, users_id, True)
    df_non_seen_items = df_non_seen_items[['business_id']].drop_duplicates()

    df_business_features = transform_business_features(df_business, reprocess = False)
    df_reviews_user = df_reviews_user[['business_id', 'review_stars']].merge(df_business_features, on='business_id', how='left')
    
    cols_X = [x for x in df_reviews_user.columns if x not in ['business_id', 'review_stars', 'city', 'stars']]
    X, y = df_reviews_user[cols_X], df_reviews_user['review_stars']
    
    pesos_features, pval= chi2(X, y)
    pesos_features = np.nan_to_num(pesos_features)
    pesos_features_mask = pesos_features>0
    max_f = np.argsort(pesos_features)[-10:]
    important_features = [x.replace("_'none'", " Free") for x in X.columns[max_f]]
    X = X[X.columns[pesos_features_mask]]
    
    df_non_seen_items = df_non_seen_items.merge(df_business_features, on='business_id', how='left')
    X_rec, y_rec = df_non_seen_items[X.columns], df_non_seen_items[['business_id']]
    
    return X, y, X_rec, y_rec, important_features

def generate_recommendations(X, y, X_rec, y_rec, df_business, K_rec):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = KNeighborsClassifier(2)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    prec, recall, fscore, supp = precision_recall_fscore_support(y_test, y_pred)
    
    predictions = model.predict(X_rec)
    df_pred = pd.DataFrame(data = y_rec, columns=['business_id'])
    df_pred.loc[:, ['review_stars_predicted']] = predictions
    
    df_pred = df_pred.sort_values(by='review_stars_predicted', ascending=False).head(K_rec)
    df_pred = df_pred.merge(df_business, on='business_id', how='left')
    df_pred = df_pred[['name', 'address', 'city', 'state', 'review_stars_predicted', 'latitude', 'longitude']]
    return df_pred