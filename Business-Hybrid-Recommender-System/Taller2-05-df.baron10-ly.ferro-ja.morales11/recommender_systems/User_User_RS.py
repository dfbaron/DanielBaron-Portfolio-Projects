#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
from tqdm import tqdm
import time

from scipy import spatial

import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
from sklearn.feature_selection import chi2

import preprocessing as pp

def generate_recommendations(df_review, df_reviews_user, df_business, df_non_seen_items, similarity_users, user_id, K_sim_user, K_rec):
    
    means_user_ratings = df_reviews_user[['user_id', 'review_stars']].groupby('user_id').mean().rename_axis('user_id').reset_index()
    similarity_users = similarity_users.merge(means_user_ratings, on='user_id', how='left')
    
    top_sim_user = list(similarity_users['user_id'])
    df_items = df_non_seen_items[df_non_seen_items['user_id'].isin(top_sim_user)][['business_id', 'user_id', 'review_stars']]
    df_items = df_items.pivot_table(values='review_stars', index='user_id', columns='business_id').rename_axis('user_id').reset_index()
    
    similarity_users = similarity_users.merge(df_items, on='user_id', how='left').fillna(0)
    
    unseen_items = list(df_non_seen_items['business_id'].drop_duplicates())
    df_recommendations = pd.DataFrame(data = unseen_items, columns= ['item'])
    df_recommendations['prediction'] = 0
    
    ra = similarity_users[similarity_users['user_id']==user_id]['review_stars'].values[0]
    cols = similarity_users.columns
    users = []
    for unseen_item in unseen_items:
        if unseen_item in cols:
            users = users + list(similarity_users[similarity_users[unseen_item]!=0]['user_id'].values)
            sample = similarity_users[similarity_users[unseen_item]!=0].head(K_sim_user+1).tail(K_sim_user)
            num = np.dot(sample['similarity'], (sample[unseen_item]-sample['review_stars']))
            den = sum(sample['similarity'])
            ri = ra + num/den
            df_recommendations.loc[df_recommendations['item']==unseen_item, ['prediction']] = ri
    df_recommendations = df_recommendations.merge(df_business, left_on='item', right_on='business_id', how='left')
    df_recommendations = df_recommendations.sort_values(by='prediction', ascending=False)
    df_recommendations = df_recommendations[['name', 'address', 'city', 'state', 'prediction', 'latitude', 'longitude']]
    return df_recommendations.head(K_rec), users