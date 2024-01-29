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

def load_dataset():
    
    dataset_path = 'dataset/'
    df_business = pd.read_json(dataset_path + 'business.json', lines=True)
    
    size = 200000
    df_reviews = pd.read_json(dataset_path + 'review.json', lines=True, 
                          dtype={'review_id':str,'user_id':str,
                                 'business_id':str,'stars':int,
                                 'date':str,'text':str,'useful':int,
                                 'funny':int,'cool':int},
                          chunksize=200000, nrows=800000)
    reviews_list = []
    for df_review in tqdm(df_reviews):
        df_review = df_review.drop(['review_id','useful','funny','cool'], axis=1)
        df_review = df_review.rename(columns={'stars': 'review_stars'})
        df_review_m = pd.merge(df_business, df_review, on='business_id', how='inner')
        reviews_list.append(df_review_m)
    df_review = pd.concat(reviews_list, ignore_index=True, join='outer', axis=0)
    
    df_users = pd.read_json(dataset_path + 'user.json', lines=True, 
                          chunksize=200000, nrows=2000000)
    users_list = []
    for df_user in tqdm(df_users):
        users_list.append(df_user[['user_id', 'name', 'review_count', 'yelping_since']])

    df_users = pd.concat(users_list, ignore_index=True, join='outer', axis=0)
    
    return df_review, df_business, df_users

def filter_by_user(df_review, user_ids, non_seen_items):
    user_cities = list(df_review[df_review['user_id'].isin(user_ids)]['city'].values)
    user_items = list(set(list(df_review[df_review['user_id'].isin(user_ids)]['business_id'])))
    reviews_user = df_review[(df_review['city'].isin(user_cities)) & (df_review['business_id'].isin(user_items))]
    if non_seen_items:
        user_cities = list(df_review[df_review['user_id'].isin([user_ids[0]])]['city'].values)
        user_items = list(set(list(df_review[df_review['user_id'].isin([user_ids[0]])]['business_id'])))
        non_seen_items = df_review[(df_review['city'].isin(user_cities)) & (df_review['business_id'].isin(user_items) == False)]
        return reviews_user, non_seen_items[['business_id', 'user_id', 'review_stars']]
    else:
        return reviews_user, None

def cosine_similarity(matrix):
    return 1-pairwise_distances(matrix, metric="cosine")

def get_similarity_users(df_reviews_user, user_id):
    
    reviews_user = df_reviews_user[['user_id', 'business_id', 'review_stars']].drop_duplicates()
    review_user_matrix = reviews_user.pivot_table(values='review_stars', index='user_id', columns='business_id').fillna(0)
    idx = list(review_user_matrix.index)
    cosine_sim = cosine_similarity(review_user_matrix)
    cosine_sim_matrix = pd.DataFrame(data = cosine_sim, index = idx, columns = idx)
    user_sim = cosine_sim_matrix.filter(items=[user_id], axis=0)
    most_sim_k_users = user_sim.max().rename_axis('user').reset_index().sort_values(by=0, ascending=False)
    most_sim_k_users.columns = ['user_id', 'similarity']
    
    return most_sim_k_users