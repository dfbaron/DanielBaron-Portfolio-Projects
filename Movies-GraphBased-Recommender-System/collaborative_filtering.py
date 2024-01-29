import pandas as pd

import numpy as np

from sklearn.metrics import pairwise_distances

def cosine_similarity(matrix):
    return 1-pairwise_distances(matrix, metric="cosine")

def get_similarity_users(ratings_user, user_id, K):
    
    sim_users = pd.DataFrame(data=[], columns = ['user_id', 'similarity'])
    cond = True
    while cond:
        reviews_user = ratings_user[['user_id', 'movie_id', 'rating']].drop_duplicates().sample(10000)
        review_user_matrix = reviews_user.pivot_table(values='rating', index='user_id', columns='movie_id').fillna(0)
        idx = list(review_user_matrix.index)
        cosine_sim = cosine_similarity(review_user_matrix)
        cosine_sim_matrix = pd.DataFrame(data = cosine_sim, index = idx, columns = idx)
        user_sim = cosine_sim_matrix.filter(items=[user_id], axis=0)
        most_sim_k_users = user_sim.max().rename_axis('user').reset_index().sort_values(by=0, ascending=False)
        most_sim_k_users.columns = ['user_id', 'similarity']
        non_nan_sim_users = most_sim_k_users[most_sim_k_users['similarity'].fillna(-1) != -1]
        sim_users = pd.concat([sim_users, non_nan_sim_users])
        if len(sim_users)>=K:
            cond = False
        
    return most_sim_k_users.head(K)

def get_recommendations(ratings, user_id, K_rec):
    
    # Extract most similar users that watch the same movies that the user
    user_movies = list(ratings[ratings['user_id']==user_id]['movie_id'].drop_duplicates())
    ratings_user = ratings[ratings['movie_id'].isin(user_movies)]
    K = 50
    sim_users = get_similarity_users(ratings_user, user_id, K)
    
    means_user_ratings = ratings_user[['user_id', 'rating']].groupby('user_id').mean().rename_axis('user_id').reset_index()
    sim_users = sim_users.merge(means_user_ratings, on='user_id', how='left')
    
    top_sim_user = list(sim_users['user_id'])
    df_non_seen_items = ratings[ratings['movie_id'].isin(user_movies) == False]
    df_items = df_non_seen_items[df_non_seen_items['user_id'].isin(top_sim_user)][['movie_id', 'user_id', 'rating']]
    df_items = df_items.pivot_table(values='rating', index='user_id', columns='movie_id').rename_axis('user_id').reset_index()
    
    sim_users = sim_users.merge(df_items, on='user_id', how='left').fillna(0)
    
    unseen_items = list(df_non_seen_items['movie_id'].drop_duplicates())
    df_recommendations = pd.DataFrame(data = unseen_items, columns= ['movie_id'])
    df_recommendations['prediction'] = 0
    
    K_sim_user = 10
    ra = sim_users[sim_users['user_id']==user_id]['rating'].values[0]
    cols = sim_users.columns
    users = []
    
    for unseen_item in unseen_items:
        if unseen_item in cols:
            users = users + list(sim_users[sim_users[unseen_item]!=0]['user_id'].values)
            sample = sim_users[sim_users[unseen_item]!=0].head(K_sim_user+1).tail(K_sim_user)
            num = np.dot(sample['similarity'], (sample[unseen_item]-sample['rating']))
            den = sum(sample['similarity'])
            ri = ra + num/den
            df_recommendations.loc[df_recommendations['movie_id']==unseen_item, ['prediction']] = ri
    df_recommendations = df_recommendations.sort_values(by='prediction', ascending=False)
    return sim_users[['user_id']].head(K_rec), df_recommendations.head(K_rec)