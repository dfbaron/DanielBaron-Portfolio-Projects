from pyexpat import features
from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import pandas as pd
import numpy as np
import json
import folium


import time

from sklearn.model_selection import train_test_split
from scipy import spatial

from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
from sklearn.feature_selection import chi2

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import precision_recall_fscore_support

import User_User_RS as uu_rs
import Content_Based_RS as cb_rs
import preprocessing as pp


df_review, df_business, df_users = pp.load_dataset()
print(df_review["user_id"].value_counts())

def combine_recommendations(user_id, K_rec):
    K_sim_user = 10
    df_reviews_user, df_non_seen_items = pp.filter_by_user(df_review, [user_id], True)
    similarity_users = pp.get_similarity_users(df_reviews_user, user_id)
    
    X, y, X_rec, y_rec, imp_feat = cb_rs.extract_user_features(df_review, df_business, similarity_users, user_id, K_sim_user)
    cb_recommendations = cb_rs.generate_recommendations(X, y, X_rec, y_rec, df_business, K_rec)
    
    uu_recommendations, imp_user = uu_rs.generate_recommendations(df_review, df_reviews_user, df_business, df_non_seen_items, \
                                                        similarity_users, user_id, K_sim_user, K_rec)

    uu_recommendations.columns = ['name', 'address', 'city', 'state', 'review_stars_predicted', 'latitude', 'longitude']
    cb_recommendations.columns = ['name', 'address', 'city', 'state', 'review_stars_predicted', 'latitude', 'longitude']
    
    recommendations = pd.concat([cb_recommendations, uu_recommendations]) 
    return recommendations, imp_feat, imp_user



app = Flask(__name__)
CORS(app)


@app.route("/")
def hello_from_root():
    return jsonify(message = "Hello from Root")



@app.route("/get_usuario/<id>", methods= ["POST", "GET"])
def get_usuario_df(id):
    print(id)

    user_info = df_users[df_users['user_id'] == id][['name', 'review_count', 'yelping_since']]

    return user_info.to_json(orient="records")




@app.route("/get_business/<id>", methods=["POST", "GET"])
def get_business(id):
    print(request.json)

    return request.json




@app.route("/get_recomendaciones/<id>", methods=["POST", "GET"])
def get_recomendaciones(id):
    print(id)

    K_rec = 10
    recommendations, imp_feat, imp_user = combine_recommendations(id, K_rec)
    imp_user = df_users[df_users['user_id'].isin(imp_user)][['name', 'review_count', 'yelping_since']]

    #recommendations = [{"name": "1", "latitude":4.713991455266561, "longitude": -74.0299935}, 
                        #{"name": "2", "latitude":4.705394596794235, "longitude": -74.03334089677242}]

    #imp_feat = ["Ford", "Ford", "Ford"]
    #imp_user = [{"model": "Mustang"}, {"model": "Mustang"}, {"model": "Mustang"}]
    print(recommendations)
    return jsonify(recommendaciones=recommendations.to_json(orient="records"), features= imp_feat, usuarios = imp_user.to_json(orient="records"))


@app.route("/get_mapa", methods=["POST"])
def get_mapa():
    print(request.json)
    recommendations = pd.DataFrame(data=request.json)
    print(recommendations)


    m = folium.Map(location=[recommendations['latitude'].mean(), recommendations['longitude'].mean()], \
               tiles='Stamen Terrain'
    )
    for index, row in recommendations.iterrows():
        name = row['name']
        lat, long = row['latitude'], row['longitude']
        folium.Marker([lat, long], popup=name, tooltip=name).add_to(m)

    print(m)
    m.save("index.html")

    with open("index.html", "r") as file:
        data = file.read().replace("\n", "")

    print(type(data))
    

    return data