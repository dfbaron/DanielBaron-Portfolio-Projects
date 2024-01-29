# -*- coding: utf-8 -*-

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def get_recommendations(movies, ratings, recommendations_user_user, recommendations_content_based):
    starring = movies[['movie_name', 'starring']]
    director = movies[['movie_name', 'director']]
    distributor = movies[['movie_name', 'distributor']]
    music_composer = movies[['movie_name', 'musicComposer']]
    writer = movies[['movie_name', 'writer']]
    cinematography = movies[['movie_name', 'cinematography']]
    
    
    starring.columns = ['source', 'target']
    director.columns = ['source', 'target']
    distributor.columns = ['source', 'target']
    music_composer.columns = ['source', 'target']
    writer.columns = ['source', 'target']
    cinematography.columns = ['source', 'target']
    
    writer = writer.set_index('source')['target'].str.split(',\s*', expand=True).stack().reset_index(name='target')                   .drop('level_1',1)
    
    starring = starring.set_index('source')['target'].str.split(',\s*', expand=True).stack().reset_index(name='target')                   .drop('level_1',1)
    
    starring['target'] =  'Starring: ' + starring['target']
    director['target'] =  'Director: ' + director['target']
    distributor['target'] =  'Distributor: ' + distributor['target']
    music_composer['target'] =  'Music_composer: ' + music_composer['target']
    writer['target'] =  'Writer: ' + writer['target']
    cinematography['target'] =  'Cinematography: ' + cinematography['target']
    
    df_nodes_movies = pd.concat([starring, director, distributor, music_composer, writer, cinematography], axis=0,                             ignore_index=True)
    df_nodes_movies = df_nodes_movies[df_nodes_movies['target'] != 'no specify']
    
    ratings_n = ratings.merge(movies[['movie_id', 'movie_name']], on='movie_id', how='inner')
    df_nodes_user_movies = ratings_n[['user_id', 'movie_name']]
    df_nodes_user_movies.columns = ['source', 'target']
    
    df_nodes = pd.concat([df_nodes_user_movies, df_nodes_movies])
    
    recommendations = pd.concat([recommendations_user_user[['movie_id']], recommendations_content_based])
    
    rec_movies = recommendations.merge(movies[['movie_id', 'movie_name']], on='movie_id', how='inner')
    
    targets = list(rec_movies['movie_name'])
    df_nodes_u = df_nodes[(df_nodes['source'].isin(targets)) | (df_nodes['target'].isin(targets))]
    
    G = nx.from_pandas_edgelist(df_nodes_u, "source", "target", create_using=nx.MultiDiGraph())
    
    pr = nx.pagerank(G, alpha=0.8, max_iter=5000)
    
    K_final_rec = 20
    movies_rec = [x for x in pr.keys() if not(str(x).isnumeric()) and ':' not in x][:K_final_rec]
    
    df_rec = df_nodes[df_nodes['source'].isin(movies_rec)]
    
    return movies_rec, df_rec

def gen_figure(recom):
    G_rec = nx.from_pandas_edgelist(recom, "source", "target", create_using=nx.MultiDiGraph())

    plt.figure(figsize=(30,30))

    color_map = []
    for node in G_rec:
        if str(node).isnumeric():
            color_map.append('blue')
        elif ':' in node:
            color_map.append('green')
        else: 
            color_map.append('red')  

    pos = nx.spring_layout(G_rec)
    nx.draw(G_rec, with_labels=True, node_color=color_map, edge_cmap=plt.cm.Blues, pos = pos)
    plt.savefig('recommended_movies_graph.jpg')