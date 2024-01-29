
import collaborative_filtering as cf
import preprocessing as pp
import content_based as cb
import graph_based as gb

user_id = 183233
K_rec = 50

#Load datasets
ratings, movies = pp.load_datasets()
print('Datasets loaded.')

def generate_recommendations(user_id, K_rec, ratings, movies):
    
    #Generate collaborative recommendations
    sim_users, recommendations_user_user = cf.get_recommendations(ratings, user_id, K_rec)
    print('Collaborative filtering recommendations created.')
    
    #Generate content based recommendations
    recommendations_content_based = cb.get_recommendations(movies, ratings, user_id, K_rec)
    print('Content based recommendations created.')
    
    #Rank recommendations generated with CB and CF using graphs and PageRank
    movies_rec, df_rec = gb.get_recommendations(movies, ratings, recommendations_user_user, recommendations_content_based)
    print('Recommendations ranked using PageRank.')
    
    #Generation of figure corresponding to user movie recommendations graph
    gb.gen_figure(df_rec)
    print('Graph created.')
    
    return movies_rec

recommendations = generate_recommendations(user_id, K_rec, ratings, movies)
print(recommendations)