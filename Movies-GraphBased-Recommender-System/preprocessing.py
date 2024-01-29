import pandas as pd
from tqdm import tqdm
import requests
import preprocessing as pp

def load_datasets():
    movies = pd.read_excel('dataset/movies.xlsx')
    ratings = pd.read_csv('dataset/ratings.csv')
    movies_add_info = pd.read_csv('dataset/movies_additional_info.csv', sep='|')
    ratings.columns = ['user_id', 'movie_id', 'rating', 'timestamp']


    movies['movie_id'] = movies['movie_id'].astype(int)
    movies = movies.merge(movies_add_info, on='movie_id', how='inner')

    ratings = ratings[ratings['movie_id'].isin(list(movies['movie_id'])) ]

    movies = movies.drop('movie_link', axis=1)

    genre_columns = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary",                 "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",                 "Thriller", "War", "Western", "(no genres listed)"
                    ]
    for genre_column in tqdm(genre_columns):
        movies[genre_column] = movies.apply(lambda x: pp.set_genres(genre_column, x['genres']), axis=1)
    movies.drop(['genres'], axis = 1, inplace=True)
    movies = movies.fillna('no specify').replace('', 'no specify')
    return ratings, movies

def extract_ontological_movies_information():
    movie_links = pd.read_csv('dataset/MappingMovielens2DBpedia.tsv', sep='\t', header=None)
    movie_links.columns = ['movie_id', 'movie_name', 'movie_link']

    extracted_prop = ['http://dbpedia.org/ontology/abstract',
                      'http://dbpedia.org/ontology/director',
                      'http://dbpedia.org/ontology/distributor',
                      'http://dbpedia.org/ontology/starring',
                      'http://dbpedia.org/ontology/musicComposer',
                      'http://dbpedia.org/ontology/writer',
                      'http://dbpedia.org/ontology/cinematography',
                      'http://dbpedia.org/ontology/producer']

    cols = ['abstract','director','distributor','starring','musicComposer','writer','cinematography','producer']
    movie_links[cols] = ''
    errors = []
    for idx, row in tqdm(movie_links.iterrows(), total=movie_links.shape[0]):
        try:
            movie_name = row['movie_link'].split('/')[-1]
            data = requests.get('http://dbpedia.org/data/{}.json'.format(movie_name)).json()
            movie_data = data[row['movie_link']]
            for prop in extracted_prop:
                if prop in list(movie_data.keys()):
                    prop_name = prop.split('/')[-1]
                    prop_info = movie_data[prop]
                    if prop == 'http://dbpedia.org/ontology/abstract':
                        movie_links[prop_name].iloc[idx] = [x['value'] for x in prop_info if x['lang'] == 'en'][0]
                    else:
                        if len(prop_info)==1:
                            value = [x['value'] for x in prop_info][0].split('/')[-1].replace('_', ' ')
                            movie_links[prop_name].iloc[idx] = value
                        else:
                            value = ', '.join([x['value'].split('/')[-1].replace('_', ' ') for x in prop_info])
                            movie_links[prop_name].iloc[idx] = value
                else:
                    movie_links[prop_name].iloc[idx] = 'no specify'
        except:
            errors.append(idx)

    movie_links = movie_links.drop(errors)        
    movie_links.to_csv('dataset/movies_additional_info.csv', sep='|', index=False)
    return movie_links

def set_genres(genres,col):
    if genres in col.split('|'): return 1
    else: return 0