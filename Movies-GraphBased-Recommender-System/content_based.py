import re, unicodedata
import contractions
import inflect
import pandas as pd

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def preproccesing(words):
    words = to_lowercase(words)
    #words = replace_numbers(words)
    words = remove_punctuation(words)
    words = remove_non_ascii(words)
    words = remove_stopwords(words)
    return words

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def stem_and_lemmatize(words):
    words = stem_words(words)
    words = lemmatize_verbs(words)
    return words

def preprocess_train(words):

    words = words.apply(contractions.fix)
    words = words.apply(word_tokenize)
    words = words.apply(preproccesing)
    #words = words.apply(stem_and_lemmatize)
    words = words.apply(lambda x: ' '.join(map(str, x)))

    vectorizer = TfidfVectorizer()
    words = vectorizer.fit_transform(words)
    words = scipy.sparse.csr_matrix.todense(words)

    return words, vectorizer.vocabulary_

def preprocess_test(words, vocabulary):

    words = words.apply(contractions.fix)
    words = words.apply(word_tokenize)
    words = words.apply(preproccesing)
    #words = words.apply(stem_and_lemmatize)
    words = words.apply(lambda x: ' '.join(map(str, x)))

    vectorizer = TfidfVectorizer(vocabulary=vocabulary)
    words = vectorizer.fit_transform(words)
    words = scipy.sparse.csr_matrix.todense(words)

    return words

def get_recommendations(movies, ratings, user_id, K_rec):
    
    movies_cb = movies[['movie_id', 'movie_name', 'abstract', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                        'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 
                        'Sci-Fi', 'Thriller', 'War', 'Western', '(no genres listed)']]
    
    words, vocabulary = preprocess_train(movies_cb['abstract'])
    
    movies_cb = pd.concat([movies_cb, pd.DataFrame(words)], axis=1)
    
    ratings_u = ratings[ratings['user_id']==user_id]
    dataset = ratings_u.merge(movies_cb, on='movie_id', how='inner')
    
    x_cols = [x for x in dataset.columns if x not in ['user_id','movie_id','rating','timestamp','movie_name','abstract']]
    X, y = dataset[x_cols], dataset[['rating']]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    KNNR = KNeighborsRegressor(n_neighbors=2)
    KNNR.fit(X_train, y_train)
    
    y_pred = KNNR.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print('RMSE: {}'.format(rmse))
    
    user_movies = list(ratings[ratings['user_id']==user_id]['movie_id'].drop_duplicates())
    df_non_seen_items = ratings[ratings['movie_id'].isin(user_movies) == False].sample(1000)
    
    movies_cb = movies[['movie_id', 'movie_name', 'abstract', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 
                    'Sci-Fi', 'Thriller', 'War', 'Western', '(no genres listed)']]
    df_non_seen_items = df_non_seen_items.merge(movies_cb, on='movie_id', how='inner')
    
    words_ns = preprocess_test(df_non_seen_items['abstract'], vocabulary)
    
    df_non_seen_items = pd.concat([df_non_seen_items, pd.DataFrame(words_ns)], axis=1)
    
    X_non_seen = df_non_seen_items[X_train.columns]
    new_pred = KNNR.predict(X_non_seen)
    df_non_seen_items['prediction'] = new_pred
    
    recommendations_content_based = df_non_seen_items.sort_values(by='prediction', ascending =False)[['movie_id']].head(K_rec)
    
    return recommendations_content_based