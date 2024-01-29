import pandas as pd
from nltk.corpus import stopwords
from glob import glob
from tqdm import tqdm
import numpy as np

class UserMentionTagger:
    def __init__(self, df_clientes, df_noticias, df_clientes_noticias):
        self.df_clientes = df_clientes 
        self.df_noticias = df_noticias
        self.df_clientes_noticias = df_clientes_noticias

    def preprocess_news(self):
        self.df_noticias['complete_news'] = self.df_noticias['news_title'] + ' ' + self.df_noticias['news_text_content']
        df_news_user = self.df_clientes_noticias[['nit', 'news_id']].merge(self.df_clientes, how='left', on='nit')
        df_news_user = df_news_user.merge(self.df_noticias[['news_id', 'complete_news']], how='left', on='news_id')
        self.df_news_user = df_news_user

    def n_grams(self, new, N):
        words = new.split()
        return [' '.join(x) for x in [words[idx:idx+N] for idx in range(len(words)-N + 1)]]

    def clean_news(self, text):
        text_c = str(text).lower()
        text_l = text_c.split()
        text_c = ' '.join([x for x in text_l if x.isalpha()])
        return text_c

    def clean_names(self, text, stop_w):
        text_c = str(text).lower()
        text_l = text_c.split()
        del_words = stop_w
        text_c = ' '.join([x for x in text_l if x.isalpha() and x not in del_words])
        return text_c

    def most_frequent(self, List):
        return max(set(List), key = List.count)

    def match_news_users(self, new, user, stop_w):
        new = self.clean_news(new)
        user = self.clean_names(user, stop_w)
        n_max = np.max((1, len(user.split())))
        votation = []
        for i in range(1, n_max+1):
            grams = self.n_grams(user, i)
            number_grams = len([1 for x in grams if ' {} '.format(x) in new])>0
            k = np.min((i, 2))
            votation += [number_grams for j in range(k)]
        return self.most_frequent(votation)

    def tag_user_mentions(self):
        self.preprocess_news()
        stop_w = stopwords.words('spanish')
        self.df_news_user['user_mentioned'] = self.df_news_user[['complete_news', 'nombre']].apply(lambda x: self.match_news_users(x['complete_news'], x['nombre'], stop_w), axis=1)
        return self.df_news_user[['nit', 'news_id', 'user_mentioned']]