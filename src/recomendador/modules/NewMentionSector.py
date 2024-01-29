from sentence_transformers import SentenceTransformer, util
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

class NewMentionSector:
    def __init__(self, df_clientes, df_noticias, df_clientes_noticias):
        self.df_clientes = df_clientes 
        self.df_noticias = df_noticias
        self.df_clientes_noticias = df_clientes_noticias

    def preprocess_news(self):
        print('Preprocessing news...')
        self.df_noticias['complete_news'] = self.df_noticias['news_title'] + ' ' + self.df_noticias['news_text_content']
        df_news_user = self.df_clientes_noticias[['nit', 'news_id']].merge(self.df_clientes, how='left', on='nit')
        df_news_user = df_news_user.merge(self.df_noticias[['news_id', 'complete_news']], how='left', on='news_id')
        self.df_news_user = df_news_user

    def clean_news(self, text):
        text_c = str(text).lower()
        text_l = text_c.split()
        text_c = ' '.join([x for x in text_l if x.isalpha()])
        return text_c

    def initiate_model(self):
        print('Loading transformer model...')
        self.model = SentenceTransformer('hiiamsid/sentence_similarity_spanish_es')

    def sentence_similarity(self, new, sector):
        new_c = str(new).lower()
        sector_c = str(sector).lower()
        embedding_new = self.model.encode(new_c, convert_to_tensor=True)
        embedding_sector = self.model.encode(sector_c, convert_to_tensor=True)
        cos_sim = util.pytorch_cos_sim(embedding_new, embedding_sector)
        return True if cos_sim>0.5 else False

    def evaluate_news(self):
        self.preprocess_news()
        self.initiate_model()
        print('Evaluating news...')
        self.df_news_user['complete_news'] = self.df_news_user['complete_news'].apply(self.clean_news)
        self.df_news_user['sector_mentioned'] = self.df_news_user.progress_apply(lambda x: self.sentence_similarity(x['complete_news'], x['desc_ciiu_division']), axis=1)
        return self.df_news_user[['nit', 'news_id', 'sector_mentioned']]
