import pandas as pd
from pysentimiento import create_analyzer

class SentimentAnalysis:
    def __init__(self, df_noticias):
        self.df_noticias = df_noticias

    def preprocess_news(self):
        print('News preprocessed...')
        self.df_noticias['text'] = self.df_noticias['news_title'] + ' ' + self.df_noticias['news_text_content']

    def initialize_model(self):
        print('Model loaded...')
        self.analyzer = create_analyzer(task="sentiment", lang="es")

    def generate_sentiment_analysis(self):
        print('Evaluating sentiment analysis on news...')
        self.df_noticias['sentiment'] = self.analyzer.predict(self.df_noticias['text'])
        self.df_noticias['sentiment_label'] = self.df_noticias['sentiment'].apply(lambda x: x.output)
        self.df_noticias['sentiment_probability'] = self.df_noticias['sentiment'].apply(lambda x: x.probas[x.output])
        self.df_noticias.loc[self.df_noticias['sentiment_label']=='NEU', 'sentiment_probability'] = 0
        return self.df_noticias[['news_id', 'sentiment_label', 'sentiment_probability']]