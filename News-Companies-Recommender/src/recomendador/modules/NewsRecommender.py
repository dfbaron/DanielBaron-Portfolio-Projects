import pandas as pd

class NewsRecommender:
    def __init__(self, df_prediction) -> None:
        self.df_prediction = df_prediction

    def classify_participation(self, row):
        if row['user_mentioned']  and row['sector_mentioned']:
            return 'Cliente'
        elif not row['user_mentioned']  and row['sector_mentioned']:
            return 'Sector'
        else:
            return 'No aplica'

    def classify_category(self, row):
        if row['participacion'] == 'No aplica':
            return 'descartable'
        elif row['classification_probability']>=row['lexicon_probability']:
            return row['classification_category']
        else:
            return row['lexicon_category']

    def assign_category_weight(self, row):
        if row['categoria'] == 'macroeconomia':
            return 2.0
        elif row['categoria'] == 'sostenibilidad':
            return 1.8
        elif row['categoria'] == 'innovacion':
            return 1.7
        elif row['categoria'] == 'regulaciones':
            return 1.6
        elif row['categoria'] == 'alianzas':
            return 1.4
        elif row['categoria'] == 'reputacion':
            return 1.2
        elif row['categoria'] == 'otra':
            return 0.5
        elif row['categoria'] == 'descartable':
            return 0.0

    def generate_recommendations(self):
        self.df_prediction['participacion'] = self.df_prediction.apply(self.classify_participation, axis=1)
        self.df_prediction['categoria'] = self.df_prediction.apply(self.classify_category, axis=1)
        self.df_prediction['peso_categoria'] = self.df_prediction.apply(self.assign_category_weight, axis=1)
        self.df_prediction['nombre_equipo'] = 'DKR'
        final_columns_r = ['nombre_equipo', 'nit', 'news_id', 'participacion', 'categoria', 'recomendacion']
        final_columns_c = ['nombre_equipo', 'nit', 'news_id', 'participacion', 'categoria']
        recommendations = pd.DataFrame(data=[], columns=final_columns_r)
        categorization = self.df_prediction[final_columns_c]
        self.df_prediction['recomendacion'] = 0
        for user in self.df_prediction['nit'].drop_duplicates():
            df_prediction_tmp = self.df_prediction[(self.df_prediction['nit']==user) & (self.df_prediction['categoria']!='descartable')]
            df_prediction_tmp = df_prediction_tmp.sort_values(by = ['nit', 'peso_categoria', 'sentiment_probability'], ascending=[False, False, False])
            df_prediction_tmp['recomendacion'] = [i for i in range(1, len(df_prediction_tmp)+1)]
            recommendations = pd.concat([recommendations, df_prediction_tmp[final_columns_r]])
        return recommendations, categorization