from modules.LexiconCategories import LexiconCategories
from modules.ClassificationModel import ClassificationModel
from modules.UserMentionTagger import UserMentionTagger
from modules.SentimentAnalysis import SentimentAnalysis
from modules.NewMentionSector import NewMentionSector
from modules.NewsRecommender import NewsRecommender
import pandas as pd

#Loading files
df_clientes = pd.read_csv('../data/archivos_auxiliares/clientes.csv').head(100)
df_noticias = pd.read_csv('../data/archivos_auxiliares/noticias.csv').head(100)
df_clientes_noticias = pd.read_csv('../data/archivos_auxiliares/clientes_noticias.csv').head(100)
train_data = pd.read_csv('../data/archivos_auxiliares/scrapped_news.csv')

# Lexicon for category classification
lexC = LexiconCategories(df_noticias)
lexC.create_lexicon('category_lexicon')
df_news_lexicon = lexC.evaluate_news_lexicon(df_noticias)

# Model to determine if an user is mentioned in a new 
userMT = UserMentionTagger(df_clientes, df_noticias, df_clientes_noticias)
df_news_user_tag = userMT.tag_user_mentions()

# Classification model with scrapped news
cm = ClassificationModel(train_data, df_noticias)
cm.train_model()
df_news_class_model = cm.predict()

# Sentiment analysis to determine if on a new is a positive or negative sentiment
sa = SentimentAnalysis(df_noticias)
sa.preprocess_news()
sa.initialize_model()
df_news_sent_anal = sa.generate_sentiment_analysis()

# Evaluating if a user's sector is mentioned in the new
nms = NewMentionSector(df_clientes, df_noticias, df_clientes_noticias)
df_news_sector_men = nms.evaluate_news()

# Join all the information
print('Joining all the information...')
df_prediction = df_news_lexicon.merge(df_news_user_tag, how='inner', on='news_id')
df_prediction = df_prediction.merge(df_news_sector_men, how='inner', on=['news_id', 'nit'])
df_prediction = df_prediction.merge(df_news_class_model, how='inner', on='news_id')
df_prediction = df_prediction.merge(df_news_sent_anal, how='inner', on='news_id')
order_columns = ['nit', 'news_id', 'user_mentioned', 'sector_mentioned', 'lexicon_category', 'lexicon_probability', 'classification_category', \
                 'classification_probability', 'sentiment_label', 'sentiment_probability']
df_prediction = df_prediction[order_columns]

# Generate recommendations and categorizations for the different news
nr = NewsRecommender(df_prediction)
df_recommendations, df_categorization = nr.generate_recommendations()

# Exporting information
print('Exporting the information...')
df_recommendations.to_csv('../data/output/recomendacion.csv')
df_categorization.to_csv('../data/output/categorizacion.csv')
