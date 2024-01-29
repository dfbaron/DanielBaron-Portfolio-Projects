import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tqdm import tqdm
import gensim
import pandas as pd
import numpy as np
import json
import os
import string
import unidecode
from scipy.special import softmax

class LexiconCategories:
    '''
    Parameters
    ----------
    train_data : DataFrame
        El parametro de entrada para esta función se realiza es un DataFrame
        que contenga noticias relacionadas con los temas especificos a las 
        categorias que se desean clasificar.
    stemming : bool
        Parámetro que define si al preprocesar las noticias se realiza stemming.

    Returns
    -------
        Dataframe que contiene el id de la noticia, la categoria asociada a cada noticia
        y su probabilidad asociada.

    '''
    def __init__(self, train_data, stemming=True):
        self.train_data = train_data
        self.stemming = stemming

    def clean_news(self, new):
        """
        Función para limpiar las noticias (eliminar puntuación, hacer stemming de acuerdo 
        al parametro especificado y pasar todo a minuscula).
        """
        new = str(new)
        if self.stemming:
            ps = PorterStemmer()
            new_c = ' '.join([ps.stem(word.lower()) for word in new.split()])
        else:
            new_c = new.lower()
        return new_c.translate(str.maketrans('', '', string.punctuation))
        
    def count_words_in_new(self, new, words_sims):
        counter = 0
        for word_sim in words_sims:
            word = word_sim[0]
            sim = word_sim[1]
            count_word = new.count(word)
            counter += count_word*sim
        return counter

    def generate_sentences(self):
        print('Generating sentences...')
        self.train_data['news_title_content'] = self.train_data['news_title'] + ' ' + self.train_data['news_text_content']
        self.train_data['news_title_content'] = self.train_data['news_title_content'].map(self.clean_news)
        sentences = list(self.train_data['news_title_content'])
        sentences = [unidecode.unidecode(x) for x in sentences]
        self.sentences = [x.translate(str.maketrans('', '', string.punctuation)).split() for x in sentences]

    def train_model(self):
        if self.stemming:
            self.model_path = "../data/archivos_auxiliares/models/{}_stemming.model".format(self.lexicon_name)
        else:
            self.model_path = "../data/archivos_auxiliares/models/{}_no_stemming.model".format(self.lexicon_name)
        if os.path.exists(self.model_path):
            model = gensim.models.Word2Vec.load(self.model_path)
            print('Model exists. Loading model...')
        else:
            print('Model does not exists. Training model...')
            model = gensim.models.Word2Vec(self.sentences, vector_size = 20,  min_count=2, sg=1)
            nrEpochs = 20
            for epoch in range(1, nrEpochs+1):
                if epoch % 2 == 0:
                    print('Training epoch: %s' % epoch)
                model.train(self.sentences, start_alpha=0.025, epochs=nrEpochs, total_examples=model.corpus_count)
                model.alpha -= 0.002  # decrease the learning rate
                model.min_alpha = model.alpha  # fix the learning rate, no decay
            model.save(self.model_path)
        self.model = model

    def create_lexicon(self, lexicon_name):
        self.lexicon_name = lexicon_name
        categories = ['macroeconomia', 'sostenibilidad', 'innovacion', 'regulaciones', 'alianzas', 'reputacion']
        if self.stemming:
            ps = PorterStemmer()
            categories_s = [ps.stem(x) for x in categories]
        else:
            categories_s = categories
        json_path = '../data/archivos_auxiliares/jsons/{}_stemming.json'.format(self.lexicon_name) if self.stemming else 'jsons/{}_no_stemming.json'.format(self.lexicon_name)
        if not os.path.exists(json_path):
            print('Lexicon does not exists. Generating lexicon...')
            self.generate_sentences()
            self.train_model()  
            self.lexicon = {}
            for category in categories_s:
                values_lex = [x[0] for x in self.lexicon.values()]
                self.lexicon[category] = [[x[0].lower(), x[1]] for x in self.model.wv.most_similar(category, topn=300) if x[0].lower() not in values_lex][:100]
            with open(json_path, 'w') as fp:
                json.dump(self.lexicon, fp)
        else:
            print('Lexicon exists. Loading lexicon file...')
            with open(json_path, 'r') as fp:
                self.lexicon = json.load(fp)

    def evaluate_news_lexicon(self, eval_data):
        print('Evaluating news...')
        eval_data['news_title_content'] = eval_data['news_title'] + ' ' + eval_data['news_text_content']
        eval_data['news_title_content'] = eval_data['news_title_content'].map(self.clean_news)
        for category in self.lexicon.keys():
            eval_data[category] = eval_data['news_title_content'].apply(lambda x: self.count_words_in_new(x, self.lexicon[category]))
        eval_data['otras'] = 3
        keys = list(self.lexicon.keys()) + ['otras']
        eval_data['lexicon_probability'] = np.amax(softmax(np.array(eval_data[keys]), axis=1), axis=1)
        eval_data['lexicon_category'] = eval_data[keys].idxmax(axis=1)
        return eval_data[['news_id', 'lexicon_category', 'lexicon_probability']]

