#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
_author_       = [" David Lizarazo" , " Ricardo Heredia" , "Daniel Baron"]
__copyright__  = "Copyright 2022, DK&R"
__credits__    = [" David Lizarazo" , " Ricardo Heredia" , "Daniel Baron"]
__license__    = "GPL"
__version__    = "0.0.1"
__email__      = "davidlizarazovesga@hotmail.com"
__status__     = "Development"


import os 

def modulesInstallation ():
    """Esta funcion realiza el proceso de instalacion de las dependencias para el presente proyecto """
    try:
        os.system('pip install sklearn --no-input')
        os.system('pip install Unidecode --no-input')
        os.system('pip install gensim --no-input')
        os.system('pip install parsel --no-input')
        os.system('pip install pandas --no-input')
        os.system('pip install google-search-results --no-input')
        os.system('pip install newspaper3k --no-input')
        os.system('pip install parsel --no-input')
        os.system('pip install tqdm --no-input')
        os.system('pip install GoogleNews --user')
        return True
    except Exception as e : 
        print ("ERROR: No fue posible realizar la instalacion de los paquetes requeridos")
        print ("------------------------------------------------------------------------")
        print (e)
        return False

def testDependencies ():
    # Importing Libraries 
    try:
        import requests, json, re
        from parsel     import Selector
        from newspaper  import Article
        from serpapi    import GoogleSearch
        from tqdm       import tqdm

        import json
        import unidecode
        import csv
        import gensim
        import pandas as pd
        import nltk
        nltk.download('stopwords')
        from nltk.corpus import stopwords
        from nltk.stem.porter import PorterStemmer
        import time
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.feature_extraction.text import CountVectorizer



        
        return True

    except Exception as e:
        print ("ERROR: No fue posible importar todas las dependencias")
        print ("------------------------------------------------------------------------")
        print (e)
        return False


if __name__ == "__main__":
    
    print("#### Modulo de instalacion ####")
    print("1. Instalacion de dependencias")
    print("2. Probar dependencias")
    user_in = input("Ingrese su opcion : ")
    print ("------------------")

    if user_in == "1":
        modulesInstallation()
    elif user_in == "2":
        testDependencies()

