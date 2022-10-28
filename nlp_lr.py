# Imports
import pandas as pd
# import numpy as np

import pickle

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

import nltk
nltk.download('all')

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# regular expression
import re
import string

def clean(text):
    text = text.lower() # нижний регистр
    text = re.sub(r'http\S+', " ", text) # удаляем ссылки
    text = re.sub(r'@\w+',' ',text) # удаляем упоминания пользователей
    text = re.sub(r'#\w+', ' ', text) # удаляем хэштеги    
    text = re.sub("<.*?>", " ", text) # удаляем все теги <div /><div /> 
    text = re.sub("himselfâ\w+", " ", text) # удаляем теги слова начинающиеся с himselfâ\
    text = re.sub(r'\d+', ' ', text) # удаляем числа
    text = text.translate(str.maketrans('', '', string.punctuation))
    # text = re.sub(r'<.*?>',' ', text) # 
    return text

def remove_emojis(text):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', text)    

def lemma(text):
    # create lemmatizer object
    wn_lemmatizer = WordNetLemmatizer()
    # Применяем лемматизацию к тексту    
    lemmatized_text = []
    lemmatized_text.append(' '.join([wn_lemmatizer.lemmatize(word) for word in text.split()]))    
    return lemmatized_text


def tokenize_and_clean(lemmatized_text):
    # Stopwords: неинформативные слова
    # Lemmatizer: каноническая форма слова
    # Tokenizer: разбивает текст на токены - слова

    # Create and apply tokenizer
    reg_tokenizer = RegexpTokenizer('\w+')
    tokenized_text = reg_tokenizer.tokenize_sents(lemmatized_text)

    # стоп-слова надо кэшировать
    sw = stopwords.words('english')

    # clean tokenized comments
    clean_tokenized_comments = [] 
    for i, element in enumerate(tokenized_text):
        clean_tokenized_comments.append(' '.join([word for word in element if word not in sw]))

    return clean_tokenized_comments

def vectorize(clean_tokenized_comments):
    # Create objects
    # cvec = CountVectorizer(ngram_range=(1, 1))
    # tfid = TfidfVectorizer(ngram_range=(1, 1))

    # load the model from disk
    filename = './tfid_model.pkl'
    loaded_tfid_model = pickle.load(open(filename, 'rb'))
    answer = loaded_tfid_model.transform(clean_tokenized_comments)
    return answer

def negative_or_positive(tfid_representation):
    # load the model from disk
    filename = './LogReg_model.pkl'
    loaded_Log_Reg_model = pickle.load(open(filename, 'rb'))
    answer = loaded_Log_Reg_model.predict(tfid_representation)
    # print(answer)
    return answer

def common_func(text):
    text = clean(text)
    # print(f'clean(text): {text}')
    
    text = remove_emojis(text)
    # print(f'remove_emojis(text): {text}')

    text = lemma(text)
    # print(f'lemma(text): {text}')

    text = tokenize_and_clean(text)
    # print(f'tokenize_and_clean(text): {text}')

    text = vectorize(text)
    # print(f'vectorize(text): {text}')

    text = negative_or_positive(text)
    # print(f'negative_or_positive(text): {text}')

    if text[0] == 1:
        answer = 'positive'
    else:
        answer = 'negative'

    return answer