# -*- coding: utf-8 -*-
"""
Program to run tf-idf + cosine similarity between constitutions.

@author: marcola
"""
import glob
import os
import pandas as pd
import io
import re
from collections import Counter

import nltk
from nltk.corpus import stopwords

import spacy
import gensim
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess

import logging
import warnings

# Not important to understand, make it easier to see what is happenning in
# the process
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

# =============================================================================
# You can run all this again or use the DataFrame Created on LDA.py
# =============================================================================
# Local of texts
loc_total = r'data\all'

# Getting the list of all texts
all_texts = glob.glob(os.path.join(loc_total, '*.txt'))


def get_df(texts):
    """
    Create a Df from a list of texts - their name in local memory.

    It makes a DF of the name, year (from its names) and content of each text.

    Parameters
    ----------
    texts : TYPE list
        DESCRIPTION. list of all txt files (constitutions) of the same
        language.


    Returns
    -------
    None.

    """
    country = []
    year = []
    constitution = []

    for txt in texts:
        country_year = os.path.splitext((os.path.basename(txt)))[0]
        country.append(' '.join(country_year.split('_')[:-1]))
        year.append(country_year.split('_')[-1])

        with io.open(txt, 'r', encoding='utf-8', errors='ignore') as fout:
            constitution.append(''.join(fout.readlines()))

    corpus = pd.DataFrame({'country': country, 'year': year,
                           'constitution': constitution})

    return corpus


def lemmatization(texts):
    """See https://spacy.io/api/annotation."""
    texts_out = []
    nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])
    for sent in texts:
        doc = nlp(sent)
        texts_out.append([token.lemma_ for token in doc if token.pos_])
    return texts_out


# Creating DF and then applying lemmatization. Important to do before
# preprocessing.
corpus_all = get_df(all_texts)

data_all = corpus_all.constitution.tolist()
all_lemma = lemmatization(data_all)
all_lemma = [' '.join(item) for item in all_lemma]


def clean_text(text):
    """Clean up text data.

    Make text lowercase, remove text in square brackets, remove
    punctuation, remove digits in general, remove urls, remove
    emails and remove "" caracteres.
    Also remove some common and bad words such as "Chapter".
    """
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[!,:\-;\.\?\(\)]', '', text)
    text = ''.join(i for i in text if not i.isdigit())
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[‘’“”…]', '', text)
    text = re.sub('\n', ' ', text)
    text = re.sub(r'<list>', ' ', text)
    text = re.sub(r'<title>', ' ', text)
    text = re.sub(r'<title', ' ', text)
    text = re.sub(r'<preamble>', ' ', text)
    text = re.sub(r'</list>', '', text)
    text = re.sub(r'/', '', text)
    text = re.sub(r'Chapter', '', text)
    text = re.sub(r'chapter', '', text)
    text = re.sub('(\\b[A-Za-z] \\b|\\b [A-Za-z]\\b)', '', text)
    text = re.sub(r'section', '', text)
    text = re.sub(r'&', '', text)
    text = re.sub(r'-(?!\w)|(?<!\w)-', '', text)
    text = re.sub(r'_', '', text)
    text = re.sub(r'copyright', '', text)
    text = re.sub(r'@', '', text)
    text = re.sub(r'iii', '', text)
    text = re.sub(r'ii', '', text)
    text = re.sub(r'vi', '', text)
    text = re.sub(r'vii', '', text)
    text = re.sub(r'vii', '', text)
    text = re.sub(r'viii', '', text)
    text = re.sub(r'ix', '', text)
    text = re.sub(r'x', '', text)
    text = re.sub(r'xi', '', text)
    text = re.sub(r'xii', '', text)
    text = re.sub(r'<list>', '', text)

    return text


# Adding the new lemmatized data to the DF
corpus_all['constitution_lemma'] = all_lemma
# Applying some cleaninh
corpus_all.constitution_lemma = corpus_all['constitution_lemma'].apply(
    lambda x: clean_text(x))

df = corpus_all
# =============================================================================
# Using the Dataframe we already created:
# =============================================================================
df = pd.read_pickle('./results/df_total_all_set.pkl')
df = df.sort_values(by=['country', 'year'])
df = df.reset_index(drop=True)

# Tokening the data set


def sent_to_words(sentences):
    """Do a simple tokenization."""
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))


# Time to tokening words and create necessary objects
data = df.constitution_lemma.to_list()
data_words = list(sent_to_words(data))


# We create the list of stopwords
gensim_stop = STOPWORDS
nltk.download('stopwords')
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'i', 'ii', 'iii',
                   'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x', 'xi', 'xii',
                   'xiii', 'xiv', 'xv', 'xvi', 'xvii', 'xviii', 'xix', 'xx',
                   'xxi', 'xxii', 'xxiii', 'xxiv', 'xxv', 'xxvi', 'xxvii',
                   'xxviii', 'xxix', 'xxx', 'xxxi', 'xxxii', 'xxxiii',
                   'xxxiv', 'xxxv', 'xxxvi', 'xxxvii', 'xxxviii', 'xxxix',
                   'xl', 'xli', 'xlii', 'xlii', 'xliii',  'xliv', 'xlv',
                   'xlvi', 'xlvii', 'xlviii', 'xlix', 'l', 'li', 'lii', 'liii',
                   'liv', 'lv', 'lvi', 'lvii', 'lviii', 'lix', 'lx', 'lxi',
                   'lxii', 'lxiii', 'lxiv', 'lxv', 'lxvi', 'lxvii', 'lxviii',
                   'lxix', 'lxx', 'lxxi', 'lxxii', 'lxxiii', 'lxxiv', 'lxxv',
                   'lxxvi', 'lxxvii', 'lxxviii', 'lxxix', 'lxxx', 'lxxxi',
                   'lxxxii', 'lxxxiii', 'lxxxiv', 'lxxxv',  'lxxxvi',
                   'lxxxvii', 'lxxxviii', 'lxxxix', 'xc', 'xci', 'xcii',
                   'xciii', 'xciv', 'xcv', 'xcvi', 'xcvii', 'xcviii', 'xcix',
                   'c', 'january', 'february', 'march', 'april', 'may',
                   'june', 'july', 'august', 'september', 'october',
                   'november', 'december', '+', 'shall', '*', 'copyright',
                   'project', 'constitutions', 'projects', 'stanford',
                   'havard', 'united nations', 'volume', 'ohio', 'preamble'
                   '©', 'bibliography', '<', '>', '/', 'may',
                   'constitution', 'country', 'shall', 'hoover', 'project',
                   'institution', 'new york', 'washington', 'law'])
stop_words.extend(gensim_stop)
stop_words.extend(['sub', 'clause', 'article'])


# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    """Remove stopwords."""
    return [[word for word in simple_preprocess(str(doc)) if word not in
             stop_words] for doc in texts]


data_words = remove_stopwords(data_words)

# Create Dictionary
id2word = Dictionary(data_words)
corpus = [id2word.doc2bow(text) for text in data_words]

# Get the number of differents words in the corpus
n_words = id2word.num_pos
# Get the most common 50 words in the corpus (possiblity removing them as
# stopwords in further runs)
words_list = [word for words in data_words for word in words]
most_common_words = Counter(words_list).most_common(50)

# Create the TF-IDF model
model = TfidfModel(corpus)
corpus_tfidf = model[corpus]

# Create matrix of similarity -  based on cosine angle (similarity)
index = gensim.similarities.SparseMatrixSimilarity(model[corpus],
                                                   num_features=len(id2word))

# The similarity DF!
sims = index[corpus_tfidf]
sims_df = pd.DataFrame(sims)

# The DF that must be saved
df_final = df[['country', 'year']]
df_final = df_final.join(sims_df)

column_names = (df_final['country'] + '_' + df_final['year']).to_list()

df_final.columns = ['country', 'year'] + column_names

df_final.to_excel('results/tf_idf/constitutions.xlsx')

base = df_final.drop('year', axis=1)
# base = base.drop('country', axis=1)
base['country_year'] = column_names
base = base.set_index('country_year')

df_group = base.groupby(['country'])
dfs = dict(tuple(df_group))

# This small loop do the calculation in the right rows (for comparation between
# first and all furthers constitutions).
similarity_prime = []
for country in dfs:
    for i in range(len(dfs[country])):
        name_column = dfs[country].iloc[0, ].name
        series_next = dfs[country].iloc[i, ]
        value = series_next.loc[name_column, ]
        similarity_prime.append(value)

final_prime = df[['country', 'year']]
final_prime['similarity'] = similarity_prime


# This small loop do the calculation in the right rows (for comparation in
# sequence).

similarity_sequence = []
for country in dfs:
    similarity_sequence.append(1.0)
    for i in range(len(dfs[country])-1):
        name_column = dfs[country].iloc[i, ].name
        series_next = dfs[country].iloc[i+1, ]
        similarity_sequence.append(series_next.loc[name_column, ])

final_sequence = df[['country', 'year']]
final_sequence['similarity'] = similarity_sequence

# Finally, we must save data
final_prime.to_csv('results/csv/prime_tfidf.csv')
# final_prime.to_excel('results/csv/sequence_tfidf.xlsx')
final_sequence.to_csv('results/csv/sequence_tfidf.csv')
# final_sequence.to_excel('results/tf_idf/sequence_tfidf.xlsx')
final_sequence.to_csv('results/csv/sequence_tfidf.csv')
# final_sequence.to_excel('results/tf_idf/sequence_tfidf.xlsx')
