# -*- coding: utf-8 -*-
"""
Program to run tf-idf + cosine similarity between constitutions.

@author: marcos
"""
# Import required libraries
import csv
import pandas as pd

from collections import Counter

import nltk
from nltk.corpus import stopwords

import gensim
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess
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
        yield gensim.utils.simple_preprocess(str(sentence), deacc=True)


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

# Get the number of different words in the corpus
n_words = id2word.num_pos
# Get the most common 50 words in the corpus (possibility removing them as
# stopwords in further runs)
words_list = [word for words in data_words for word in words]
most_common_words = Counter(words_list).most_common(50)

# Save the list of most frequent words into a csv file
with open('results/csv/common_words.csv', 'w') as fout:

    # using csv.writer method from CSV package
    write = csv.writer(fout)
    write.writerow(most_common_words)

# Create the TF-IDF model
model = TfidfModel(corpus)
corpus_tfidf = model[corpus]

# Create matrix of similarity -  based on cosine angle (similarity)
index = gensim.similarities.SparseMatrixSimilarity(model[corpus],
                                                   num_features=len(id2word))

# The similarity DF!
sims = index[corpus_tfidf]
sims_df = pd.DataFrame(sims)

# Creating the DF that must be saved
df_final = df[['country', 'year']]
df_final = df_final.join(sims_df)

# Change the columns names
column_names = (df_final['country'] + '_' + df_final['year']).to_list()
df_final.columns = ['country', 'year'] + column_names

df_final.to_excel('results/tf_idf/constitutions.xlsx')

base = df_final.drop('year', axis=1)
# base = base.drop('country', axis=1)
base['country_year'] = column_names
base = base.set_index('country_year')

df_group = base.groupby(['country'])
dfs = dict(tuple(df_group))

# This small loop do the calculation in the right rows (for comparative between
# first and all furthers constitutions).
similarity_prime = []
for country in dfs:
    for i in range(len(dfs[country])):
        name_column = dfs[country].iloc[0, ].name
        series_next = dfs[country].iloc[i, ]
        value = series_next.loc[name_column, ]
        similarity_prime.append(value)

final_prime = df[['country', 'year']]
final_prime.insert(2, 'similarity', similarity_prime, True)


# This small loop do the calculation in the right rows (for comparative in
# sequence).

similarity_sequence = []
for country in dfs:
    similarity_sequence.append(1.0)
    for i in range(len(dfs[country])-1):
        name_column = dfs[country].iloc[i, ].name
        series_next = dfs[country].iloc[i+1, ]
        similarity_sequence.append(series_next.loc[name_column, ])

final_sequence = df[['country', 'year']]
final_sequence.insert(2, 'similarity', similarity_sequence, True)

# Finally, we must save data
final_prime.to_csv('results/csv/prime_tfidf.csv')
final_sequence.to_csv('results/csv/sequence_tfidf.csv')
