# -*- coding: utf-8 -*-
"""
Created to Do similarity with spacy transformer measure.

We saw that having the fullset of words leads to problems of curse of
dimensionality. We atempt to use only specific set of "meaningful" words.
This "meaningfulness", although arbitratry, is a common way to preprocess
text-in-data projects and it is used in other programs we created. We
basically did all the preprocessing in other file and open the dataframe
cointaing this preprocessed data in this one.
@author: correia-marcos
"""

import os

import pandas as pd
# import spacy
import spacy_universal_sentence_encoder as spacy_USE

import logging
import warnings

# Not important to understand, make it easier to see running code
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# From LDA.py we take open the saved DataFrame. Remember that Text data
# have already been processed in this DataFrame (applied Lemmatization,
# removing stopwords...).
df = pd.read_pickle('./results/df_total_all_set.pkl')
df = df.sort_values(by=['country', 'year'])

# Using lists will help in time running
text = df.constitution_lemma.to_list()


def get_spacy_list(texts):
    """
    Take the list of texts and turn into nlp spacy list.

    Also, it removes stopwords words from the data, where spacy transformer
    pipeline have the definition of stopwords.
    (See: https://spacy.io/usage/spacy-101).

    Parameters
    ----------
    texts : list
        DESCRIPTION. List of texts

    Returns
    -------
    list of nlp pipeline for each text.
    """
    texts_out = []
    nlp = spacy_USE.load_model('en_use_lg')
    for text in texts:
        doc = nlp(text)
        texts_out.append([token.text for token in doc if not token.is_stop])

    text_nlp = [' '.join(item) for item in texts_out]
    text_pipe = [nlp(tex) for tex in text_nlp]

    return text_pipe


# Apply function
text_pipe = get_spacy_list(text)

df['nlp_doc'] = text_pipe
df = df[['country', 'year', 'nlp_doc']]

# Main data scturure to be processed
df_group = df.groupby(['country'])
dfs = dict(tuple(df_group))

# This small loop do the calculation in the right rows (for comparation between
# first and all furthers constitutions).
similarity_first = []
for country in dfs:
    for i in range(len(dfs[country])):
        sim = dfs[country].iloc[0, 2].similarity(dfs[country].iloc[i, 2])
        similarity_first.append(sim)

df_first = df.drop('nlp_doc', axis=1)
df_first['similarity'] = similarity_first


# This small loop do the calculation in the right rows (for comparation in
# sequence).
similarity_sequence = []
for country in dfs:
    similarity_sequence.append(1.0)
    for i in range(len(dfs[country])-1):
        sim = dfs[country].iloc[i, 2].similarity(dfs[country].iloc[i+1, 2])
        similarity_sequence.append(sim)


df_sequence = df.drop('nlp_doc', axis=1)
df_sequence['similarity'] = similarity_sequence

# Saving data

df_first.to_csv('results/csv/comp_prime_USE.csv')
df_first.to_excel('results/excel/comparações_prime_USE.xlsx')
df_sequence.to_csv('results/csv/com_sequence_USE.csv')
df_sequence.to_excel('results/excel/comparações_seq_USE.xlsx')
