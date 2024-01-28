# -*- coding: utf-8 -*-
"""
Program to measure the Distance between probability distributions.

Those probabilities distributions were previous developed in LDA.py by the
LDA method.

@author: marcola
"""
# Import the required libraries
import pandas as pd
import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon

# Reading the DF previous created:
folder = r'results/Topics_models'
df_final = pd.read_pickle(f'{folder}/df_topics(50topics_08_05(full_set)).pkl')
df_final = df_final.sort_values(by=['country', 'year'])
df_final = df_final.reset_index(drop=True)

# The following function were created but not used! We were only making
# sure we understand the variable definition


def jensen_shannon(query, matrix):
    """
    Create a Jensen-Shannon similarity.

    it is between the input query (an LDA topic distribution for a document)
    and the entire corpus of topic distributions.
    It returns an array of length M where M is the number of documents in
    the corpus.
    """
    # lets keep with the p,q notation above
    p = query[None, :].T  # take transpose
    q = matrix.T  # transpose matrix
    m = 0.5*(p + q)

    return np.sqrt(0.5*(entropy(p, m) + entropy(q, m)))


# Reducing the columns of the Dataframe in order to measure the Jensen Shanon
# Distance.
names = [f'Topic {i+1}' for i in range(df_final.shape[1]-2)]
names = ['country'] + names


# Here we create our dataset in the according scruture.
base = df_final[names]
base = base.set_index('country')
df_group = base.groupby(['country'])
dfs = dict(tuple(df_group))


# This small loop do the calculation in the right rows (for comparasion between
# first and all furthers constitutions).
similarity_first = []
for country in dfs:
    for i in range(len(dfs[country])):
        sim = jensenshannon(dfs[country].iloc[0, ].values,
                            dfs[country].iloc[i, ].values)
        similarity_first.append(sim)

final_first = df_final[['country', 'year']]
final_first.insert(2, "similarity", similarity_first, True)

# This small loop do the calculation in the right rows (for comparasion in
# sequence).
similarity_sequence = []
for country in dfs:
    similarity_sequence.append(0.0)
    for i in range(len(dfs[country])-1):
        sim = jensenshannon(dfs[country].iloc[i, ].values,
                            dfs[country].iloc[i+1, ].values)
        similarity_sequence.append(sim)


final_sequence = df_final[['country', 'year']]
final_sequence.insert(2, "similarity", similarity_sequence, True)

# Saving data
final_first.to_excel(
    'results/excel/comparações_prime_45topics(fullset_06_02).xlsx')
final_sequence.to_excel(
    'results/excel/comparações_seq_45topics(fullset_06_02final_sequence).xlsx')

final_first.to_csv('results/csv/comp_prime_50topics(fullset_08_05).csv')
final_sequence.to_csv('results/csv/comp_seq_50topics(fullset_08_05).csv')

# =============================================================================
# =============================================================================
# Taking Results from STM model (R) as well
# =============================================================================
# =============================================================================

folder_stm = r'results/csv/'

# Open file
df_stm = pd.read_csv(rf'{folder_stm}/stm(50topics).csv', header=None,
                     sep=' ')

# Adding country and year of constitution (dataframe is in the same order)
df_stm[['country', 'year']] = df_final[['country', 'year']]

# Changing labels
names = [f'Topic {i+1}' for i in range(df_final.shape[1]-2)]
cols = names + ['country', 'year']
df_stm.columns = cols

# Changing columns posititions
new_cols = cols[-2:] + cols[:-2]
df_stm = df_stm[new_cols]

# Here we create our dataset in the according scruture.
names = ['country'] + names
base_stm = df_stm[names]
base_stm = base_stm.set_index('country')
df_group = base_stm.groupby(['country'])
dfs_stm = dict(tuple(df_group))


# This small loop do the calculation in the right rows (for comparasion between
# first and all furthers constitutions).
similarity_first_stm = []
for country in dfs_stm:
    for i in range(len(dfs_stm[country])):
        sim = jensenshannon(dfs_stm[country].iloc[0, ].values,
                            dfs_stm[country].iloc[i, ].values)
        similarity_first_stm.append(sim)

final_first_stm = df_stm[['country', 'year']]
final_first_stm.insert(2, "similarity", similarity_first_stm, True)


# This small loop do the calculation in the right rows (for comparasion in
# sequence).
similarity_sequence_stm = []
for country in dfs_stm:
    similarity_sequence_stm.append(0.0)
    for i in range(len(dfs_stm[country])-1):
        sim = jensenshannon(dfs_stm[country].iloc[i, ].values,
                            dfs_stm[country].iloc[i+1, ].values)
        similarity_sequence_stm.append(sim)


final_sequence_stm = df_final[['country', 'year']]
final_sequence_stm.insert(2, 'similarity', similarity_sequence_stm, True)


# Saving dataset
final_first_stm.to_csv('results/csv/stm_prime_50topics(fullset_03_06).csv')
final_first_stm.to_csv('results/csv/stm_seq_50topics(fullset_03_06).csv')
