# -*- coding: utf-8 -*-
"""
Created on Fri May 20 10:06:13 2022

Trying to see the relantionship between usa and other constitutions.
@author: marcola
"""
import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt

folder = r'results/Topics_models'
df_final = pd.read_pickle(f'{folder}/df_topics(50topics_08_05(full_set)).pkl')
df_final = df_final.sort_values(by=['country', 'year'])
df_final = df_final.reset_index(drop=True)


usa_const = df_final.loc[df_final.country == 'United states']
usa_values = usa_const.iloc[0, 2:].values.tolist()

canada_const = df_final.loc[df_final.country == 'Canada']
canada_values = canada_const.iloc[0, 2:].values.tolist()


def similarity(df_final, comparison):
    """Trivial."""
    sim = []
    for row in range(df_final.shape[0]):
        similarity = jensenshannon(comparison,
                                   df_final.iloc[row, 2:].values.tolist())
        sim.append(similarity)

    final_sequence = df_final[['country', 'year']]
    final_sequence['distance'] = sim

    return final_sequence


lda_usa = similarity(df_final, usa_values)
lda_canada = similarity(df_final, canada_values)

ax1 = lda_usa.plot.scatter(x='year',
                           y='distance',
                           c='DarkBlue')

ax2 = lda_canada.plot.scatter(x='year',
                              y='distance',
                              c='Red')

lda_usa.year = pd.to_numeric(lda_usa.year)
lda_canada.year = pd.to_numeric(lda_canada.year)
corr_usa = lda_usa.corr()
corr_canada = lda_canada.corr()


# =============================================================================
# Old LDA version
# =============================================================================
df_final2 = pd.read_pickle(f'{folder}/df_topics(40topics_02_02(full_set)).pkl')
df_final2 = df_final2.sort_values(by=['country', 'year'])
df_final2 = df_final2.reset_index(drop=True)

usa_const2 = df_final2.loc[df_final2.country == 'United states']
usa_values2 = usa_const2.iloc[0, 2:].values.tolist()

canada_const2 = df_final2.loc[df_final2.country == 'Canada']
canada_values2 = canada_const2.iloc[0, 2:].values.tolist()

lda2_usa = similarity(df_final2, usa_values2)
lda2_canada = similarity(df_final2, canada_values2)

lda2_usa.year = pd.to_numeric(lda2_usa.year)
corr2_usa = lda2_usa.corr()

# =============================================================================
# TFIDF
# =============================================================================

folder = r'results/tf_idf/'

tfidf = pd.read_excel(f'{folder}constitutions.xlsx')
tfidf = tfidf.drop(['Unnamed: 0'], axis=1)

df = tfidf[['country', 'year', 'United states_1789']]
df.columns = ['country', 'year', 'tfidf_usa']

df.loc[:, 'distance_usa'] = 1 - df.loc[:, 'tfidf_usa']
df['distance_usa'] = np.where(df['distance_usa'] <= 8e-06,
                              0,
                              df['distance_usa'])
