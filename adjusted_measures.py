# -*- coding: utf-8 -*-
"""
In this program, we compute some important controls variables.

Also, we compute adjusted measures of our project:

    Change between first and second constitutions divided by number of years
    of first regime


    Change between first and current constitution, divided by number of years
    of constitutional government in total

    Change between first and current constitution, with no adjustment for
    years of regime in total


"""

import pandas as pd
import numpy as np

folder = r'results/csv'

# Reading all csv files (comparing first const with all others)
lda_prime = pd.read_csv(f'{folder}/comp_prime_50topics(fullset_08_05).csv',
                        index_col=0)
tfidf_prime = pd.read_csv(f'{folder}/prime_tfidf.csv',
                          index_col=0)
use_prime = pd.read_csv(f'{folder}/prime_use.csv',
                        index_col=0)
stm_prime = pd.read_csv(f'{folder}/stm_prime_50topics(fullset_03_06).csv',
                        index_col=0)

# Reading all csv files (comparing each const with their sucessor)
lda_seq = pd.read_csv(f'{folder}/comp_seq_50topics(fullset_08_05).csv',
                      index_col=0)
tfidf_seq = pd.read_csv(f'{folder}/sequence_tfidf.csv',
                        index_col=0)
use_seq = pd.read_csv(f'{folder}/sequence_use.csv',
                      index_col=0)
stm_seq = pd.read_csv(f'{folder}/stm_seq_50topics(fullset_03_06).csv',
                      index_col=0)

# =============================================================================
# =============================================================================
# FOR NOW ON, LET'S FIRST ANAYLSE THE PRIME DATAFRAMES
# AFTER WE ADJUST FOR SEQUENCE DAtAFRAMES
# =============================================================================
# =============================================================================

# =============================================================================
# Important remark: tfidf and USE semantic similarity were measured based on
# cosine similarity. In order to define the distance between constitutions,
# we use cosine distance which is: distance =  1 - cosine similarity.
# For more information, see:
# https://en.wikipedia.org/wiki/Cosine_similarity
# docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
# =============================================================================

# Changing columns names and creating cosine distance
tfidf_prime['similarity'] = 1 - tfidf_prime['similarity']

tfidf_prime['similarity'] = np.where(tfidf_prime['similarity'] <= 8e-06,
                                     0,
                                     tfidf_prime['similarity'])

use_prime['similarity'] = 1 - use_prime['similarity']

use_prime['similarity'] = np.where(use_prime['similarity'] <= 4e-06,
                                   0,
                                   use_prime['similarity'])

tfidf_prime.columns = ['country', 'year', 'distance_tfidf']
lda_prime.columns = ['country', 'year', 'distance_lda50']
use_prime.columns = ['country', 'year', 'distance_use']
stm_prime.columns = ['country', 'year', 'distance_stm50']

# Just to make sure that all dataframes are sorted
tfidf_prime = tfidf_prime.sort_values(by=['country', 'year'])
tfidf_prime = tfidf_prime.reset_index(drop=True)

lda_prime = lda_prime.sort_values(by=['country', 'year'])
lda_prime = lda_prime.reset_index(drop=True)

use_prime = use_prime.sort_values(by=['country', 'year'])
use_prime = use_prime.reset_index(drop=True)

stm_prime = stm_prime.sort_values(by=['country', 'year'])
stm_prime = stm_prime.reset_index(drop=True)

# Creating unique dataframe for all measures and the necessary data structure

# IMPORTANT: DON'T CHANGE THIS ORDER UNLESS Y CHANGE THE ORDER ON THE FUNCTION
df_all = tfidf_prime.copy()
df_all = df_all.merge(lda_prime, how='outer')
df_all = df_all.merge(use_prime, how='left')
df_all = df_all.merge(stm_prime, how='outer')

# Save correlations as df
corr = df_all.corr()
corr.to_csv(f'{folder}/correlations(fullset_03_06).csv')

# Data structure we need
base = df_all.set_index('country')
df_group = base.groupby(['country'])
dfs = dict(tuple(df_group))


def get_adjusted(dfs):
    """
    Create adjusted measures and new control variables.

    First, we collect the year of the first constitution in each country.
    Second, we collect the distance between first and second constitutions.
    Third, we collect the distance between first and current constitutions.
    Fourth, we calculate the distance between first and second constitutions
    divided by number of years of first regime.
    Fifth, we calculate the distance between first and current constitution,
    divided by number of years of constitutional government in total.
    Sixth, we calculate distance between first and current constitution,
    with no adjustment for years of regime in total.
    Finally, we construct a Dataframe with all this variables in 204 countries.

    Parameters
    ----------
    dfs : dict of dataframes
        we groupby the dataframe of all constitutions distances based on
        countries.

    Returns
    -------
    Dataframe containing all 192 countries and all necessary variables.

    """
    # Defining necessary variables
    countries = []
    first_constitution_year = []
    second_constitution_year = []
    change_first_second_tfidf = []
    change_first_current_tfidf = []
    change_first_second_lda = []
    change_first_current_lda = []
    change_first_second_use = []
    change_first_current_use = []
    change_first_second_stm = []
    change_first_current_stm = []

    # loop to get all necessary data. Note that many countries have only
    # one constitution so we made a simple if condition on this.

    for country in dfs:
        countries.append(country)
        if dfs[country].shape[0] > 1:
            year = dfs[country].iloc[0, 0]
            year_second = dfs[country].iloc[1, 0]
            change_small_tfidf = dfs[country].iloc[1, 1]
            change_big_tfidf = dfs[country].iloc[-1, 1]
            change_small_lda = dfs[country].iloc[1, 2]
            change_big_lda = dfs[country].iloc[-1, 2]
            change_small_use = dfs[country].iloc[1, 3]
            change_big_use = dfs[country].iloc[-1, 3]
            change_small_stm = dfs[country].iloc[1, 4]
            change_big_stm = dfs[country].iloc[-1, 4]

            first_constitution_year.append(year)
            second_constitution_year.append(year_second)

            change_first_second_tfidf.append(change_small_tfidf)
            change_first_second_lda.append(change_small_lda)
            change_first_second_use.append(change_small_use)
            change_first_second_stm.append(change_small_stm)

            change_first_current_tfidf.append(change_big_tfidf)
            change_first_current_lda.append(change_big_lda)
            change_first_current_use.append(change_big_use)
            change_first_current_stm.append(change_big_stm)

        else:
            year = dfs[country].iloc[0, 0]
            year_second = np.nan
            change_small_tfidf = np.nan
            change_big_tfidf = np.nan
            change_small_lda = np.nan
            change_big_lda = np.nan
            change_small_use = np.nan
            change_big_use = np.nan
            change_small_stm = np.nan
            change_big_stm = np.nan

            first_constitution_year.append(year)
            second_constitution_year.append(year_second)

            change_first_second_tfidf.append(change_small_tfidf)
            change_first_second_lda.append(change_small_lda)
            change_first_second_use.append(change_small_use)
            change_first_second_stm.append(change_small_stm)

            change_first_current_tfidf.append(change_big_tfidf)
            change_first_current_lda.append(change_big_lda)
            change_first_current_use.append(change_big_use)
            change_first_current_stm.append(change_big_stm)

    df = pd.DataFrame({'country': countries,
                       '1st_const_year': first_constitution_year,
                       '2nd_const_year': second_constitution_year,
                       '1st_2nd_tfidf': change_first_second_tfidf,
                       '1st_current_tfidf': change_first_current_tfidf,
                       '1st_2nd_lda': change_first_second_lda,
                       '1st_current_lda': change_first_current_lda,
                       '1st_2nd_use': change_first_second_use,
                       '1st_current_use': change_first_current_use,
                       '1st_2nd_stm': change_first_second_stm,
                       '1st_current_stm': change_first_current_stm})

    return df


data = get_adjusted(dfs)
data['constitutional_time'] = 2022 - data['1st_const_year']
data['first_regime_time'] = data['2nd_const_year'] - data['1st_const_year']
# Many countries didn't change their constitutions. We add the first regime
# time as the constitutional time for this situation
data.first_regime_time.fillna(data.constitutional_time, inplace=True)

# Last definintion of variables. There is what we were thinking of as possible
# adjusted measures
data['1st_2nd_tfidf_adj'] = data['1st_2nd_tfidf']/data['first_regime_time']
data['1st_2nd_lda_adj'] = data['1st_2nd_lda']/data['first_regime_time']
data['1st_2nd_use_adj'] = data['1st_2nd_use']/data['first_regime_time']
data['1st_2nd_stm_adj'] = data['1st_2nd_stm']/data['first_regime_time']

data['1st_curr_tfidf_adj'] = data['1st_current_tfidf'] / \
    data['constitutional_time']
data['1st_curr_lda_adj'] = data['1st_current_lda'] / \
    data['constitutional_time']
data['1st_curr_use_adj'] = data['1st_current_use'] / \
    data['constitutional_time']
data['1st_curr_stm_adj'] = data['1st_current_stm'] / \
    data['constitutional_time']

# Open csv with countries codes and merge to data
codes = pd.read_csv(r'Data/country_codes/codes.csv', sep=';')
data['codes'] = codes['country_code']
cols = ['codes', 'country', '1st_const_year', '2nd_const_year',
        '1st_2nd_tfidf', '1st_current_tfidf',
        '1st_2nd_lda', '1st_current_lda',
        '1st_2nd_use', '1st_current_use',
        '1st_2nd_stm', '1st_current_stm',
        'constitutional_time', 'first_regime_time',
        '1st_2nd_tfidf_adj',
        '1st_2nd_lda_adj',
        '1st_2nd_use_adj',
        '1st_2nd_stm_adj',
        '1st_curr_tfidf_adj',
        '1st_curr_lda_adj',
        '1st_curr_use_adj',
        '1st_curr_stm_adj']
data = data[cols]

# Save data
data.to_csv(f'{folder}/all_textual_variables(03_06).csv')

# Save dataframe with all correlations
matrix = data.corr()
matrix.to_csv(f'{folder}/all_correlations(03_06).csv')

# =============================================================================
# =============================================================================
# SEQUENCE DATAFRAME TIME
# =============================================================================
# =============================================================================

# Changing columns names and creating cosine distance
tfidf_seq['similarity'] = 1 - tfidf_seq['similarity']

tfidf_seq['similarity'] = np.where(tfidf_seq['similarity'] <= 8e-06,
                                   0,
                                   tfidf_seq['similarity'])

use_seq['similarity'] = 1 - use_seq['similarity']

use_seq['similarity'] = np.where(use_seq['similarity'] <= 4e-06,
                                 0,
                                 use_seq['similarity'])

tfidf_seq.columns = ['country', 'year', 'distance_tfidf']
lda_seq.columns = ['country', 'year', 'distance_lda50']
use_seq.columns = ['country', 'year', 'distance_use']
stm_seq.columns = ['country', 'year', 'distance_stm50']

# Just to make sure that all dataframes are sorted
tfidf_prime = tfidf_seq.sort_values(by=['country', 'year'])
tfidf_seq = tfidf_seq.reset_index(drop=True)

lda_seq = lda_seq.sort_values(by=['country', 'year'])
lda_seq = lda_seq.reset_index(drop=True)

use_seq = use_seq.sort_values(by=['country', 'year'])
use_seq = use_seq.reset_index(drop=True)

stm_seq = stm_seq.sort_values(by=['country', 'year'])
stm_seq = stm_seq.reset_index(drop=True)

# Creating unique dataframe for all measures and the necessary data structure

# IMPORTANT: DON'T CHANGE THIS ORDER UNLESS Y CHANGE THE ORDER ON THE FUNCTION
df_all_seq = tfidf_seq.copy()
df_all_seq = df_all_seq.merge(lda_seq, how='outer')
df_all_seq = df_all_seq.merge(use_seq, how='left')
df_all_seq = df_all_seq.merge(stm_seq, how='outer')

corr = df_all_seq.corr()

base_seq = df_all_seq.set_index('country')
df_group_seq = base_seq.groupby(['country'])
dfs_seq = dict(tuple(df_group_seq))


def get_adjusted_seq(dfs_seq):
    """
    Create adjusted measures and new control variables.

    First, we collect the year of the first constitution in each country.
    Second, we collect the distance between first and second constitutions.

    Finally, we construct a Dataframe with all this variables in 204 countries.

    Parameters
    ----------
    dfs : dict of dataframes
        we groupby the dataframe of all constitutions distances based on
        countries.

    Returns
    -------
    Dataframe containing all 192 countries and all necessary variables.

    """
    # Defining necessary variables

    countries = []
    first_constitution_year = []
    second_constitution_year = []
    tfidf_change = []
    lda_change = []
    use_change = []
    stm_change = []

    for country in dfs_seq:
        countries.append(country)
        year = dfs_seq[country].iloc[0, 0]
        all_change_tfidf = dfs_seq[country].iloc[:, 1].sum()
        all_change_lda = dfs_seq[country].iloc[:, 2].sum()
        all_change_use = dfs_seq[country].iloc[:, 3].sum()
        all_change_stm = dfs_seq[country].iloc[:, 4].sum()

        try:
            year_second = dfs_seq[country].iloc[1, 0]
        except IndexError:
            year_second = np.nan

        first_constitution_year.append(year)
        second_constitution_year.append(year_second)
        tfidf_change.append(all_change_tfidf)
        lda_change.append(all_change_lda)
        use_change.append(all_change_use)
        stm_change.append(all_change_stm)

    df = pd.DataFrame({'country': countries,
                       '1st_const_year': first_constitution_year,
                       '2nd_const_year': second_constitution_year,
                       'tfidf_distance': tfidf_change,
                       'lda_distance': lda_change,
                       'use_distance': use_change,
                       'stm_distance': stm_change})

    return df


# Get and save dataset
data = get_adjusted_seq(dfs_seq)

# Creating last columns
data['constitutional_time'] = 2022 - data['1st_const_year']
data['first_regime_time'] = data['2nd_const_year'] - data['1st_const_year']

# Open csv with countries codes and merge to data
codes = pd.read_csv(r'Data/country_codes/codes.csv', sep=';')
data['codes'] = codes['country_code']

cols = ['codes', 'country', '1st_const_year', '2nd_const_year',
        'tfidf_distance', 'lda_distance', 'use_distance', 'stm_distance']
data.to_csv(f'{folder}/all_textual_variables_seq(03_06).csv')
