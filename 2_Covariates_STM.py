"""
Get the colonies of each country among other variables for STM model.

Variables:
    Dummy for country colonization
    Dummy for colonial origin
    Number of words in each constitution
    Number of amendments in total

VERY IMPORTANT: we have constitutions from CPP project and we are using
one CPP dataset. However, there seems to be some strange in the dataset
cause, for example, Afghanistan 1987 were create a new constitution
but the new length only happens 3 year later. This happened 145 times in
our dataset.
We tried to resolve this problem by filling "nan" values with the
subsequent value (in periods of constitutional change), but it wasn't much
precise either.
We also compared the difference between amount of post pre-processed words
and
the one in the dataset. The amount of pre-processment reduced greatly the
variance of the text so this isn't a good proxy also.
We them taught about taking the amount of words in txt text we had and
hand coded this which might take a long time to achieve.

@author: marcos

"""
# Importing files

import pandas as pd
import pycountry
from pandas_profiling import ProfileReport
# import ast


# place of files/folders
FILE_CONSTITUTIONS = r'results/csv/constitutions.csv'
COLONIES_FILES = r'Data/colonies/'
CCP_FILE = r'Data/comparative_project/'

# Read file
data_constitutions = pd.read_csv(FILE_CONSTITUTIONS, index_col=0)
# profile = ProfileReport(data_constitutions, title="Pandas Profiling Report")

# Add column for number of words in lemma documents
data_constitutions['length_preprocess'] = \
    data_constitutions['constitution_lemma'].apply(
    lambda x: len(x.split()))

# Save List of lengths 'cause CPP data has to many empty values in our data.
better_length = data_constitutions.length_preprocess.to_list()
# =============================================================================
# Colonies Dataset
# =============================================================================
# Read colonies file (download from COLDAT dataset)
colonies = pd.read_csv(fr'{COLONIES_FILES}COLDAT_colonies.csv',
                       encoding="cp1252",
                       engine='python')

# Only keep what is important to us for now (dummies for colonizer)
colonies = colonies.iloc[:, 0:9]

# =============================================================================
# Get the ALPHA 3 CODE
# =============================================================================
countries = list(colonies.country)


def get_codes(list_countries):
    """Uses pycountry package to get codes of countries from dataset.

    Args:
        countries (list): List of countries

    Returns:
        list: list of codes that pycountry found
    """
    countries_dict = {}
    for country in pycountry.countries:
        countries_dict[country.name] = country.alpha_3

    codes_list = [countries_dict.get(country, 'Unknown code') for country
                  in list_countries]

    return codes_list


codes = get_codes(countries)

# Add this codes to dataframes
colonies.insert(1, 'code', codes)

# From the colonies_trouble we saw the order of the countries and created a
# list "by hand" of the correct ISO 3-ALpha abbreviations
corrector = ['ATG', 'BOL', 'BIH', 'BRN', 'CPV', 'COG', 'COD',
             'CIV', 'IRN', 'KOS', 'LAO', 'MKD', 'FSM', 'MDA',
             'MMR', 'PRK', 'RUS', 'STP', 'KOR', 'KNA', 'LCA',
             'VCT', 'SWZ', 'SYR', 'TWN', 'TZA', 'TTO', 'VEN',
             'VDR']


def get_new_codes(df_colonies, list_corrector):
    """Get dataframes with wrong codes -changing it to right - and with rights.

    We then concatenated then both
    Args:
        colonies (DataFrame): dataframe
        corrector (list): list of right colonies

    Returns:
        Dataframe: Df changing the column of codes with the right ones.
    """
    colonies_trouble = df_colonies[df_colonies['code'] == 'Unknown code']
    colonies_trouble.loc[:, ('code')] = list_corrector
    colonies_right = df_colonies[df_colonies['code'] != 'Unknown code']

    colonies_new_df = pd.concat([colonies_right, colonies_trouble],
                                ignore_index=True).sort_values(
        by='country'
    ).reset_index()
    colonies_new_df = colonies_new_df.drop('index', axis=1)

    return colonies_new_df


colonies_new = get_new_codes(colonies, corrector)

# =============================================================================
# Cppc dataset - We get the number of amendments and number of words in
# each constitution
# =============================================================================
cpp_data = pd.read_csv(rf'{CCP_FILE}/ccpcnc_v3_small.csv')
cpp_data = cpp_data[['country', 'year', 'length', 'evnttype']]

# Get a dataframe with total number of Amendments


def get_amendments(cpp_data_df):
    """ Calculates total amount of amendments (evnttype=1) for each country."""
    grouped = cpp_data_df.groupby('country')['evnttype'].agg(
        lambda x: x.eq(1).sum())
    grouped = grouped.reset_index()
    grouped.columns = ['country', 'num_amendments']

    return grouped


df_grouped = get_amendments(cpp_data)

# Save this data of amendments
df_grouped.to_csv('results/csv/amendments.csv')

# We can drop evnttype now
cpp_data = cpp_data.drop('evnttype', axis=1)

# VERY IMPORTANT: we have constitutions from CPP project and we are using
# one CPP dataset. However, there seems to be something strange in the dataset
# cause, for example, Afghanistan 1987 were create a new constitution
# but the new length only happens 3 year later. There are 145 cases like that.
# Thus we took the lemmatized, cleaned, length of the texts.

# cpp_data_treated = cpp_data.fillna(method='bfill')
# =============================================================================
# Open csv file containing codes for the constitutions.csv as well
# =============================================================================
codes_df = pd.read_csv(r'Data/country_codes/codes.csv', sep=';')

data_constitutions['codes'] = data_constitutions.merge(
    codes_df,
    on='country')['country_code']


# Change columns order
data_constitutions = data_constitutions[['country', 'codes', 'year',
                                         'length_preprocess', 'document']]

# =============================================================================
# Merging Everything
# =============================================================================
data_add = pd.merge(left=data_constitutions, right=cpp_data,
                    left_on=(['country', 'year']),
                    right_on=['country', 'year'],
                    how='left', copy=False)
# Changing column names
data_add.columns = ['country', 'codes', 'year', 'length', 'document',
                    'length_nans']

# Final merged dataframe
df = pd.merge(left=data_add, right=colonies,
              left_on='codes', right_on='code', how='left')
df = df.drop('country_y', axis=1)
df = df.drop('code', axis=1)
df.columns = ['country', 'code', 'year', 'length', 'document', 'length_nans',
              'col.belgium',
              'col.britain', 'col.france', 'col.germany', 'col.italy',
              'col.netherlands', 'col.portugal', 'col.spain']

# There are some countries that were not on the colonial dataset.
# We fill tha na values with zeros since the were never colonies (like
# Yugoslavia)
cols = ['col.belgium', 'col.britain', 'col.france', 'col.germany', 'col.italy',
        'col.netherlands', 'col.portugal', 'col.spain']
df[cols] = df[cols].fillna(value=0)

profile = ProfileReport(df, title="Pandas Profiling Report")

# Saving Dataframe
df.to_csv('results/csv/constitutions_for_stm.csv')
