# -*- coding: utf-8 -*-
"""
Get the colonies of each country among other variables for STM model.

Variables:
    Dummy for country colonization
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
import ast

# place of files/folders
file_constitutions = r'results/csv/constitutions.csv'
colonies_folder = r'Data/colonies/'
cppc_file = r'Data/comparative_project/'

# Read file
data_constitutions = pd.read_csv(file_constitutions, index_col=0)

# Drop unnecessary columns
data_constitutions = data_constitutions.drop('constitution', axis=1)
data_constitutions = data_constitutions.drop('constitution_lemma', axis=1)

# Finally, make better type of object on document column (was a list and we
# want a single string)
# Attention: actually was not a list but actually a string formatted as such
# We employed this "ast" library to save trouble
data_constitutions['document'] = data_constitutions['document'].apply(
    lambda x: ast.literal_eval(x))

data_constitutions['document'] = data_constitutions['document'].apply(
    lambda x: " ".join(x))

# Add column for number of words in lemma documents
data_constitutions['length_preprocess'] = data_constitutions['document'].apply(
    lambda x: len(x.split()))
# =============================================================================
# Colonies Dataset
# =============================================================================
# Read colonies file (download from COLDAT dataset)
colonies = pd.read_csv(fr'{colonies_folder}COLDAT_colonies.csv',
                       encoding="cp1252",
                       engine='python')

# Only keep what is important to us for now (dummies for colonizer)
colonies = colonies.iloc[:, 0:9]

# =============================================================================
# Get the ALPHA 3 CODE
# =============================================================================
countries = list(colonies.country)

countries_dict = {}
for country in pycountry.countries:
    countries_dict[country.name] = country.alpha_3

codes = [countries_dict.get(country, 'Unknown code') for country
         in countries]

# Add this codes to dataframes
colonies.insert(1, 'code', codes)

# IMPORTANT: some countries were not find by the pycountry. We worked through
# "by hand".

# =============================================================================
# Cppc dataset - We get the number of amendments and number of words in
# each constitution
# =============================================================================
cpp_data = pd.read_csv(rf'{cppc_file}/ccpcnc_v3_small.csv')
cpp_data = cpp_data[['country', 'year', 'length', 'evnttype']]

# Get a dataframe with total number of Amendments
df_grouped = cpp_data.groupby('country')['evnttype'].agg(
    lambda x: x.eq(1).sum())
df_grouped = df_grouped.reset_index()
df_grouped.columns = ['country', 'num_amendments']

# Save this data of amendments
df_grouped.to_csv('results/csv/amendments.csv')

# We can drop evnttype now
cpp_data = cpp_data.drop('evnttype', axis=1)

# VERY IMPORTANT: we have constitutions from CPP project and we are using
# one CPP dataset. However, there seems to be some strange in the dataset
# cause, for example, Afghanistan 1987 were create a new constitution
# but the new length only happens 3 year later. We took the amount of words
# in txt text we had for them.

cpp_data_treated = cpp_data.fillna(method='bfill')
# =============================================================================
# Open csv file containing codes for the constitutions.csv as well
# =============================================================================
codes_df = pd.read_csv(r'Data/country_codes/codes.csv', sep=';')

data_constitutions['codes'] = data_constitutions.merge(
    codes_df,
    on='country')['country_code']


# Change columns order
data_constitutions = data_constitutions[['country', 'codes',
                                         'year', 'document']]

# =============================================================================
# Merging Everting
# =============================================================================
data_add = pd.merge(left=data_constitutions, right=cpp_data,
                    left_on=(['country', 'year']),
                    right_on=['country', 'year'],
                    how='left', copy=False)

# Final merged dataframe
df = pd.merge(left=data_add, right=colonies,
              left_on='codes', right_on='code', how='left')
df = df.drop('country_y', axis=1)
df = df.drop('code', axis=1)
df.columns = ['country', 'code', 'year', 'document', 'col.belgium',
              'col.britain', 'col.france', 'col.germany', 'col.italy',
              'col.netherlands', 'col.portugal', 'col.spain']

# There are some countries that were not on the colonial dataset.
# We fill tha na values with zeros since the were never colonies (like
# Yugoslavia)
cols = ['col.belgium', 'col.britain', 'col.france', 'col.germany', 'col.italy',
        'col.netherlands', 'col.portugal', 'col.spain']
df[cols] = df[cols].fillna(value=0)

# Saving Dataframe
df.to_csv('results/csv/constitutions_for_stm.csv')
