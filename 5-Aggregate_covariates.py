# -*- coding: utf-8 -*-
"""
Add a lot a covariates.

@author: marcola
"""
import pandas as pd
import numpy as np

folder = r'results/csv'

# =============================================================================
# =============================================================================
# =============================================================================
# Again let's divide in sequence databases and Prime (comparing each
# constitution by the first one)
# =============================================================================
# =============================================================================
# =============================================================================
data = pd.read_csv(fr'{folder}/all_textual_variables(03_06).csv',
                   index_col=False)
data = data.drop('Unnamed: 0', axis=1)

# =============================================================================
# =============================================================================
# COVARIATES FROM DIFFERENTS DATSETS
# =============================================================================
# =============================================================================


# =============================================================================
# AHDI FROM LAPORTA
# =============================================================================
# Adding info from the AHDI file in Data/AHDI
ahdi = pd.read_excel('Data/AHDI/AHDI_1.1-1.xlsx', sheet_name='AHDI')
ahdi.columns = ['codes', 'country_ahdi', 'ahdi_1870', 'ahdi_1880',
                'ahdi_1890',
                'ahdi_1900', 'ahdi_1913', 'ahdi_1925', 'ahdi_1929',
                'ahdi_1933', 'ahdi_1938', 'ahdi_1950', 'ahdi_1955',
                'ahdi_1960', 'ahdi_1965', 'ahdi_1970', 'ahdi_1975',
                'ahdi_1980', 'ahdi_1985', 'ahdi_1990', 'ahdi_1995',
                'ahdi_2000', 'ahdi_2005', 'ahdi_2007', 'ahdi_2010',
                'ahdi_2015']

ahdi_data = pd.merge(data, ahdi, left_on='codes', right_on='codes',
                     how='left')
ahdi_data = ahdi_data.drop('country_ahdi', 1)
# Saving data
# ahdi_data.to_csv(f'{folder}/variables_plus_ahdi(06_05).csv')


# =============================================================================
# SOME COVARIATES FROM WORD BANK
# =============================================================================
# Adding others excel files with control and dependent variable for
# future regressions
covariates = pd.read_excel(r'Data/Word_bank/Covariates_2.xlsx',
                           sheet_name='Data')
covariates = covariates.iloc[:, :-4]

# The way we exported data from Word_bank, the dataset were in a
# "tidy data" format that I don't use much and is not good for regressions.
# This was done to create a dataset
# in the right way for running regressions.
covariates = covariates.set_index('Series Name')
df_grouped = covariates.groupby(['Series Name'])
dfs = dict(tuple(df_grouped))

dataframe = data.copy()
for variable in dfs:
    columns = list(dfs[variable].columns)
    for i in range(len(columns)-1):
        columns[i+1] = variable + ' - ' + str(columns[i+1])
    dfs[variable].columns = columns
    dataframe = dataframe.merge(dfs[variable], on='codes',
                                how='left')

# =============================================================================
# GOVERNANCE INDICADORS FROM WORDBANK
# =============================================================================
# Now, adding governance indicators

governance = pd.read_excel(r'Data/Word_bank/Governance_indicators.xlsx',
                           sheet_name='Data')
governance = governance.set_index('Series Name')
df_grouped = governance.groupby(['Series Name'])
dfs = dict(tuple(df_grouped))

for variable in dfs:
    columns = list(dfs[variable].columns)
    for i in range(len(columns)-1):
        columns[i+1] = variable + ' - ' + str(columns[i+1])
    dfs[variable].columns = columns
    dataframe = dataframe.merge(dfs[variable], on='codes',
                                how='left')

# =============================================================================
# WORDBANK REAL INTEREST
# =============================================================================
# Finally, adding real interest

real_interest = pd.read_excel(r'Data/Word_bank/Real_interest_rate.xlsx',
                              sheet_name='Data')
real_interest = real_interest.set_index('Indicator Name')
df_grouped = real_interest.groupby(['Indicator Name'])
dfs = dict(tuple(df_grouped))

for variable in dfs:
    columns = list(dfs[variable].columns)
    for i in range(len(columns)-1):
        columns[i+1] = variable + ' - ' + str(columns[i+1])
    dfs[variable].columns = columns
    dataframe = dataframe.merge(dfs[variable], on='codes',
                                how='left')

# We noticed some columns were not read as number. We noticed that empty values
# were coded as ".." which make pythons misunderstand the variable type
dataframe = dataframe.replace("..", np.nan)

# =============================================================================
# CPP PROJECT COVARIATES (SIZE OF CONSTITUTIONS, AMOUNT OF EMENDMENTS)
# =============================================================================
cpp_data = pd.read_csv(fr'{folder}/constitutions_for_stm.csv')
cpp_data = cpp_data.drop('document', axis=1)
cpp_data = cpp_data.drop('Unnamed: 0', axis=1)
cpp_data.columns = ['country', 'codes', 'year', 'length_preprocess_lastconst',
                    'length_lastconst',
                    'col.belgium', 'col.britain', 'col.france', 'col.germany',
                    'col.italy', 'col.netherlands', 'col.portugal',
                    'col.spain']

cpp_data = cpp_data.drop_duplicates(['country'], keep='last')
cpp_data = cpp_data.drop('year', axis=1)


# Merging with the dataframe
df_added = pd.merge(dataframe, cpp_data, how='left', on='codes',
                    copy=False)

# Dropping unecessary columns(like percentile estimates, number of sourcers
# and estimates of standard errors)
df_added = df_added[df_added.columns.drop(list(df_added.filter(
    regex='Percentile')))]
df_added = df_added[df_added.columns.drop(list(df_added.filter(
    regex='Source')))]
df_added = df_added[df_added.columns.drop(list(df_added.filter(
    regex='Standard Error')))]

# Adding together dataframe and ahdi
final_df = pd.merge(ahdi, df_added, how='outer', on='codes', copy=False)
final_df = final_df.drop('country_x', axis=1)
final_df.to_csv(f'{folder}/all_variables_reg(03_06).csv')


# =============================================================================
# =============================================================================
# =============================================================================
# SEQUENCE OF DATAFRAMES
# DATAFRAMES THAT SUM ALL CHANGE BETWEEN EACH CONST AND PREDECESSOR
# =============================================================================
# =============================================================================
# =============================================================================
data = pd.read_csv(f'{folder}/all_textual_variables_seq(03_06).csv',
                   index_col=False)
data = data.drop('Unnamed: 0', axis=1)

# =============================================================================
# =============================================================================
# COVARIATES FROM DIFFERENTS DATSETS
# =============================================================================
# =============================================================================


# =============================================================================
# AHDI FROM LAPORTA
# =============================================================================
# Adding info from the AHDI file in Data/AHDI
ahdi = pd.read_excel('Data/AHDI/AHDI_1.1-1.xlsx', sheet_name='AHDI')
ahdi.columns = ['codes', 'country_ahdi', 'ahdi_1870', 'ahdi_1880',
                'ahdi_1890',
                'ahdi_1900', 'ahdi_1913', 'ahdi_1925', 'ahdi_1929',
                'ahdi_1933', 'ahdi_1938', 'ahdi_1950', 'ahdi_1955',
                'ahdi_1960', 'ahdi_1965', 'ahdi_1970', 'ahdi_1975',
                'ahdi_1980', 'ahdi_1985', 'ahdi_1990', 'ahdi_1995',
                'ahdi_2000', 'ahdi_2005', 'ahdi_2007', 'ahdi_2010',
                'ahdi_2015']

ahdi_data = pd.merge(data, ahdi, left_on='codes', right_on='codes',
                     how='left')
ahdi_data = ahdi_data.drop('country_ahdi', 1)
# Saving data
# ahdi_data.to_csv(f'{folder}/variables_plus_ahdi(06_05).csv')


# =============================================================================
# SOME COVARIATES FROM WORD BANK
# =============================================================================
# Adding others excel files with control and dependent variable for
# future regressions
covariates = pd.read_excel(r'Data/Word_bank/Covariates_2.xlsx',
                           sheet_name='Data')
covariates = covariates.iloc[:, :-4]

# The way we exported data from Word_bank, the dataset were in a
# "tidy data" format that I don't use much and is not good for regressions.
# This was done to create a dataset
# in the right way for running regressions.
covariates = covariates.set_index('Series Name')
df_grouped = covariates.groupby(['Series Name'])
dfs = dict(tuple(df_grouped))

dataframe = data.copy()
for variable in dfs:
    columns = list(dfs[variable].columns)
    for i in range(len(columns)-1):
        columns[i+1] = variable + ' - ' + str(columns[i+1])
    dfs[variable].columns = columns
    dataframe = dataframe.merge(dfs[variable], on='codes',
                                how='left')

# =============================================================================
# GOVERNANCE INDICADORS FROM WORDBANK
# =============================================================================
# Now, adding governance indicators

governance = pd.read_excel(r'Data/Word_bank/Governance_indicators.xlsx',
                           sheet_name='Data')
governance = governance.set_index('Series Name')
df_grouped = governance.groupby(['Series Name'])
dfs = dict(tuple(df_grouped))

for variable in dfs:
    columns = list(dfs[variable].columns)
    for i in range(len(columns)-1):
        columns[i+1] = variable + ' - ' + str(columns[i+1])
    dfs[variable].columns = columns
    dataframe = dataframe.merge(dfs[variable], on='codes',
                                how='left')

# =============================================================================
# WORDBANK REAL INTEREST
# =============================================================================
# Finally, adding real interest

real_interest = pd.read_excel(r'Data/Word_bank/Real_interest_rate.xlsx',
                              sheet_name='Data')
real_interest = real_interest.set_index('Indicator Name')
df_grouped = real_interest.groupby(['Indicator Name'])
dfs = dict(tuple(df_grouped))

for variable in dfs:
    columns = list(dfs[variable].columns)
    for i in range(len(columns)-1):
        columns[i+1] = variable + ' - ' + str(columns[i+1])
    dfs[variable].columns = columns
    dataframe = dataframe.merge(dfs[variable], on='codes',
                                how='left')

# We noticed some columns were not read as number. We noticed that empty values
# were coded as ".." which make pythons misunderstand the variable type
dataframe = dataframe.replace("..", np.nan)

# =============================================================================
# CPP PROJECT COVARIATES (SIZE OF CONSTITUTIONS, AMOUNT OF EMENDMENTS)
# =============================================================================
cpp_data = pd.read_csv(fr'{folder}/constitutions_for_stm.csv')
cpp_data = cpp_data.drop('document', axis=1)
cpp_data = cpp_data.drop('Unnamed: 0', axis=1)
cpp_data.columns = ['country', 'codes', 'year', 'length_preprocess_lastconst',
                    'length_lastconst',
                    'col.belgium', 'col.britain', 'col.france', 'col.germany',
                    'col.italy', 'col.netherlands', 'col.portugal',
                    'col.spain']

cpp_data = cpp_data.drop_duplicates(['country'], keep='last')
cpp_data = cpp_data.drop('year', axis=1)


# Merging with the dataframe
df_added = pd.merge(dataframe, cpp_data, how='left', on='codes',
                    copy=False)

# Dropping unecessary columns(like percentile estimates, number of sourcers
# and estimates of standard errors)
df_added = df_added[df_added.columns.drop(list(df_added.filter(
    regex='Percentile')))]
df_added = df_added[df_added.columns.drop(list(df_added.filter(
    regex='Source')))]
df_added = df_added[df_added.columns.drop(list(df_added.filter(
    regex='Standard Error')))]


# Adding together dataframe and ahdi
final_df = pd.merge(ahdi, df_added, how='outer', on='codes', copy=False)

# Droping and changing var name
final_df = final_df.drop('country_x', axis=1)
final_df = final_df.rename(columns={'country_ahdi': 'country'})

# GET THE DATA WITH AMENDMENTS
amend = pd.read_csv(f'{folder}/ammendmnts.csv', index_col=False)
amend = amend.drop('Unnamed: 0', axis=1)

final_df = final_df.merge(amend, on='country', how='left')

final_df.to_csv(f'{folder}/all_variables_reg_seq(03_06).csv')
