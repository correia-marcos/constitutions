# -*- coding: utf-8 -*-
"""
Add a lot of covariates.

We took data from a lot o sources, from Word Bank and Polity 4
to some papers (see https://scholar.harvard.edu/shleifer/publications)

@author: marcos
"""
import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
FOLDER = r'results/csv'

data = pd.read_csv(fr'{FOLDER}/all_textual_variables(03_06).csv',
                   index_col=False)
data = data.drop('Unnamed: 0', axis=1)


# The way we exported data from Word_bank, the dataset were in a
# "tidy data" format that I don't use much and is not good for regressions.
# The function below was done to create a dataset
# in the right way for running regressions.


def get_table(dataset_text, covariates_data, index='Series Name'):
    """Get the merged dataframe with dataset_text and covariates or others.

    Args:
        dataset_text (DataFrame): Dataframe created before with textual_data
        covariates_data (DataFrame): all Dataframes from World Bank works

    Returns:
        DataFrame: merge dataframe
    """
    covariates_data = covariates_data.set_index(index)
    df_grouped = covariates_data.groupby([index])
    dfs = dict(tuple(df_grouped))

    dataframe = dataset_text.copy()
    for variable in dfs:
        columns = list(dfs[variable].columns)
        for i in range(len(columns)-1):
            columns[i+1] = variable + ' - ' + str(columns[i+1])
        dfs[variable].columns = columns
        dataframe = dataframe.merge(dfs[variable], left_on='code',
                                    right_on='codes', how='left',
                                    suffixes=('', '_delme'))
    # We have a lot of columns with same labels and values, so we removed then
    dataframe = dataframe[[c for c in dataframe.columns if not
                           c.endswith('_delme')]]

    return dataframe

# =============================================================================
# COVARIATES FROM DIFFERENT DATASETS
# =============================================================================

# AHDI FROM LAPORTA
# Adding info from the AHDI file in Data/AHDI


ahdi = pd.read_excel('Data/AHDI/AHDI_1.1-1.xlsx', sheet_name='AHDI')
ahdi.columns = ['code', 'country_ahdi', 'ahdi_1870', 'ahdi_1880',
                'ahdi_1890',
                'ahdi_1900', 'ahdi_1913', 'ahdi_1925', 'ahdi_1929',
                'ahdi_1933', 'ahdi_1938', 'ahdi_1950', 'ahdi_1955',
                'ahdi_1960', 'ahdi_1965', 'ahdi_1970', 'ahdi_1975',
                'ahdi_1980', 'ahdi_1985', 'ahdi_1990', 'ahdi_1995',
                'ahdi_2000', 'ahdi_2005', 'ahdi_2007', 'ahdi_2010',
                'ahdi_2015']
# =============================================================================
# SOME COVARIATES FROM WORD BANK
# =============================================================================

# Adding others excel files with control and dependent variable for
# future regressions
# Covariates_1 is the bigger file and covariates_2 add some variables like
# total debt service (% of GNI)
covariates1 = pd.read_excel(r'Data/Word_bank/covariates_1.xlsx',
                            sheet_name='Data')
# We must remove unnecessary columns
covariates1 = covariates1.drop('Series Code', axis=1)
covariates1 = covariates1.drop('Country Name', axis=1)

data_cov = get_table(data, covariates1)

# ==============================================================================
# Covariates_2 data from Word Bank
# ==============================================================================
covariates2 = pd.read_excel(r'Data/Word_bank/covariates_2.xlsx',
                            sheet_name='Data')

data_cov = get_table(data_cov, covariates2)

# =============================================================================
# GOVERNANCE INDICATORS FROM WORLD BANK
# =============================================================================

governance = pd.read_excel(r'Data/Word_bank/Governance_indicators.xlsx',
                           sheet_name='Data')

data_cov = get_table(data_cov, governance)

# WORLD BANK REAL INTEREST
# adding real interest
# =============================================================================
real_interest = pd.read_excel(r'Data/Word_bank/Real_interest_rate.xlsx',
                              sheet_name='Data')

data_cov = get_table(data_cov, real_interest, index='Indicator Name')

# We noticed some columns were not read as number. We noticed that empty values
# were coded as ".." which make pythons misunderstand the variable type
data_cov = data_cov.replace("..", np.nan)

# =============================================================================
# CPP PROJECT COVARIATES (SIZE OF CONSTITUTIONS, AMOUNT OF AMENDMENTS)
# =============================================================================
cpp_data = pd.read_csv(fr'{FOLDER}/constitutions_for_stm.csv')
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
df_added = pd.merge(data_cov, cpp_data, how='left', on='codes',
                    copy=False)

# Dropping unnecessary columns(like percentile estimates, number of sources
# and estimates of standard errors)
df_added = df_added[df_added.columns.drop(list(df_added.filter(
    regex='Percentile')))]
df_added = df_added[df_added.columns.drop(list(df_added.filter(
    regex='Source')))]
df_added = df_added[df_added.columns.drop(list(df_added.filter(
    regex='Standard Error')))]

# =============================================================================
# DATA FROM The Economic Consequences of Legal Origins (2008)
# =============================================================================
legal_origins = pd.read_excel(r'Data/legal_origins_paper/Data.xlsx',
                              sheet_name='Table 1')

# Removing unnecessary columns - we only need country code and legal origin
legal_origins = legal_origins[['code', 'legor_uk', 'legor_fr',
                               'legor_ge', 'legor_sc', 'legor_so']]

# Adding together dataframe and ahdi
final_df = pd.merge(df_added, ahdi, how='left', on='code', copy=False)
final_df = pd.merge(final_df, legal_origins, on='code', how='left', copy=False)

# Cleaning some more
final_df = final_df.drop('country_y', axis=1)
final_df = final_df.drop('codes', axis=1)
final_df = final_df.rename(columns={'country_x': 'country'})

# =============================================================================
# Adding up number of amendments
# =============================================================================
amendments = pd.read_csv(f'{FOLDER}/amendments.csv')
amendments = amendments.drop('Unnamed: 0', axis=1)
final_df = final_df.merge(amendments, on='country', how='left', copy=False)

# =============================================================================
# Adding up number of constitutions
# =============================================================================
constitutions = pd.read_csv(rf'{FOLDER}/constitutions.csv')

# Dropping have and uncesserary data

constitutions = constitutions.drop('constitution_lemma', axis=1)
constitutions = constitutions.drop('constitution', axis=1)
constitutions = constitutions.drop('document', axis=1)

# Get the number of times each country repeats in the dataset: the number of
# constitutions
constitutions = constitutions['country'].value_counts()
constitutions = constitutions.reset_index(level=0)
constitutions.columns = ['country', 'num_constitutions']

# Adding this to the final_dataset
final_df = final_df.merge(constitutions, on='country', how='left', copy=False)
# =============================================================================
# Saving Final dataset and creating
# =============================================================================

# Creating Pandas Profiling - too much variables to handle
# small_df = final_df.iloc[:, 0:50].join(final_df.iloc[:, -50:])
report = ProfileReport(final_df, title='Pandas Profile Report')

final_df.to_csv(f'{FOLDER}/all_variables_reg(03_06).csv')
# report.to_file(r'results/HTML/all_variables.html')
