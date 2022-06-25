# -*- coding: utf-8 -*-
"""
Add a lot of covariates.

We took data from a lot o sources, from Word Bank and Polity 4
to some papers (see https://scholar.harvard.edu/shleifer/publications)

@author: marcos
"""
import pandas as pd
import numpy as np

polity = pd.read_excel(r'Data/polity5/polity5v2018.xlsx')

df_grouped = polity.groupby('scode')
dfs = dict(tuple(df_grouped))

for country in df_grouped:
    