# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 00:07:44 2023

@author: fayya
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
csv_path = r'C:\Users\fayya\Downloads\combined_data.csv'
df = pd.read_csv(csv_path, header=0)

#%%
import scipy.stats as stats
F = set(df.keys()).difference(['dataset', 'patient_id', 'sample_id'])
P = []
Afun = [np.mean,np.std] #aggregation or reduction function

for f in F:    
    for fun in Afun:
        dfp = df.groupby('patient_id')
        clabel = dfp.dataset.first()
        U = list(set(clabel))
        
        x = dfp[f].apply(fun)
        x_group1 = x[clabel == U[0]]
        x_group2 = x[clabel == U[1]]
        # Perform an independent two-sample t-test
        t_statistic, p_value = stats.mannwhitneyu(x_group1, x_group2)
        P.append((f,fun,p_value))
        print(P[-1])
P = np.array(P)     

from statsmodels.stats.multitest import multipletests
q = multipletests(np.array(P[:,-1],dtype = np.float))