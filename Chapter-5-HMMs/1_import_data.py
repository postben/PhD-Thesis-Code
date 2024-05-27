'''
2021_10_22
Get raw dataset from SQL database
'''

import SAIL_python
import pandas as pd
import pickle
from tqdm import tqdm
import os
os.chdir('p:/postb/work/hmm_work/hmm_1_0')

# create connection
conn=SAIL_python.connect()

# GET ALL VISITS FOR 2016/2017
df_raw = pd.read_sql_query(
    '''
    SELECT 
    ALF_PE
    ,EVENT_DT
    ,YR
    ,AGE_2021
    ,ADM_2018
    FROM SAILW1323V.BP_GP_INTERACTIONS_WITH_ADMISSIONS
    where (YR = 2016 or yr = 2017)
    and (AGE_2021 >= 21 or AGE_2021  <=113) -- this means that the cohort is between 18-110 years old
    ''', conn)


# save above as pickled df (takes a long time to read-in)
filename= 'raw_gp_interactions_16_17'
pickle.dump(df_raw, open('hmm_pkl_dfs/'+filename,'wb'))



# get all deaths (table created by me in DB2)
deceased = pd.read_sql_query('SELECT * FROM SAILW1323V.BP_DECEASED', conn)

# save above as pickled df
filename= 'all_deaths'
pickle.dump(deceased, open('hmm_pkl_dfs/'+filename,'wb'))


# create a dataframe of patients with a composite outcome of death/admission

def create_deadadmit_table(df, deceased_table, study_year):
    from tqdm import tqdm
    # df is big main raw table above
    # get patients who died in target year and merge with main table
    died = deceased_table[deceased_table['DEATH_YEAR'] == study_year][['ALF_PE','DEATH_YEAR']]
    df = df.merge(died,how ='left', on = 'ALF_PE')
    # drop patients that died before the target year
    df = df[~df.ALF_PE.isin(deceased_table[deceased_table['DEATH_YEAR'] < study_year]['ALF_PE'])]

    # create a column to indicate if a patient died or was admitted in 2018
    def death_admit(dataframe):
        if (np.isnan(dataframe['ADM_2018']) and np.isnan(dataframe['DEATH_YEAR'])):
            return 0
        else:
            return 1

    tqdm.pandas()
    # this takes over an hour to run!!!
    df['dead_or_admit'] = df.progress_apply(death_admit, axis=1)

    # save this!
    filename= 'hmm_1_0_composite_outcome_table_' + str(study_year)
    pickle.dump(df, open('p:/postb/work/hmm_work/hmm_1_0/hmm_pkl_dfs/'+filename,'wb'))


