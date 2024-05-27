'''
2021_10_28

Sys.args:
[1] sample size
[2] train size
[3] random state

Clean raw data (from 1_import_data.py)
1. Create clean study table for building hmms, this includes creating a composite end point column of death or admission
2. Split into 6 subgroup dataframes of patient ids (age groups x3, admission/not x2)
3. Divide these into train-test split so that the training set is equal for all subsets in number of patients
    (and small enough to train in a sensible amount time)
    The number of patients to be included in the model is entered at the command line (argv[1])
4. Pickle all of these dfs of ids
'''


import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import sys
import os
from tqdm import tqdm

os.chdir('p:/postb/work/hmm_work/hmm_1_0')

# hide warnings (SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame)
pd.options.mode.chained_assignment = None

# read in composite outcome table created by script 1_import_data.py
outcome_df = pd.read_pickle('p:/postb/work/hmm_work/hmm_1_0/hmm_pkl_dfs/hmm_1_0_composite_outcome_table_2018')


# split into age-group and admit/not-admit
def split_age_admissions(df, study_year):
    '''
    :param df: OUTCOME dataframe
    :param study_year: target year for composite endpoint of death/admission
    :return: a dictionary of dfs containing the ids for patients by age group and end_point_status
    '''
    # corrected age for year of study (beginning of study period is 2 years before target year)
    df['corr_age'] = df['AGE_2021'] - (2021-(study_year-2))
    # remove patients <18
    df = df[df['corr_age'] >= 18]
    # get a single table of patient ids with their age and admission status
    ids = df[['ALF_PE', 'corr_age', 'dead_or_admit']].drop_duplicates()

    # divide by age group
    young = ids.query('corr_age <40')
    middle = ids.query('corr_age >=40 and corr_age <80')
    old = ids.query('corr_age >=80')

    # divide by admission category
    datasets = {}
    for frame, name in zip([young, middle, old], ['young', 'middle', 'old']):
        datasets[f"{name}_y"] = frame.query('dead_or_admit == 1')
        datasets[f"{name}_n"] = frame.query('dead_or_admit == 0')

    return datasets


# get subsample of datasets above
def select_sub_sample(dictionary, sample_size, random_state):
    sub_ids_dict = {}
    for key in dictionary.keys():
        sub_ids_dict[key] = dictionary[key].sample(n = sample_size, random_state=random_state)

    return sub_ids_dict


# create study table with every date in 2 years and attendances
def create_study_table(outcomes_df, sub_ids):
    # note the 'outcome_df' refers to the imported table with the composite outcome

    # unpack dictionary to a table
    columns = sub_ids['young_y'].columns
    sub_ids_table = pd.DataFrame(columns=columns)

    for key in sub_ids.keys():
        sub_ids_table = sub_ids_table.append(sub_ids[key])

    # read in main study table and keep only data associated with sub_ids
    ids = sub_ids_table['ALF_PE'].unique()
    df = outcomes_df[outcomes_df['ALF_PE'].isin(ids)]

    # convert col to datetime for later merging
    df['EVENT_DT'] = pd.to_datetime(df['EVENT_DT'])

    # make dataframe of all dates for study period
    dates = pd.date_range(start='2016-01-01', end='2017-12-31')
    dates = pd.DataFrame(dates)
    dates.columns = ['date_list']

    # create empty df, that will be the final df to return
    study_df = pd.DataFrame(columns=['ALF_PE','date_list', 'attendance', 'AGE_2021', 'dead_or_admit'])

    for i in tqdm(ids):
        pt_df = df[df['ALF_PE'] == i]
        # indicator col to show where a patient attended
        pt_df['attendance'] = 1
        # merge with dates from study period
        sub_df = dates.merge(pt_df, how='left', left_on='date_list', right_on='EVENT_DT')
        # fill all NaN with 0 (NaN indicates a patient didn't attend on this date -> replace with 0)
        sub_df['attendance'] = sub_df['attendance'].fillna(0)
        # fill df with patient's id
        for i in ['ALF_PE', 'AGE_2021', 'dead_or_admit']:
            sub_df[i] = sub_df[i].fillna(sub_df[i].max())
        df_to_append = sub_df[['ALF_PE','date_list', 'attendance', 'AGE_2021', 'dead_or_admit']]
        study_df = study_df.append(df_to_append)

    path = 'p:/postb/work/hmm_work/hmm_1_0/hmm_pkl_dfs/'
    # sample_size = sample_size
    # train_size = train_size
    pkl_name = 'study_df' + '_sample_' + str(sample_size) + '_train_' + str(train_size)  + '_randstate_' + str(random_state)
    pickle.dump(study_df, open(path+pkl_name,'wb'))

    return study_df


# using dictionary from above (created by 'select_sub_sample') and get train/test split
def train_test_split_subsets(dict, train_size, random_state):
    '''
    :param dict: dictionary of subset ids (created by split_age_admissions function)
    :param train_size: ideally specify n (rather than ratio) so that all hmms are trained on the same number of patients
    :param random_state: set random state
    :return: 12 dataframes of patient ids ready for train-test
    '''
    train_test_datasets = {}
    for key in dict.keys():
        train_test_datasets[f"{key}_train"], train_test_datasets[f"{key}_test"] = train_test_split(dict[key], train_size=train_size, random_state=random_state)

    return train_test_datasets


def pickle_subsets(dict):
    '''
    :param dict: dictionary of sub-divided dfs created by split_age_admission
    :param path: path for files to be pickled to
    :param train_size: number of patients to be included in training model
    :return: nil (pickled files saved
    '''
    path = 'p:/postb/work/hmm_work/hmm_1_0/hmm_pkl_dfs/'
    for name in dict.keys():
        pkl_name = str(name) + '_sample_' + str(sample_size) + '_train_' + str(train_size) + '_randstate_' + str(random_state)
        pickle.dump(dict[name], open(path+pkl_name,'wb'))


if __name__ == "__main__":
    sample_size = int(sys.argv[1])
    train_size = int(sys.argv[2])
    random_state = int(sys.argv[3])

    # sample_size = 3000
    # train_size = 1000
    # random_state = 101

    all_ids = split_age_admissions(outcome_df, study_year=2018)
    print('got all ids')
    sub_ids = select_sub_sample(all_ids, sample_size=sample_size, random_state=random_state)
    print('got sub ids')
    study_df = create_study_table(outcome_df, sub_ids)
    print('study table created')
    train_test_datasets = train_test_split_subsets(sub_ids, train_size=train_size, random_state=random_state)
    print('train-test split done')
    pickle_subsets(train_test_datasets)
    print('train-test ids pickled!')