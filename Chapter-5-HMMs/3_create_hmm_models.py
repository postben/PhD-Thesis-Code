'''
2021_10_30

Sys.args:
[1] cohort (e.g. young_y, old_n)
[2] sample_size
[3] train_size
[4] rand_state
[5] number of states to model

1. Import main cleaned study table (created by 2_clean_and_split.py)
2. Import specific dataframe of subgroup for modelling e.g. hmm_pkl_dfs/young_y_train_5000 (i.e. younger age group, were admitted, 5000 patients for training)
    (this is determined at the command line with sys.argv[1] {2nd argument after this file})
3. Create hmm model (number of states is sys.argv[2])
4. Pickle hmm model and save with save name as the input subgroup
'''


import pandas as pd
import numpy as np
from hmmlearn.hmm import MultinomialHMM
import pickle
import sys
from tqdm import tqdm

cohort = sys.argv[1]
sample_size = sys.argv[2]
train_size = sys.argv[3]
rand_state = sys.argv[4]
num_states = sys.argv[5]

# read in cleaned study data frame
study_df = pd.read_pickle('p:/postb/work/hmm_work/hmm_1_0/hmm_pkl_dfs/study_df_sample_'+ str(sample_size) + '_train_' + str(train_size) + '_randstate_' + str(rand_state))
# specific dataframe of ids for specified subset
pt_ids = pd.read_pickle('p:/postb/work/hmm_work/hmm_1_0/hmm_pkl_dfs/' +str(cohort) + '_train_sample_' + str(sample_size) + '_train_' + str(train_size) + '_randstate_' + str(rand_state))


def make_hmm(big_df, pt_ids, n_components):
    from tqdm import tqdm
    # get just the patient ids from the dataset
    pt_ids = pt_ids['ALF_PE']
    # from large study table, keep data points that correspond only to this cohort
    df = big_df[big_df['ALF_PE'].isin(pt_ids)]

    # Print length of df and number of data points
    print('patient cohort: ', str(sys.argv[1]).rsplit('/',1)[-1])
    print('number of patients: ', len(pt_ids))
    print('number of datapoints: ', len(df))

    # shape data as needed by hmmlearn package
    hmm_array = []
    hmm_array_lengths = []

    for i in tqdm(pt_ids):
        x = df[df.ALF_PE == i]['attendance'].dropna().to_numpy()
        x = x.astype(int)
        hmm_array.append(x)
        hmm_array_lengths.append(len(x))

    hmm_array = np.concatenate((hmm_array))
    hmm_array_lengths = np.array(hmm_array_lengths).flatten()

    # fit model
    model = MultinomialHMM(n_components=int(n_components),
                        n_iter=100)
    model.fit(hmm_array.reshape(-1, 1), lengths=hmm_array_lengths)

    # get score and stats
    score = model.score(hmm_array.reshape(-1, 1), lengths=hmm_array_lengths)
    ll = score
    k = model.n_features
    N = len(hmm_array)
    # AIC = -2 * log-likelihood + 2*num params
    aic = -2*ll + 2*k
    # BIC = -2 * loglikelihood + log(num samples in training set)*num params
    bic = -2 * ll + np.log(N) * k

    return model, score, aic, bic


if __name__ == "__main__":
    # make model
    model,score,aic,bic = make_hmm(study_df, pt_ids, num_states)
    print('Score = ', score)
    print('aic = ', aic)
    print('bic = ', bic)
    # save model
    path = 'p:/postb/work/hmm_work/hmm_1_0/hmm_pkl_models/'
    filename = cohort + '_' + str(sample_size) + '_' + str(train_size) + '_randstate_' + str(rand_state) + '_numstates_' + str(num_states)
    pickle.dump(model, open(path+filename,'wb'))

