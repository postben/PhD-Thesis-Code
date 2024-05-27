
import pandas as pd
import numpy as np
from hmmlearn.hmm import MultinomialHMM
import pickle
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt


cohort = sys.argv[1] # e.g. young_y, old_n, etc
sample_size = sys.argv[2]
train_size = sys.argv[3]
random_state = sys.argv[4]
num_states = sys.argv[5] # max number of states to plot

# read in cleaned study data frame
study_df = pd.read_pickle('p:/postb/work/hmm_work/hmm_1_0/hmm_pkl_dfs/study_df_sample_'+ str(sample_size) + '_train_' + str(train_size) + \
    '_randstate_' + str(random_state))
# specific dataframe of ids for specified subset
pt_ids = pd.read_pickle('p:/postb/work/hmm_work/hmm_1_0/hmm_pkl_dfs/' +str(cohort) + '_train_sample_' + str(sample_size) + \
                        '_train_' + str(train_size) + '_randstate_' + str(random_state))


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
    model = MultinomialHMM(n_components=n_components,
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


def plot_num_states(df, pt_ids, maxstates):

    maxstates = int(maxstates)
    performance_df = pd.DataFrame(columns=['num_states', 'aic', 'bic'])
    for i in range(1, maxstates + 1):
        _, _, aic, bic = make_hmm(df, pt_ids, i)
        performance_df = performance_df.append({'num_states': i, 'aic': aic, 'bic': bic}, ignore_index=True)
        print('finished model with %d states' % i)

    path = 'p:/postb/work/hmm_work/hmm_1_0/hmm_pkl_dfs/'
    filename = 'num_states_performance_' + str(cohort) + '_' + str(sample_size) + '_' + str(train_size) + '_randstate_' + str(random_state) +  '_numstate_' + str(num_states)
    pickle.dump(performance_df, open(path+filename,'wb'))

    fig, ax = plt.subplots(1, 1, figsize=(10,6))
    ax.plot(performance_df['num_states'], performance_df['aic'], label='aic')
    ax.plot(performance_df['num_states'], performance_df['bic'], label='bic')
    ax.set_title('number of set states vs AIC/BIC')
    ax.set_xlabel('number of states')
    ax.set_ylabel('AIC/BIC')
    ax.legend()
    plt.savefig('p:/postb/work/hmm_work/hmm_1_0/hmm_plots/' + str(cohort) + '_sample_' + str(sample_size) + '_train_size_' + str(train_size) \
                + '_randstate_' + str(random_state) + '_numstates_' + str(num_states) + '_performance.png')


if __name__ == "__main__":
    cohort = sys.argv[1] # e.g. young_y, old_n, etc
    sample_size = sys.argv[2]
    train_size = sys.argv[3]
    random_state = sys.argv[4]
    num_states = sys.argv[5] # max number of states to plot
    plot_num_states(study_df, pt_ids, num_states)