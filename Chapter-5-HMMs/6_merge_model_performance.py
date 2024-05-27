'''
2021-11-06
Merge model performance results from different random states used
'''

import pandas as pd
import numpy as np
from hmmlearn.hmm import MultinomialHMM
import pickle
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

os.chdir('p:/postb/work/hmm_work/hmm_1_0')

numstates = sys.argv[1] #sets the number of states of data tested to use (created by 5_explore_num_states.py)


cohorts = [
    'young',
    'middle'
    ,'old'
    ]
admission_status = [
    'y'
    ,'n'
]

def plot_all_performances(cohort,admission_status, numstates, aic_or_bic = 'BIC'):
    data = {}
    randstate = ['101','102','103','104','105']
    # get all dfs for relevant cohort
    for state in randstate:
        for status in admission_status:
            data[f"{state}"] = pd.read_pickle('hmm_pkl_dfs/num_states_performance_' + str(cohort) + '_' + str(status) + '_5000_1000_randstate_' +
                                          str(state) + '_numstate_' + str(numstates))

    # create empty data frame to merge all of above dfs
    df = pd.DataFrame(range(1,int(numstates)+1),columns = ['num_states'])
    for key in data.keys():
        if aic_or_bic == 'AIC':
            df[f"{key}_aic"] = data[key]['aic']
        elif aic_or_bic == 'BIC':
            df[f"{key}_bic"] = data[key]['bic']
        else:
            print('error: choose aic or bic')

    # do stats
    import statsmodels.stats.api as sms

    # get mean of data
    mean = df.mean(numeric_only=True,axis=1)

    # get CIs of data for each state
    def lower_ci(x):
        lower_ci, _ = sms.DescrStatsW(x).tconfint_mean()
        return lower_ci
    def upper_ci(x):
        _, upper_ci = sms.DescrStatsW(x).tconfint_mean()
        return upper_ci

    lower_ci = df.apply(lambda x: lower_ci(x), axis=1)
    upper_ci = df.apply(lambda x: upper_ci(x), axis=1)

    # join into
    df = pd.concat([df,mean,lower_ci,upper_ci], axis=1)
    df = df.rename(columns={0:'mean',1:'lower_ci',2:'upper_ci'})

    from matplotlib.ticker import MaxNLocator
    fig, (ax1, ax2) = plt.subplots(2,1,figsize = (10,10))
    ax1.plot(df['num_states'], df['mean'])
    ax1.fill_between(df['num_states'],df['lower_ci'], df['upper_ci'],color='red',alpha=0.2)
    ax1.set_xticks(np.arange(min(df['num_states']), max(df['num_states']+1),1))
    ax1.set_xlabel('Number of States')
    ax1.set_ylabel(str(aic_or_bic))
    if admission_status == 'y':
        admission_label = 'admitted'
    elif admission_status == 'n':
        admission_label = 'not admitted'
    else:
        pass
    if cohort == 'young':
        cohort_label = '<40'
    elif cohort == 'middle':
        cohort_label = '40-80'
    elif cohort == 'old':
        cohort_label = '>80'
    ax1.set_title('Combined Model Performance for 5 Model Runs (' + aic_or_bic + ') with 95% CIs')

    ax2.plot(df['num_states'], df['mean'])
    ax2.set_xticks(np.arange(min(df['num_states']), max(df['num_states']+1),1))
    ax2.set_xlabel('Number of States')
    ax2.set_ylabel('BIC')
    ax2.set_title('Model Performance (median' + aic_or_bic + ' only)')

    fig.suptitle('Patients aged ' + cohort_label + ' and ' + admission_label, fontsize=16)
    plt.savefig('hmm_plots/combined_performance_' + cohort + '_' + admission_status + '_numstates_' + str(numstates))


if __name__ == "__main__":
    # plot for all cohorts
    for cohort in cohorts:
        for status in admission_status:
            plot_all_performances(cohort, status, numstates, 'BIC')