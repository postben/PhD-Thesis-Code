''' ,.****
2021_10_17

Sys.args:
[1] name of cohort for testing
[2] sample size (from 2nd script)
[3] train size (from 2nd script)
[4] random state of train_test split df
[5] number of states of model to import
[6] test sample size (num of patients to test from test cohort)

1. Import all hmm models (created by 3_create_hmm_models.py)
2. Import all necessary data and patient ids for testing (created by 2_clean_and_split.py)
3. Test models by finding probability under models (admission vs non-admission models)
4. Plot these results (prob vs prob as per Aldo) and performance metrics
'''

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import sys
import os
from tqdm import tqdm

# set working directory
os.chdir('P:/postb/work/hmm_work/hmm_1_0')

# choose cohort for testing
cohort = sys.argv[1]
sample_size = sys.argv[2]
train_size = sys.argv[3]
rand_state = sys.argv[4]
num_states = sys.argv[5]
num_test_pts = sys.argv[6]

# cohort = 'young'
# sample_size = 5000
# train_size = 2500
# rand_state = 321
# num_states = 10
# num_test_pts = 2500

def import_models_and_data(cohort, sample_size, train_size):
    '''
    :param cohort: chosen chort ('young', 'middle', 'old') Note: this is entered in the terminal as argv[1]
    :return: admission model, non-admission model, ids corresponding to all test patients, the large study df
    '''
    ## import models
    # admission model
    filename = 'hmm_pkl_models/' + str(cohort) + '_y_' + str(sample_size) + '_' + str(train_size)  + '_randstate_' + str(rand_state) + '_numstates_' + str(num_states)
    admit_model = pickle.load(open(filename,'rb'))
    # not admission model
    filename = 'hmm_pkl_models/' + str(cohort) + '_n_' + str(sample_size) + '_' + str(train_size)  + '_randstate_' + str(rand_state) + '_numstates_' + str(num_states)
    not_model = pickle.load(open(filename,'rb'))

    # import test data pt ids (note this needs to be a combination of all the test data for the admitted and not-admitted patients)

    x = pd.read_pickle('p:/postb/work/hmm_work/hmm_1_0/hmm_pkl_dfs/' + str(cohort) + '_y_test_sample_' + str(sample_size) + '_train_' + str(train_size) \
                       + '_randstate_' + str(rand_state))
    y = pd.read_pickle('p:/postb/work/hmm_work/hmm_1_0/hmm_pkl_dfs/' + str(cohort) + '_n_test_sample_' + str(sample_size) + '_train_' + str(train_size) \
                       + '_randstate_' + str(rand_state))
    test_ids = pd.concat([x,y])

    # import large study_df
    study_df = pd.read_pickle('p:/postb/work/hmm_work/hmm_1_0/hmm_pkl_dfs/study_df_sample_'+ str(sample_size) + '_train_' + str(train_size) \
                              + '_randstate_' + str(rand_state))

    return admit_model, not_model, test_ids, study_df


def test_models(admit_model, not_model, test_ids, study_df, num_test_pts, random_state=101):
    '''
    input from above fuction
    :param admit_model: cohort-specific admission model
    :param not_model: cohort-specific non-admission model
    :param test_ids: ids associated with this cohort's test patients
    :param study_df: large study_df
    :return: results dataframe
    '''

    from tqdm import tqdm

    test_ids_sample = test_ids.sample(n=int(num_test_pts), random_state=random_state)
    print('test sample size: ', len(test_ids_sample))

    # keep only data points associated with test patient ids
    test_df = study_df[study_df['ALF_PE'].isin(test_ids_sample['ALF_PE'])]
    print('number of data points: ', len(test_df))

    # empty arrays for model scores
    ALF_PE =[]
    admit_score = []
    not_score = []

    # create input data for each patient that can be handled by model for testing
    for i in tqdm(test_ids_sample['ALF_PE']):
        input_array = test_df[test_df['ALF_PE']==i]['attendance'].to_numpy().reshape(-1,1)
        input_array = input_array.astype(int)
        admit_score_ = admit_model.score(input_array)
        not_score_ = not_model.score(input_array)
        ALF_PE.append(i)
        admit_score.append(admit_score_)
        not_score.append(not_score_)

    results_df = pd.DataFrame([ALF_PE,admit_score,not_score])
    results_df = results_df.transpose()
    results_df.columns = ['ALF_PE','admit_score','not_score']

    # classify model based on highest prob under the model between admission vs non-admission model
    def make_label(df):
        if df['admit_score'] > df['not_score']:
            return 1
        else:
            return 0

    results_df['pred_label'] = results_df.apply(make_label, axis=1)
    results_df = results_df.merge(test_ids[['ALF_PE','dead_or_admit']], how='left', on='ALF_PE')
    results_df = results_df.rename(columns={'dead_or_admit':'true_label'})
    results_df['correctly_classified'] = (results_df['true_label'] == results_df['pred_label']).astype(int)
    # save above as pickled df (takes a long time to read-in)
    filename= 'test_results_'+ cohort + '_sample_' + str(sample_size) + '_' + str(train_size) + '_numstates_' + str(num_states) + '_randstate_' + str(rand_state) + \
            '_num_test_pts_' + str(num_test_pts)
    pickle.dump(results_df, open('hmm_pkl_dfs/'+filename,'wb'))

    return results_df


def plot_probs(results_df):
    # plot probs
    fig, ax = plt.subplots(1,1, figsize=(15,9))
    df = results_df.assign(true_label = results_df.true_label.replace([0,1],['Not Admitted','Admitted']))
    ax = sns.scatterplot(data = df, x='admit_score', y='not_score', hue = 'true_label', palette='magma', s=10)
    ax.set_aspect('equal')
    ax.axline([0,0],[1,1],color='silver',linewidth='0.25', linestyle='-')
    lim_min = np.amin([results_df['admit_score'].min(), results_df['not_score'].min()])
    lim_max = np.amax([results_df['admit_score'].max(), results_df['not_score'].max()])
    ax.set_xlim(lim_min*0.999,lim_max*1.001)
    ax.set_ylim(lim_min*0.999,lim_max*1.001)
    if cohort == 'young':
        figtitle = 'Age <40'
    elif cohort == 'middle':
        figtitle = 'Age 40-80'
    elif cohort == 'old':
        figtitle = 'Age >80'
    ax.set_title('Comparing HMM Model Performance (admission vs non-admission) for: \n' + ' ' + figtitle + ' (Test n = ' + str(num_test_pts) + ')',fontsize=26)
    ax.set_xlabel('LL under Admission Model',fontsize=22)
    ax.xaxis.set_tick_params(labelsize=18)
    ax.set_ylabel('LL under Non-admission Model',fontsize=22)
    ax.yaxis.set_tick_params(labelsize=18)
    ax.legend(title=None)
    fig.tight_layout()
    plt.savefig('hmm_plots/compare_model_prob_' + cohort + '_sample_' + str(sample_size) + '_' + str(train_size) + '_numstates_' + str(num_states) + \
                '_randstate_' + str(rand_state) + '_num_test_pts_' + str(num_test_pts))
    plt.show()


def plot_classification_metrics(results_df):
    data = results_df[['true_label','pred_label']]
    confusion_matrix = pd.crosstab(data['true_label'],data['pred_label'], rownames=['True Label'], colnames=['Predicted label'])

    TN = confusion_matrix.iloc[0,0]
    FP = confusion_matrix.iloc[0,1]
    FN = confusion_matrix.iloc[1,0]
    TP = confusion_matrix.iloc[1,1]
    POPULATION = confusion_matrix.values.sum()

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    specificity = TN / (TN + FP)
    f1 = (2*TP) / (2*TP + FP + FN)

    metrics = pd.DataFrame({'Accuracy': accuracy,
                            'Precision': precision,
                            'Recall/Sensitivity': recall,
                            'Specificity': specificity,
                            'F1': f1}, index = [0])
    metrics = metrics.round(3)

    fig, ax = plt.subplots(2, 1, figsize=(10,6))
    sns.heatmap(confusion_matrix, annot=True, fmt='g', ax = ax[0], cmap = 'viridis') # fmt='g' means this doesn't print in exponential form
    ax[0].set_title('Confusion Matrix')

    the_table = ax[1].table(cellText = metrics.values, colLabels= metrics.columns, loc ='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    ax[1].axis('off')
    ax[1].axis('tight')
    if cohort == 'young':
        figtitle = 'Age <40'
    elif cohort == 'middle':
        figtitle = 'Age 40-80'
    elif cohort == 'old':
        figtitle = 'Age >80'
    fig.suptitle('Classification Metrics: ' + figtitle + ' (Test n = ' + str(num_test_pts) + ')', fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0,hspace=0)
    plt.savefig('hmm_plots/classification_performance_'+ cohort + '_sample_' + str(sample_size) + '_' + str(train_size) + '_numstates_' + str(num_states) + \
                '_randstate_' + str(rand_state) + '_num_test_pts_' + str(num_test_pts))
    plt.show()


if __name__ == "__main__":
    # import data and models
    # use below for testing/debugging
    cohort = 'middle'
    sample_size = 5000
    train_size = 2500
    rand_state = 321
    num_states = 10
    num_test_pts = 2000

    admit_model, not_model, test_ids, study_df = import_models_and_data(cohort, sample_size, train_size)
    print('Number of total test patients = ', len(test_ids))
    # test imported models against test data
    results_df = test_models(admit_model=admit_model, not_model=not_model, test_ids=test_ids, study_df=study_df, num_test_pts = num_test_pts, random_state=101)
    # plot prob vs prob AND classification metrics
    plot_probs(results_df)
    plot_classification_metrics(results_df)