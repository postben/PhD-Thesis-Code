import pandas as pd
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from hmmlearn.hmm import MultinomialHMM
import os
import sys

os.chdir('p:/postb/work/')
path = 'p:/postb/work/'

# 1. Read in HMM train and test datasets (dictionary of IDs)

cohorts = ['young','middle','old']
admitted = ['y','n']
train_test = ['train','test']
randstate = 101 #sys.argv[1]

all_ids = {}
for cohort in cohorts:
    for adm in admitted:
        for tt in train_test:
            all_ids[f"{cohort}_{adm}_{tt}"] = pd.read_pickle(path+'hmm_work/hmm_1_0/hmm_pkl_dfs/'
                                                             +cohort+'_'+adm+'_'+tt+'_sample_5000_train_1000_randstate_'+str(randstate))

'''
Because of the complexity of downstream testing the ids above need to be further subdivided:
1. Need a training set for the HMM - this will stay as those labelled 'train' in the 'all_ids' dictionary 
(e.g 'young_y_train', middle_n_train'. Referred to as Group A
2. Need a 2nd set to test the HMM models and train the demo model. Referred to as Group B
3. Need a 3rd set to test the demo model AND the combined demo/hmm model. Referred to as Group C
4. Group B and Group C will be formed by dividing the 'test' datasets in 'all_ids' into 2 equally sized DFs (denoted by suffix _b and _c)

So:
HMM trained on A and tested on B
Demo trained on B and tested on C
HMM+Demo trained on B and tested on C

'''
test_keys = [key for key in all_ids.keys() if 'test' in key]
test_ids_split = {}
for i in test_keys:
    test_ids_split[f'{i}_b'] = all_ids[i].sample(frac = 0.5, random_state=101)
    test_ids_split[f'{i}_c'] = all_ids[i].drop(test_ids_split[f'{i}_b'].index)


# 2. Read in main HMM dataframe
hmm_study_df = pd.read_pickle(path+'hmm_work/hmm_1_0/hmm_pkl_dfs/'+'study_df_sample_5000_train_1000_randstate_'+str(randstate))

# 3. Make HMM for each subset

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

# get training keys
hmm_train_keys = [key for key in all_ids.keys() if 'train' in key]

# make dictionary of hmm_models
num_states = 10
hmm_models = {}
for key in hmm_train_keys:
    hmm_models[f'{key}'],_,_,_ = make_hmm(hmm_study_df, all_ids[key], num_states)

filename = 'hmm_models'
pickle.dump(hmm_models, open('combine_demo_hmm/combined_pkl_models/'+filename+'_states_'+str(num_states),'wb'))

# or

hmm_models = pd.read_pickle('combine_demo_hmm/combined_pkl_models/'+'hmm_models'+'_states_'+str(num_states))

# 4. Read in demo study table

demo_study_df = pd.read_pickle(path+'demo_work/demo_pkl_dfs/'+'big_cleaned_demo_pmh_table2018')

# 5. Get train-test IDs for demo (group B + group C)

demo_train_keys = [key for key in test_ids_split.keys() if '_b' in key]
demo_test_keys = [key for key in test_ids_split.keys() if '_c' in key]

# Group B
group_b = pd.DataFrame(columns = ['ALF_PE', 'corr_age', 'dead_or_admit'])
for key in demo_train_keys:
    group_b = group_b.append(test_ids_split[key])

# Group C
group_c = pd.DataFrame(columns = ['ALF_PE', 'corr_age', 'dead_or_admit'])
for key in demo_test_keys:
    group_c = group_c.append(test_ids_split[key])


# need to drop duplicates as the sampling process used to create these datasets is WITH replacement
group_b = group_b.drop_duplicates()
group_c = group_c.drop_duplicates()

## merge ids with demo study table
demo_train_df = demo_study_df[demo_study_df['ALF_PE'].isin(group_b['ALF_PE'])] # this is group B
demo_test_df = demo_study_df[demo_study_df['ALF_PE'].isin(group_c['ALF_PE'])] # this is group C

# 6. Create demo models (LR)

def create_lr_model(train_df):

    # drop unnecessary cols
    train_df = train_df.drop(columns = ['ALF_PE', 'BIRTH_YEAR', 'ETHNICITY_CODE', 'ETHNICITY',
                                        'ADDRESS_IMD_START_DATE', 'ADDRESS_IMD_END_DATE', 'LSOA2011_CD',
                                        'LSOA_DESC', 'WIMD_2014_SCORE', 'WIMD_2014_RANK',
                                        'TOWNSEND_2011_SCORE', 'DEATH_YEAR', 'ADM_2018','YR'])


    # for 2018 there's a problem because ther are no CHDL results - so drop
    train_df = train_df.drop(columns = ['CHDL', 'CHDL_indicator'])

    X_train = train_df.drop(columns = ['dead_or_admit'])
    X_train_cols = X_train.columns
    y_train = train_df['dead_or_admit']

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)

    # create model (with 5 fold cross validation)
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV

    lr = LogisticRegression(max_iter=1000)
    model = lr.fit(X_train,y_train)
    calibrator = CalibratedClassifierCV(lr, cv = 5)
    calibrated_model = calibrator.fit(X_train,y_train)

    regress_coefs = pd.concat([pd.DataFrame(X_train_cols),pd.DataFrame(np.transpose(model.coef_))], axis =1)
    regress_coefs.columns = ['feature','coef']
    regress_coefs = regress_coefs.sort_values(by = ['coef'])
    # filename = 'lr_model_sample_'+ str(sample_size) + '_train_' + str(train_size) + '_totalpts_' + str(len(train_df))
    # path = 'p:/postb/work/demo_work/demo_models/'
    # pickle.dump(calibrated_model, open(path+filename,'wb'))
    return model, calibrated_model, regress_coefs

lr_model, lr_model_calibrated, lr_regress_coefs = create_lr_model(demo_train_df)

# plot regress coefs
fig, ax = plt.subplots(1,1)
ax = sns.barplot(data=lr_regress_coefs, x ='feature', y='coef', ax=ax)
ax.set_xlabel('Feature')
ax.set_ylabel('Regression Coefficient')
ax.set_title('Demo only LR')
ax.tick_params(axis='x', rotation=75)
fig.tight_layout()

# 7. Testing

## a.HMM

def hmm_model_testing(admit_model, not_model, test_ids, study_df):
    '''
    :param admit_model: cohort-specific admission model
    :param not_model: cohort-specific non-admission model
    :param test_ids: ids associated with this cohort's test patients
    :param study_df: large hmm study_df
    :return: results dataframe
    '''

    from tqdm import tqdm

    # keep only data points associated with test patient ids
    test_df = study_df[study_df['ALF_PE'].isin(test_ids['ALF_PE'])]
    print('number of data points: ', len(test_df))

    # empty arrays for model scores
    ALF_PE =[]
    admit_score = []
    not_score = []

    # create input data for each patient that can be handled by model for testing
    for i in tqdm(test_ids['ALF_PE']):
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

    return results_df


# test each model by group then join into a single test df

def hmm_choose_group_and_test(group):
    '''
    :param group: this is group 'b' or 'c'
    :return: results of training
    '''
    # get test ids for each age group
    # YOUNG (from group B)
    young_keys = [key for key in test_ids_split.keys() if 'test' in key and 'young' in key and group in key]

    young_hmm_test = pd.DataFrame(columns=['ALF_PE', 'corr_age', 'dead_or_admit'])
    for i in young_keys:
        young_hmm_test = young_hmm_test.append(test_ids_split[i])

    # MIDDLE (from group B)
    middle_keys = [key for key in test_ids_split.keys() if 'test' in key and 'middle' in key and group in key]
    middle_hmm_test = pd.DataFrame(columns=['ALF_PE', 'corr_age', 'dead_or_admit'])
    for i in middle_keys:
        middle_hmm_test = middle_hmm_test.append(test_ids_split[i])

    # OLD (from group B)
    old_keys = [key for key in test_ids_split.keys() if 'test' in key and 'old' in key and group in key]
    old_hmm_test = pd.DataFrame(columns=['ALF_PE', 'corr_age', 'dead_or_admit'])
    for i in old_keys:
        old_hmm_test = old_hmm_test.append(test_ids_split[i])

    yes_models = [key for key in hmm_models.keys() if '_y_' in key]
    no_models = [key for key in hmm_models.keys() if '_n_' in key]
    test_dfs = [young_hmm_test, middle_hmm_test, old_hmm_test]

    # dataframe for results
    hmm_test_results = pd.DataFrame(columns=['ALF_PE', 'admit_score', 'not_score', 'pred_label', 'true_label',
                                             'correctly_classified'])

    for i, j, k in zip(yes_models,no_models,test_dfs):
        temp = hmm_model_testing(hmm_models[i],hmm_models[j],k,hmm_study_df)
        hmm_test_results = hmm_test_results.append(temp)

    return hmm_test_results

hmm_test_results = hmm_choose_group_and_test('b')

## b.Demo

def demo_model_testing(calibrated_model, test_df):

    # drop unnecessary cols
    test_df = test_df.drop(columns = ['ALF_PE', 'BIRTH_YEAR', 'ETHNICITY_CODE', 'ETHNICITY',
                                      'ADDRESS_IMD_START_DATE', 'ADDRESS_IMD_END_DATE', 'LSOA2011_CD',
                                      'LSOA_DESC', 'WIMD_2014_SCORE', 'WIMD_2014_RANK',
                                      'TOWNSEND_2011_SCORE', 'DEATH_YEAR', 'ADM_2018', 'YR'])

    # AS ABOVE DROP CHDL BECAUSE NO VALUES
    test_df = test_df.drop(columns = ['CHDL','CHDL_indicator'])

    X_test = test_df.drop(columns = ['dead_or_admit'])
    y_test = test_df['dead_or_admit']

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_test = scaler.fit_transform(X_test)

    # confusion matrix
    from sklearn.metrics import confusion_matrix
    preds = calibrated_model.predict(X_test)
    confusion_matrix = pd.crosstab(y_test,preds, rownames=['True Label'], colnames=['Predicted label'])

    TN = confusion_matrix.iloc[0,0]
    FP = confusion_matrix.iloc[0,1]
    FN = confusion_matrix.iloc[1,0]
    TP = confusion_matrix.iloc[1,1]

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

    fig, ax = plt.subplots(3, 1, figsize=(10,6))
    sns.heatmap(confusion_matrix, annot=True, annot_kws={'size':20}, fmt='g', ax = ax[0], cmap = 'viridis') # fmt='g' means this doesn't print in exponential form
    ax[0].set_title('Confusion Matrix',fontsize=16)

    # calibration curve
    probs = calibrated_model.predict_proba(X_test)
    from sklearn.calibration import calibration_curve
    fop, mpv = calibration_curve(y_test.to_numpy(), probs[:,1], n_bins=10) # probs[:,1] gives the probability of class '1' (positive class)

    # fig,ax = plt.subplots(1,1, figsize = (15,8))
    ax[1].plot([0,1],[0,1],linestyle='--', label='perfect_calibration')
    ax[1].plot(mpv, fop, marker='.', label = 'LR model')
    ax[1].set_xlabel('Predicted probability of admission/death', fontsize=14)
    ax[1].set_ylabel('Fraction of actual admissions/deaths', fontsize = 14)
    ax[1].set_title('Calibration Curve for LR Model', fontsize=16)
    ax[1].legend()

    # metrics
    the_table = ax[2].table(cellText = metrics.values, colLabels= metrics.columns, loc ='center')
    # the_table.auto_set_font_size(False)
    # the_table.set_fontsize(14)
    the_table.scale(1,4)
    ax[2].set_title('Performance Metrics')
    ax[2].axis('off')
    ax[2].axis('tight')

    fig.tight_layout()
    plt.savefig('plots/log_reg_performance')

    return metrics

demo_model_testing(lr_model_calibrated, demo_test_df)

## c.Combined

# add hmm_results as feature to demo test_table (group B)
hmm_results_group_b = hmm_test_results[['ALF_PE','pred_label']]
hmm_results_group_b.columns = ['ALF_PE','hmm_pred_label']
demo_hmm_train_df = demo_train_df.merge(hmm_results_group_b,how='left',on='ALF_PE')

# train new LR with predicted values from hmm model
lr_model_with_hmm, lr_model_with_hmm_calibrated, lr_model_with_hmm_regress_coefs = create_lr_model(demo_hmm_train_df)

# plot regress coefs for combined model
fig, ax = plt.subplots(1,1)
ax = sns.barplot(data=lr_model_with_hmm_regress_coefs, x ='feature', y='coef', ax=ax)
ax.set_xlabel('Feature')
ax.set_ylabel('Regression Coefficient')
ax.set_title('Demo with HMM classifier - LR')
ax.tick_params(axis='x', rotation=75)
fig.tight_layout()

# get hmm predictions for group C
hmm_test_results_c = hmm_choose_group_and_test('c')
hmm_results_group_c = hmm_test_results_c[['ALF_PE','pred_label']]
hmm_results_group_c.columns = ['ALF_PE','hmm_pred_label']

# join to group c
demo_hmm_test_df = demo_test_df.merge(hmm_results_group_c,how='left',on='ALF_PE')

# for group C:
# show performance of each model

def hmm_plot_metrics(results_df):
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
    sns.heatmap(confusion_matrix, annot=True, annot_kws={'size':20}, fmt='g', ax = ax[0], cmap = 'viridis') # fmt='g' means this doesn't print in exponential form
    ax[0].set_title('Confusion Matrix')

    the_table = ax[1].table(cellText = metrics.values, colLabels= metrics.columns, loc ='center')
    # the_table.auto_set_font_size(False)
    # the_table.set_fontsize(10)
    the_table.scale(1,4)
    # ax[1].set_title('Performance Metrics')
    ax[1].axis('off')
    ax[1].axis('tight')
    fig.suptitle('HMM Classification Metrics', fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0,hspace=0)
    plt.show()
    # plt.savefig('hmm_plots/classification_performance_'+ cohort + '_sample_' + str(sample_size) + '_' + str(train_size) + '_numstates_' + str(num_states) + \
    #             '_randstate_' + str(rand_state) + '_num_test_pts_' + str(num_test_pts))

    return metrics


# show hmm performance
hmm_metrics = hmm_plot_metrics(hmm_test_results_c)
# show demo performance
demo_metrics = demo_model_testing(lr_model, demo_hmm_test_df.drop(columns=['hmm_pred_label']))
# show combined performance
combined_metrics = demo_model_testing(lr_model_with_hmm_calibrated, demo_hmm_test_df)

# compare metrics across models

compare_metrics = hmm_metrics.append(demo_metrics)
compare_metrics = compare_metrics.append(combined_metrics)
compare_metrics.index = ['hmm','lr','combined']

# save comparison metrics to be combined with other randstates to get CIs etc
filename = 'combined_metrics_df_randstate_'+str(randstate)
pickle.dump(compare_metrics, open('p:/postb/work/combine_demo_hmm/combined_pkl_dfs/'+filename,'wb'))

fig, ax = plt.subplots(1,1, figsize=(10,6))
ax = compare_metrics.T.plot(kind='bar', ax=ax)
ax.set_xlabel('Metric')
ax.set_ylabel('Performance')
ax.set_title('Comparing Performance of HMM and LR Models')
ax.tick_params(axis='x', rotation=45)
ax.set_ylim(0.5,1)
fig.tight_layout()
filename='comparing_model_metrics'
plt.savefig('p:/postb/work/combine_demo_hmm/combined_plots/'+filename)


'''
Below to be done when the above code has been run 5 times
'''
all_perf_results = pd.DataFrame(columns=['Accuracy', 'Precision', 'Recall/Sensitivity', 'Specificity', 'F1'])
randstates = [101,102,103,104,105]
for randstate in randstates:
    filename = 'combined_metrics_df_randstate_'+str(randstate)
    all_perf_results = all_perf_results.append(pd.read_pickle('p:/postb/work/combine_demo_hmm/combined_pkl_dfs/'+filename))

all_perf_results = all_perf_results.rename_axis('model_type').reset_index()

lr_stuff = all_perf_results[all_perf_results['model_type']=='lr'].T
lr_stuff = lr_stuff.drop(['model_type'])

import statsmodels.stats.api as sms
def lower_ci(x):
    lower_ci, _ = sms.DescrStatsW(x).tconfint_mean()
    return lower_ci
def upper_ci(x):
    _, upper_ci = sms.DescrStatsW(x).tconfint_mean()
    return upper_ci

# get summary stats
lr_stuff['mean'] = lr_stuff.iloc[:,0:5].apply(np.mean, axis=1)
lr_stuff['lower_ci'] = lr_stuff.iloc[:,0:5].apply(lambda x: lower_ci(x), axis=1)
lr_stuff['upper_ci'] = lr_stuff.iloc[:,0:5].apply(lambda x: upper_ci(x), axis=1)

# plot
fig, ax = plt.subplots(1,1,figsize = (10,10))
ax.plot(df['num_states'], df['mean'])
ax1.fill_between(df['num_states'],df['lower_ci'], df['upper_ci'],color='red',alpha=0.2)
ax1.set_xticks(np.arange(min(df['num_states']), max(df['num_states']+1),1))
ax1.set_xlabel('Number of States')
ax1.set_ylabel(str(aic_or_bic))


# using seaborn
perf_results_melt = pd.melt(all_perf_results, id_vars=['model_type'])

fig, ax = plt.subplots(1,1,figsize=(15,9))
to_keep = ['Accuracy','Specificity','F1']
data = perf_results_melt.query('variable in @to_keep')
ax = sns.barplot(x='variable', hue='model_type',y='value', data = data, ax = ax)
ax.set_ylim(0.5,1)
ax.set_xlabel('Performance Metric',fontsize=24)
ax.set_ylabel('Score + 95% CI',fontsize=24)
ax.set_title('HMM vs Logistic Regression as a Classifier (5 Runs)',fontsize=28)
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
ax.legend(fontsize=20)
fig.tight_layout()
fig.savefig('combine_demo_hmm/combined_plots/hmm_vs_lr_vs_combined')
plt.show()


import matplotlib
matplotlib.use('TKAgg')