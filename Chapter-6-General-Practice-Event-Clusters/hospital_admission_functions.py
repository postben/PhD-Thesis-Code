'''
Code - Chapter 6 - General Practice Event Clusters
'''

# Import Required Packages

import SAIL_python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
from matplotlib import gridspec, cm
import random
# matplotlib.use('TKagg')
import seaborn as sns
from tqdm.auto import tqdm
tqdm.pandas()
import pickle
# import tensorflow as tf
# from tensorflow import keras
# import keras_tuner as kt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, ConfusionMatrixDisplay
import warnings
import datetime
import string
import scipy
from itertools import combinations
from collections import OrderedDict

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import KFold
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from sklearn.model_selection import train_test_split,cross_val_predict
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,f1_score,recall_score,roc_auc_score
from sklearn.metrics import log_loss,brier_score_loss,average_precision_score
from sklearn.metrics import make_scorer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

from sklearn.calibration import calibration_curve


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


'''
Data cleaning for clustering work 
'''

def make_cluster_df(joined_df, num_iets):
    '''
    This takes the raw data from SAIL and transforms these data into sequential IETs
    :param joined_df: dataframe of raw extract from SAIL SQL database - which has been joined by patient identifier (ALF_PE)
    :param num_iets: the number of interevent times to include
    :return: dataframe with patient ID, index date and outcome indicator plus {num_iet} columns showing sequential intervisit times
    '''
    cluster_df = (
        joined_df
        # remove any patients that don't have enough visits to facilitate clustering
        .query(f'total_gp_visits>{num_iets}')
        .sort_values(['pseudo_id','order'], ascending=False)
        .reset_index(drop=True)
        # the below must happen after the above code to ensure the correct rows are returned
        .iloc[lambda df: df
            .groupby(['pseudo_id'])
            ['order']
            .head(num_iets)
            .index
            ,:] # this comma and colon is critical to slice correctly!
        # re-sort values so that events ordered chronologically
        .sort_values(['pseudo_id','order'], ascending=True)
        # update 'order' so each patient has 1->5 representing their final 5 visits
        .pipe(lambda x: x.assign(updated_order = x.groupby(['pseudo_id']).cumcount()+1))
        # shape date for clustering
        .pivot(index=['pseudo_id','outcome_or_pseudo_date','outcome_indicator'], columns='updated_order', values='date_diff')
        .add_prefix('iet_')
        .reset_index()
    )

    return cluster_df


def plot_real_patient_patterns(joined_df, num_iets, num_pts):
    '''
    Visually demonstrates what has been captured with the cluster dataframe and samples n patients to visualise (and adds noise to hide 'real' data)
    :param joined_df: dataframe of raw extract from SAIL SQL database - which has been joined by patient identifier (ALF_PE)
    :param num_iets: the number of interevent times to include
    :param num_pts: number of patients to randomly visualise for each outcome i.e. n*2 patients will be plotted
    :return:
    '''
    alphabet_list = list(string.ascii_uppercase[:26])
    num_list = list(range(0,26))

    cluster_df = make_cluster_df(joined_df, num_iets)

    # create dataframe for plotting that will allow sequential visualisation of events
    plot_df = (
        cluster_df
            .groupby('outcome_indicator')
            .sample(num_pts)
            .drop(columns=['pseudo_id','outcome_or_pseudo_date'])
            .assign(zero_iet = 0)
            .rename(columns={'zero_iet':f'iet_{num_iets+1}'})
            .reset_index(drop=True)
            .melt(id_vars=['outcome_indicator'], ignore_index=False)
            .rename(columns={'updated_order':'iet'})
            .pipe(lambda x: x.assign(iet = x.iet.str.replace('iet_','')))
            # align iet order from time of index date
            .pipe(lambda x: x.assign(iet = (pd.to_numeric(x.iet) - num_iets)-1))
            .assign(value = lambda x: (x.value)*-1)
            .reset_index()
            .sort_values(by=['index','iet'],ascending=False)
            .pipe(lambda x: x.assign(cumulative= x.groupby('index')['value'].cumsum()))
            # add noise to hide patient-specific values
            .assign(cumulative_with_noise =  lambda x: x.cumulative.apply(lambda y: y+random.randint(-10,1) if y != 0 else y))
            .rename(columns={'index':'plot_id'})
            .pipe(lambda x: x.assign(plot_id_letter = pd.Series(x.plot_id.values).replace(num_list, alphabet_list)))
    )

    # create plot
    fig, ax = plt.subplots(1,1,figsize=(30/2.54,7.5/2.54))
    for plot_id,df in plot_df.groupby('plot_id'):
        color = 'orange' if df.outcome_indicator.max() ==1 else 'grey'
        label = 'Patient Admitted' if df.outcome_indicator.max() ==1 else 'Patient NOT Admitted'
        ax.plot(df.cumulative_with_noise, df.plot_id, marker='|', markeredgecolor='blue',  markerfacecolor='blue', markerfacecoloralt='blue',
                linewidth=2.5, ms=15, color=color, label=label)

    handles,labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels,handles))
    by_label.move_to_end('Patient NOT Admitted')
    ax.legend(by_label.values(), by_label.keys(),fontsize=14, numpoints=2)
    leg = ax.get_legend()

    ax.tick_params(axis='both',labelsize=16)
    ax.set_xlabel('Timeline Before Final GP Event [days]',fontsize=16)
    ax.set_ylabel('Patient',fontsize=16)
    ax.set_xlim(right=40)

    # remove '-' from labels to avoid confusion
    labels = [int(abs(item)) for item in ax.get_xticks()]
    ax.set_xticklabels(labels)

    #add padding for better plotting
    y_min,y_max = ax.get_ylim()
    ax.set_ylim(y_min-.5,y_max+.5)

    ax.set_yticks(range(plot_df.plot_id_letter.nunique()))
    ax.set_yticklabels(pd.Series(plot_df.plot_id_letter.unique()).sort_values(ascending=False))

    y_min,y_max = ax.get_ylim()
    ax.text(x=20, y = (y_min+y_max)/2, s='Final\nGP\nEvent\nBefore\nIndex\nDate', rotation=0, fontsize=16,
            ha='center',va='center',color='red')

    # ax.set_title(f'Real Patient GPE Patterns: {num_iets+1} GPEs before Index Date', fontsize=35)
    fig.tight_layout()



'''GMM Work'''

def gmm_elbow_plot(joined_df, num_iets, max_num_components, sub_sample=None):
    '''
    Plots an elbow plot for a GMM from 2 -> n clusters (max_num_components)
    :param joined_df: dataframe of raw extract from SAIL SQL database - which has been joined by patient identifier (ALF_PE)
    :param num_iets: the number of interevent times to include
    :param max_num_components: max number of clusters/components to calculate scores for
    :param sub_sample: subsample of patients can be selected to make this process faster and for debugging
    :return: elbow plot
    '''

    # function from above to make cluster df from main study table
    cluster_df = make_cluster_df(joined_df, num_iets)

    if sub_sample==None:
        cluster_df = cluster_df
    else:
        # cluster and retain the correct proportion of labels
        # note train_test_split is only being used to subsample whilst retaining the same label proportions
        sub_sampled_data,_ = train_test_split(cluster_df, train_size=sub_sample,stratify=cluster_df.outcome_indicator)
        cluster_df = sub_sampled_data

    # Clean and scale
    X = cluster_df.drop(columns=['pseudo_id', 'outcome_or_pseudo_date', 'outcome_indicator'])
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # k-num components
    K = range(2,max_num_components+1)
    # K-folds
    kfold =  KFold(n_splits=5, shuffle=True)
    bic_score = []
    sil_score = []
    Ks = []

    # 5-fold X-Validation
    for train,_ in kfold.split(X): # note that this gives the indexes!
        for k in K:
            gmm = GaussianMixture(n_components=k, max_iter= 1000, random_state=101)
            gmm.fit(X[train])
            bic_score.append(gmm.bic(X[train]))
            preds = gmm.predict(X[train])
            preds = preds.flatten()
            km_sil = silhouette_score(X[train], preds)
            sil_score.append(km_sil)
            Ks.append(k)

    data = {'bic_score':bic_score, 'sil_score':sil_score,'num_clusters': Ks}
    gmm_vals = pd.DataFrame.from_dict(data)

    # plot bic
    fig, ax = plt.subplots(1,1,figsize=(15,9))
    ax = sns.lineplot(data=gmm_vals, x='num_clusters', y='bic_score', color='blue', label='BIC')
    ax.set_ylabel('Bayesian Information Criteria (BIC)\n(+/-95% CI)',color = 'blue', fontsize=22)

    # plot silhouette score
    ax2 = ax.twinx()
    ax2 = sns.lineplot(data=gmm_vals, x='num_clusters', y='sil_score', color='orange', label='Silhouette')
    ax2.set_ylabel('Silhouette Score (+/-95% CI)',color = 'orange', fontsize=22)
    ax2.tick_params(axis = 'both', labelsize=18)

    # plot guide lines every 5 clusters
    ax2.vlines(2,ymin=0,ymax=max(sil_score), linestyle='--')
    ax2.text(2.5,max(sil_score),'2 clusters', fontsize=16)
    for xval in range(5,max_num_components,5):
        ax2.vlines(xval,ymin=0,ymax=max(sil_score), linestyle='--')
        ax2.text(xval+0.5,max(sil_score),f'{xval} clusters', fontsize=16)

    ax.set_xlabel('Number of Clusters', fontsize=22)
    ax.set_xlim(1,max_num_components)
    ax.set_ylim(min(bic_score), max(bic_score))
    ax.tick_params(axis = 'both', labelsize=18)

    # legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=16, loc='lower left')
    ax2.get_legend().remove()

    fig.suptitle(
        f'GMM Elbow Plot with 5 Fold X-Validation - {num_iets} GP Interevent Times \n({sub_sample} Samples)'
        ,fontsize=26)

    fig.tight_layout()
    plt.show()



def gmm_cluster_and_label(joined_df, num_iets, n_components, rand_state=None):

    '''
    Trains a GMM using the joined df based on the chosen number of IETs. Labels are assigned to each patient and centroids returned.
    :param joined_df: dataframe of raw extract from SAIL SQL database - which has been joined by patient identifier (ALF_PE)
    :param num_iets: the number of interevent times to include
    :param max_num_components: max number of clusters/components to calculate scores for
    :param rand_state: select random state
    :return: 2 dataframes: a. is all the patient ids with a cluster label; b. the individual clusters and their centroids
    '''
    # function from above to make cluster df from main study table
    cluster_df = make_cluster_df(joined_df, num_iets)

    # Clean and scale
    X = cluster_df.drop(columns=['pseudo_id', 'outcome_or_pseudo_date', 'outcome_indicator'])
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # GMM
    gmm = GaussianMixture(n_components=n_components
                          ,covariance_type='full'
                          ,init_params='kmeans'
                          ,random_state=rand_state
                          )
    gmm_labels = gmm.fit_predict(X)

    # rejoin with clustering training DF so that labels are associated with an id
    clusters_with_ids = (cluster_df
                         .assign(cluster_label = gmm_labels)
                         .assign(cluster_label = lambda x: x.cluster_label.astype('int8'))
                         .filter(items = ['pseudo_id','cluster_label'])
                         .merge(joined_df
                                .filter(items=['pseudo_id','outcome_indicator']).drop_duplicates(),
                                how='left',on='pseudo_id')
                         )

    # get cluster centers
    raw_c_centers = gmm.means_
    # unscale cluster centers
    c_centers = scaler.inverse_transform(raw_c_centers)
    # get distinct cluster labels~
    unique_cluster_labs = range(0,n_components)
    # create dataframe of cluster centres with associated labels
    cluster_labs_centers = pd.DataFrame(data=c_centers, index=unique_cluster_labs)
    cluster_labs_centers.index.name = 'cluster_label'

    return clusters_with_ids, cluster_labs_centers



def multiple_gmm_trials(joined_df, num_iets_min, num_iets_max, n_components, num_trials=5):
    '''
    ## Beware that this will take a while to run ##
    Run multiple trials for GMMs - this allows downstream error bars, etc
    :param joined_df: dataframe of raw extract from SAIL SQL database - which has been joined by patient identifier (ALF_PE)
    :param num_iets_min: minimum number of interevent times to explore
    :param num_iets_max: max number of interevent times to explore
    :param n_components: number of clusters/components for each GMM
    :param num_trials: number of trials to run for each number of IETs
    :return: dictionary of results for each clustering trial
    '''

    gmm_multiple_visit_dict = {}
    rand_states = np.random.randint(1,1000,num_trials)

    for i in tqdm(range(num_iets_min,num_iets_max+1,1)):
        for rand_state in rand_states:
            gmm_clusters_with_ids, gmm_cluster_labs_centers = gmm_cluster_and_label(
                joined_df, num_iets=i, n_components=n_components, rand_state=rand_state)

            gmm_multiple_visit_dict[f'clusters_{i}_rand_{rand_state}'] = gmm_clusters_with_ids
            gmm_multiple_visit_dict[f'centers_{i}_rand_{rand_state}'] = gmm_cluster_labs_centers

    return gmm_multiple_visit_dict


def plot_high_risk_admission_cluster_multiple_trials(gmm_multiple_visit_dict,joined_df):
    '''
    :param gmm_multiple_visit_dict: the dictionary for multiple GMM clustering trials from above function
    :return: plot of high-risk cluster admission rate for multiple IETs
    '''
    visits = []
    max_adm = []
    pts_missed = []

    # this retrieves the range of number of iets explored in gmm_multiple_dict
    range_iets = (pd.Series(gmm_multiple_visit_dict.keys())
                  # get first element which will include the max and min number of iets explored
                  .str.replace('clusters_','')
                  .str.replace('centers_','')
                  # split before first underscore
                  .apply(lambda x: x.split('_')[0])
                  #  make these numeric
                  .apply(lambda x: pd.to_numeric(x,errors='coerce'))
                  )

    # get min and max
    min_num_iets = range_iets.min()
    max_num_iets = range_iets.max()

    for i in range(min_num_iets,max_num_iets+1,1):
        num_visits = i
        cluster_key = f'clusters_{i}'
        needed_keys = [key for key in gmm_multiple_visit_dict.keys() if cluster_key in key]

        for key in needed_keys:

            max_adm_rate = (
                gmm_multiple_visit_dict[key]
                .groupby('cluster_label')
                ['outcome_indicator']
                .value_counts(normalize=True)
                .to_frame()
                .rename(columns={'outcome_indicator':'proportion'})
                .assign(proportion = lambda x: x.proportion.round(3))
                .reset_index()
                .sort_values(by=['cluster_label','outcome_indicator'])
                .assign(outcome_indicator = lambda x: x.outcome_indicator.replace({0:'no event',1:'event'}))
                .query('outcome_indicator=="event"')
                .reset_index(drop=True)
                .pipe(lambda x: x.iloc[x['proportion'].idxmax()])
                ['proportion']
            )

            frac_pts_not_clustered = 1 - ((gmm_multiple_visit_dict[key].pseudo_id.nunique()) / (joined_df['pseudo_id'].nunique()))

            visits.append(num_visits)
            max_adm.append(max_adm_rate)
            pts_missed.append(frac_pts_not_clustered)


    fig, ax = plt.subplots(1,1,figsize=(15,10))
    ax = sns.lineplot(x=visits, y = max_adm, ax=ax,color='orange', label='High Risk Cluster Admission Rate')
    ax.set_ylim(0,1)
    # identify the important change point
    ax.vlines(x=18,ymin=ax.get_ylim()[0],ymax=ax.get_ylim()[1], linestyles='dashed',color='dimgrey')
    ax.text(x=20.5,y=ax.get_ylim()[1]*0.95,s='GPE-Sequence = 18', ha='center',c='dimgrey',fontsize=16)
    # show baseline admission rate based on number of GPEs
    # ax = sns.lineplot(data = adm_rate_by_num_visits_5_samples, x='total_gp_visits', y = 'admission_rate',
    #                   ax=ax, color='blue', linestyle='dotted', label ='Baseline Admission Rate (by number of GPEs)')
    pts_included = [1-i for i in pts_missed]
    ax = sns.lineplot(x=visits, y = pts_included,color='cornflowerblue', label='Fraction of Patients Included',ls='--')

    # labels
    ax.set_xlabel('Number of GPEs',fontsize=25)
    ax.set_ylabel('Admission Rate',fontsize=25)
    ax.tick_params(axis='both',labelsize=20)
    ax.yaxis.label.set_color('orange')
    ax.set_title('High Risk Cluster Admission Rate for Varying GPE-Sequences',fontsize=35)
    ax.legend(fontsize=16,loc='upper left')
    fig.tight_layout()

#################################
'''Validation Experiments'''
#################################

# STEP 1 - Repeated clustering runs for IET stability

def multiple_clustering_runs_stability_check(joined_df, num_iets, n_components, num_runs):
    '''
    :param joined_df: dataframe of raw extract from SAIL SQL database - which has been joined by patient identifier (ALF_PE)
    :param num_iets: the number of interevent times to include
    :param n_components: number of clusters/components for each clustering trial
    :param num_runs: number of clustering trials to perform
    :return: dataframe with centroids for each clustering run
    '''

    # data frame to collect max adm rate cluster centroids
    all_ordered_cluster_centroids = pd.DataFrame()

    for run in tqdm(range(1,num_runs+1)):

        gmm_clusters_with_ids, gmm_cluster_labs_centers = gmm_cluster_and_label(
            joined_df, num_iets=num_iets, n_components=n_components, rand_state=run) # set rand_state as the run number

        sorted_cluster_adm_rates = (
            gmm_clusters_with_ids
                .groupby('cluster_label')
            ['outcome_indicator']
                .value_counts(normalize=True)
                .reset_index(level=0)
                .rename(columns={'outcome_indicator':'adm_rate'})
                .reset_index()
                .query('outcome_indicator==1')
                .drop(columns=['outcome_indicator'])
                .reset_index(drop=True)
                .sort_values(by=['adm_rate'], ascending=False)
                .reset_index(drop=True)
        )

        centroid_list=pd.DataFrame()

        for i in range(len(sorted_cluster_adm_rates)):
            cluster = sorted_cluster_adm_rates.cluster_label.iloc[i]
            adm_rate = sorted_cluster_adm_rates.adm_rate.iloc[i]

            # pull centroids from this cluster
            centroids = (
                gmm_cluster_labs_centers
                    .query('cluster_label==@cluster')
                    .assign(adm_rate = adm_rate,
                            adm_order = i
                            )
            )
            centroid_list = pd.concat([centroid_list,centroids])

        # drop cluster label at this point as the label will be random dependent on initialisations
        # what's more important is the pattern of events and the associated sequence of adm risk per run
        centroid_list = centroid_list.reset_index(drop=True)

        # add to dataframe

        all_ordered_cluster_centroids =  pd.concat([all_ordered_cluster_centroids,centroid_list])

    all_ordered_cluster_centroids = all_ordered_cluster_centroids.reset_index(drop=True)

    return all_ordered_cluster_centroids


## plot the above multiple clustering runs to check stability

def plot_multiple_clustering_runs(all_ordered_cluster_centroids):
    '''
    :param all_ordered_cluster_centroids: plot for multiple trials - e.g. 1000, to check stability of clusters
    :return: plot of ARCGs with boxplots for each intervisit time
    '''
    num_clusters = all_ordered_cluster_centroids.adm_order.nunique()
    num_cols = 2
    num_rows = int(np.round(num_clusters))
    num_runs = all_ordered_cluster_centroids.adm_order.value_counts().iloc[0]

    gs = gridspec.GridSpec(num_rows, num_cols, width_ratios=[0.9,0.1])

    # get adm rates with CIs and Stds
    adm_rates_with_confints = (
        pd.DataFrame(
            all_ordered_cluster_centroids.groupby('adm_order')['adm_rate'].apply(mean_confidence_interval).to_list(),
            columns=['adm_mean','adm_ci_low','adm_ci_high']
        )
            .reset_index()
            .rename(columns={'index':'cluster'})
    )
    adm_stds = (
        all_ordered_cluster_centroids
            .groupby('adm_order')['adm_rate']
            .std()
            .reset_index()
            .rename(columns={'adm_order':'cluster','adm_rate':'adm_rate_std'})
    )
    adm_rate_labels = adm_rates_with_confints.merge(adm_stds, how='left', on='cluster')

    #### PLOT PATTERNS ####

    colors = cm.brg(np.linspace(0,1,num_clusters))

    fig = plt.figure(figsize=(42/2/2.54,25/2.54))

    for i in range(num_clusters):
        plot_cluster_stability = (
            all_ordered_cluster_centroids.query(f'adm_order=={i}')
                .drop(columns=['adm_rate','adm_order'])
                .T
                .reset_index()
                .rename(columns={'index':'time_step'})
                .pipe(lambda x: x.assign(time_step = x.time_step+1))
                # adjust time-steps to match other interval plots
                .assign(time_step = lambda x: (x.time_step.max()-x.time_step+1)*-1)
        )

        plot_cluster_stability = pd.melt(plot_cluster_stability, id_vars='time_step')

        # this calculates num visits (remember num_visits above is actually the number of intervisit times we're choosing)
        num_gpes = plot_cluster_stability.time_step.nunique()
        num_gpes = num_gpes+1

        #add subplot
        ax = fig.add_subplot(gs[i,0])

        ax = sns.boxplot(data = plot_cluster_stability, x = 'time_step',y='value',ax=ax, showfliers=False,
                         **{'boxprops':{'edgecolor':None}})
        ax.set_ylim(5,55)
        ax.set(xlabel=None)
        ax.set(ylabel=None)
        ax.set_title(f'ARCG {i}', fontsize=16, c='black', y=1, pad=-16)
        ax.tick_params(axis='both',labelsize=16)

        # remove '-' from labels to avoid confusion
        labels = [item.get_text().replace('-','') for item in ax.get_xticklabels()]
        ax.set_xticklabels(labels)

        # show only the bottom-most subplot xlabel (neater plot)
        if i == num_clusters-1:
            # hide some xticks for neatness
            for label in ax.get_xticklabels():
                if int(label.get_text())%3 == 0:
                    label.set_visible(True)
                else:
                    label.set_visible(False)
        else:
            ax.get_xaxis().set_visible(False)

        #### PLOT ADMISSION RATES ####

        ax2 = fig.add_subplot(gs[i,1])

        sns.barplot(
            data = all_ordered_cluster_centroids.query(f'adm_order=={i}'),
            y='adm_order',
            x='adm_rate',ax=ax2,
            orient='h',
            color='blue',
            alpha=0.5
        )
        ax2.tick_params(axis='x', labelsize=16, labelcolor='white')
        ax2.set_xlim(0,1)
        ax2.axis('off')

        #### ADD ADMISSION RATE LABELS ####

        df = adm_rate_labels.query(f'cluster=={i}')

        mean_val = str(df.adm_mean.multiply(100).values.round(1)).replace('[','').replace(']','')
        # CIs
        l_ci = str(df.adm_ci_low.multiply(100).values.round(1)).replace('[','').replace(']','')
        u_ci = str(df.adm_ci_high.multiply(100).values.round(1)).replace('[','').replace(']','')
        # Standard devaition
        adm_std = str(df.adm_rate_std.multiply(100).values.round(1)).replace('[','').replace(']','')
        plus_minus = '\u00B1'
        ax2.text(x=float(mean_val)*1.2/100,y=0,s=f'{mean_val}\n{plus_minus}{adm_std}',
                 va='center',fontsize=16, rotation=0)

    label_font_size =20

    # set shared x label
    fig.text(0.5,0.07,'GP Interevent Interval Before Index Date', ha='center', va='center', fontsize = label_font_size)
    # set shared y label
    fig.text(0.07,0.5,'Interevent Intervals [days]', ha='center', va='center', fontsize =label_font_size, rotation='vertical')
    # set x label for admission rates
    fig.text(0.91,0.5,'Admission Rate [%]', ha='center', va='center', fontsize =label_font_size, color='blue', rotation=90)

    fig.subplots_adjust(wspace=0,hspace=.05)


# STEP 2 - Cosine distances

def measure_cosine_distance_between_cluster(cluster_dataframe):
    '''
    This function takes a dataframe of clusters and measures the cosine distance between all clusters
    :param cluster_dataframe: dataframe of multiple clustering runs/trials
    :return: dataframe comparing all cosine distances between these clusters
    '''
    try:
        cluster_dataframe = cluster_dataframe.drop(columns=['adm_rate','adm_order'])
    except:
        pass

    all_rows = list(cluster_dataframe.index.values)

    compared_vals = []
    rows_already_used=[]

    for row in tqdm(all_rows):
        # identify base row that needs to be compared to other rows for this iteration
        base_centroid = cluster_dataframe.loc[row]
        # add this row to a list that keeps track of rows that have already been compared to all others
        rows_already_used = np.append(rows_already_used,row)
        # use this list to identify what needs to be dropped from the whole dataset, as it's already been compared
        rows_to_compare = [x for x in all_rows if x not in rows_already_used]

        # use the above list to compare with all remaining rows
        for row_to_compare in rows_to_compare:
            # pull the centroid related to this row index
            centroid_to_compare = cluster_dataframe.loc[row_to_compare]
            # get the cosine distance between this row and the base_row
            val = scipy.spatial.distance.cosine(base_centroid,centroid_to_compare)
            # add this to a list
            compared_vals = np.append(compared_vals,val)


    # check no duplicates (indicating an issue with the above loop)
    if (pd.Series(compared_vals).duplicated().sum()) == 0:
        pass
    else:
        print('duplicates detected, possible issue with loop above \n (When using for Random Clusters This is Not Relevant)')

    return compared_vals


def create_pseudo_clusters(joined_df, num_iets, num_clusters_to_make):
    '''
    This creates a set of pseudo clusters based on the distribution of interevent times from the baseline cohort
    :param joined_df: dataframe of raw extract from SAIL SQL database - which has been joined by patient identifier (ALF_PE)
    :param num_iets: the desired number of interevent times for all pseudo clusters
    :param num_clusters_to_make: the number of pseudo clusters to create
    :return: dataframe of pseudo clusters (the cluster centroids are returned)
    '''

    # get interevent distribution for whole of baseline cohort
    interevent_dist = joined_df.date_diff

    # count these and get frequency of all interevent values to create random number generator
    freq_table = interevent_dist.value_counts(normalize=True).sort_index()
    values = freq_table.index
    probabilities = freq_table.values

    # function to generate random values based on the above distribution
    def random_interevent_time_generator(num_samples):
        return random.choices(values,weights=probabilities,k=num_samples)

    # make pseudo clusters
    num_pseudo_clusters = num_clusters_to_make

    pseudo_clusters = pd.DataFrame()

    for i in range(num_pseudo_clusters):
        new_cluster = pd.Series(random_interevent_time_generator(num_iets)).to_frame().T
        pseudo_clusters = pd.concat([pseudo_clusters, new_cluster], axis=0, ignore_index=True)

    return pseudo_clusters


def compare_cosine_difference_cdf(true_cluster_cosine_distances, pseudo_cluster_cosine_distances,stat='proportion'):
    '''
    :param true_cluster_cosine_distances: dataframe of true cosine distances
    :param pseudo_cluster_cosine_distances: dataframe of pseudoe cosine distances
    :param stat: as per Seaborn ecdfplot documentation: 'proportion' or 'count'
    :return: plot comparing CDFs of 2 different distributions of cosine distances (true vs pseudo clusters)
    '''

    cosine_comp_df = (
        pd.concat([
            (pd.DataFrame(true_cluster_cosine_distances).assign(source='True Clusters')),
            (pd.DataFrame(pseudo_cluster_cosine_distances).assign(source='Pseudo Clusters'))]
        )
            .rename(columns={0:'cosine_distance'})
            .reset_index(drop=True)
    )

    fig,ax = plt.subplots(1,1,figsize=(16/2.54,20/2.54))
    # cumulative distribution 
    sns.ecdfplot(data=cosine_comp_df, x='cosine_distance',hue='source',ax=ax,
                 lw=3,stat=stat)
    ax.tick_params('y',labelsize=16)
    ax.set_ylabel('Cumulative Probability', fontsize=20)

    plt.setp(ax.get_legend().get_texts(), fontsize='16')
    ax.set_xlabel('Cosine Distance', fontsize=20)
    ax.legend_.set_title(None)
    ax.set_xlim(0,1)
    ax.tick_params('x',labelsize=16)

    fig.subplots_adjust(wspace=0,hspace=.05)


# STEP 3 - Dictionary of Clusters

def plot_distribution_cosine_distances(compared_vals, cut_off_cosine_distance):
    '''
    plots distribution of above cosine distances from above and identify threshold
    :param compared_vals: dataframe of all compared cluster cosine distances
    :param cut_off_cosine_distance: set a vertical line at a proposed cut-off
    :return: plot of distribution of cosine distances
    '''

    fig, ax = plt.subplots(1,1,figsize=(15,9))
    ax = sns.histplot(compared_vals, stat='probability', log_scale=(False,False))
    ax.axvline(x=cut_off_cosine_distance,color='orange',linestyle='--')
    ax.text(x=cut_off_cosine_distance+0.005,y=0.1,s=f'Cosine Distance = {cut_off_cosine_distance}',fontsize=18,color='orange')
    ax.set_title('Distribution of Cosine Distances - True Clusters',fontsize=26)
    ax.tick_params(axis='both',labelsize=16)
    ax.set_xlabel('Cosine Distance', fontsize=22)
    ax.set_ylabel('Probability', fontsize=22)


def make_cluster_dictionary(all_ordered_cluster_centroids, cosine_distance_threshold):
    '''
    Takes a dataframe of repeated clustering runs, and using a given cosine distance threshold,
    identifies unique clusters
    :param all_ordered_cluster_centroids: dataframe of repeated clustering runs
    :param cosine_distance_threshold: chosen cosine distance threshold (usually identified from STEP 2 above)
    :return: 2 outputs: 
            1. dataframe one is a dictionary of unique clusters
            2. dataframe 2 identifies how often a unique cluster matched and excluded another cluster
    '''

    # identify number of clusters and num of runs performed in the cluster trials dataframe
    num_clusters= all_ordered_cluster_centroids.adm_order.nunique()
    num_runs = int(len(all_ordered_cluster_centroids)/num_clusters)

    # assign label to each group of clusters from each clustering run
    group_counter = []
    for i in range(0,num_runs):
        to_add = [i]*num_clusters
        group_counter = np.append(group_counter,to_add)
    group_counter = pd.DataFrame({'group':group_counter})
    cluster_groups = pd.concat([all_ordered_cluster_centroids,group_counter], axis=1)

    # identify groups (i.e. the clusters associated with each run from above)
    groups = cluster_groups.group.unique()

    # make dictionary using clusters from 1st group (run)
    cluster_dictionary = cluster_groups.query('group==@groups[0]').drop(columns=['adm_rate','adm_order','group'])
    closest_matched_clusters = []

    for group in groups:
        # sequentially get clusters from each group
        current_group = cluster_groups.query(f'group=={group}').drop(columns=['adm_rate','adm_order','group'])
        for row in current_group.index:
            # test each cluster in each group
            test_row = current_group.loc[row]
            cos_vals = []
            cluster_indexes_tracker = []

            # calculate cosine distance between
            for cluster_index in cluster_dictionary.index:
                dict_cluster = cluster_dictionary.loc[cluster_index]
                cos_val = scipy.spatial.distance.cosine(dict_cluster,test_row)
                cos_vals = np.append(cos_vals,cos_val)
                cluster_indexes_tracker = np.append(cluster_indexes_tracker,cluster_index)

            # make dataframe of cosine distances and associated cluster
            cluster_indexes_tracker = (pd.DataFrame([cluster_indexes_tracker,cos_vals])).T
            cluster_indexes_tracker.columns = ['cluster','cos_vals']
            # this makes the index the dictionary cluster id - useful downstream
            cluster_indexes_tracker = cluster_indexes_tracker.set_index('cluster')

            # if new cluster is different enough to the cluster_dictionary, then add to the dictionary
            if all(x>cosine_distance_threshold for x in cos_vals):
                cluster_dictionary.loc[row] = test_row
            # otherwise, identify which cluster most closely matched the test cluster
            else:
                closest_cluster = cluster_indexes_tracker.idxmin().values # note the index is the same as the cluster id
                closest_matched_clusters = np.append(closest_matched_clusters,closest_cluster)

    return cluster_dictionary, closest_matched_clusters


## plot above

def plot_cluster_dictionary(cluster_dictionary, closest_matched_clusters):
    '''
     This then plots the result of the above, displaying:
    -The individual unique clusters
    -The fraction of clusters that matched other clusters
    -The original id of these clusters (which is a marker of how far down the list of clusters it was until this was identified)
    :param cluster_dictionary: dictionary of clusters from above
    :param closest_matched_clusters: matched cluster numbers
    :return:
    '''

    cluster_id_dict = (cluster_dictionary
        .reset_index()
        .rename(columns={'index':'cluster_ids'})
        .filter(items=['cluster_ids'])
        .to_dict()
    ['cluster_ids']
        )

    plot_df =(cluster_dictionary
              .reset_index(drop=True)
              .unstack()
              .reset_index()
              .rename(columns={'level_0':'visit_num', 'level_1':'cluster_id', 0:'cluster_center'})
              )


    # add blank values for final value for plotting purposes to allow a finish point
    visit_num = cluster_dictionary.columns.max()+1 # this finds the max number of visits in the clusters and adds 1
    cluster_id = range(0,len(cluster_id_dict)) # finds total number of distinct clusters in dictionary
    cluster_center=0
    empty_df = pd.DataFrame({'visit_num':visit_num, 'cluster_id':cluster_id, 'cluster_center':cluster_center})

    # merge with empty
    plot_df = pd.concat([plot_df,empty_df],axis=0)

    # add additional cols for labelling and plotting
    plot_df = (plot_df
               # making the visits negative is critical to orient the plot and cumulative sum correct
               .assign(visit_num = lambda x: (x.visit_num+1)*-1)
               .sort_values(by=['cluster_id','visit_num'])
               .pipe(lambda x: x.assign(cumulative= x.groupby('cluster_id')['cluster_center'].cumsum(),
                                        original_cluster_label = x.cluster_id.replace(cluster_id_dict)
                                        )
                     )
               .pipe(lambda x: x.assign(cumulative = x.cumulative*-1))
               )


    ## this gives the fraction of all the clusters that 'excluded' other clusters due to matching
    matches = (pd.Series(closest_matched_clusters)
               .value_counts(normalize=True)
               .round(2)
               .sort_index(ascending=False)
               .to_frame()
               .reset_index()
               .rename(columns={'index':'original_cluster_label', 0: 'fraction_of_clusters_excluded'})
               )

    # merge matches with plot df so ordering can be done by match rate
    plot_df = (
        plot_df
            .merge(matches,how ='left',on='original_cluster_label')
            .assign(fraction_of_clusters_excluded = lambda x: x.fraction_of_clusters_excluded.fillna(0))
            .sort_values(by=['fraction_of_clusters_excluded'],ascending=False)
            .assign(sorted_plot_id = lambda x: x.groupby(['original_cluster_label']).ngroup())
    )

    # this assigns an order to clusters with the higest match rates, for better plotting
    match_orders = (
        plot_df
            .filter(items=['original_cluster_label','fraction_of_clusters_excluded'])
            .drop_duplicates()
            .sort_values(by=['fraction_of_clusters_excluded'],ascending=False)
            .reset_index(drop=True)
            .reset_index()
            .rename(columns={'index':'plot_exclusion_order'})
            .drop(columns=['fraction_of_clusters_excluded'])
    )

    plot_df = (
        plot_df
            .merge(match_orders, how='left', on ='original_cluster_label')
    )

    print('total unique clusters ', plot_df.sorted_plot_id.nunique())

    plot_df = (
        plot_df
            # remove any keys that are standalone (i.e. don't have any matches)
            .query('fraction_of_clusters_excluded>0')
    )

    print('total unique clusters with at least 1 match ', plot_df.sorted_plot_id.nunique())

    #plot
    fig, ax = plt.subplots(1,1,figsize=(16/2.54,20/2.54))
    for plot_exclusion_order,df in plot_df.groupby('plot_exclusion_order'):
        ax.plot(df.cumulative, df.plot_exclusion_order, marker='|', linewidth=2.5, ms=20)

    cluster_labels_and_fractions = (
        plot_df
            .filter(items=['plot_exclusion_order','fraction_of_clusters_excluded'])
            .drop_duplicates()
            .set_index('plot_exclusion_order')
    )

    for cluster_id in cluster_labels_and_fractions.index:
        fraction = cluster_labels_and_fractions.fraction_of_clusters_excluded.loc[cluster_id]
        ax.text(x=10, y=cluster_id, s=f'{int(np.round(fraction*100,1))} %',color='blue',fontsize=16)
    ax.set_xlim(-600,100)
    ax.set_yticklabels([])
    ax.set_ylabel('Unique GPE-Clusters Identified',fontsize=20)
    ax.set_xlabel('Timeline Before Final GP Event [days]',fontsize=20)
    ax.tick_params('x',labelsize=16)
    ax.invert_yaxis()


    plt.draw()
    labels = [item.get_text().replace('−','') for item in ax.get_xticklabels()]
    ax.set_xticklabels(labels)
    # don't show any labels >0
    for item in ax.get_xticklabels():
        if item.get_position()[0] >0:
            item.set_visible(False)
        else:
            pass

    fig.text(0.94,0.5,'Matching Clusters after 1000 Trials [%]',rotation=90,fontsize=20,color='blue', ha='center', va='center')


#################################
'''Advanced Prediction Modelling'''
#################################

## At this point a 'final study table' needs to be made by joining all baseline characteristics with cluster labels

def make_final_study_table(joined_df, clusters_with_ids):

    study_table = (
        joined_df
            # remove any columns that will cause duplicates to be retained
            .filter(items =
                    ['alf_pe', 'pseudo_id', 'gndr_cd', 'ethnicity', 'wimd_2014_decile','study_start_age','outcome_indicator',
                     'hypertension','diabetes', 'af', 'ihd', 'vte', 'ckd', 'valvular', 'ccf', 'cancer',
                     'asthma_copd', 'epilepsy', 'falls', 'depression', 'chol','mania_schizo', 'cld_panc',
                     'ra_sle', 'oa', 'ctd', 'hemiplegia', 'lds', 'obesity', 'osteoporosis',
                     'pvd', 'pud', 'drug_abuse','total_drugs', 'visits_within_1yr', 'total_gp_visits',
                     'outcome_or_pseudo_date'
                     ]
                    )
            # Drop duplicates (i.e. any row that pertains to the same patient admission)
            .drop_duplicates()
            # remove any patient without a WIMD
            .dropna(subset= ['wimd_2014_decile'])
            # keep only male and female gender codes
            .query('gndr_cd <=2')
            # .merge(km_clusters_with_ids.filter(['pseudo_id','cluster_label']), how='left', on='pseudo_id')
            # .rename(columns={'cluster_label':'kmeans_cluster_label'})
            .merge(clusters_with_ids.filter(['pseudo_id','cluster_label']), how='left', on='pseudo_id')
            .rename(columns={'cluster_label':'gmm_cluster_label'})
    )

    print('''
    Any patient that has not visited the GP enough times to be clustered will be assigned a 
    cluster label = '-1'
    '''
          )

    study_table = (
        study_table
            # .assign(kmeans_cluster_label = lambda x: x.kmeans_cluster_label.fillna(-1).astype(int))
            .assign(gmm_cluster_label = lambda x: x.gmm_cluster_label.fillna(-1).astype(int))
    )

    print('''
    Important: Remember the final study table will have duplicated ALF_PEs for patients 
    admitted >1 time. Therefore the study_table will be longer than the number of unique ALF_PEs)
    ''')

    return study_table



def create_dummies_for_study_table(study_table, cluster_type):
    '''
    :param study_table: final study table that includes baseline features and cluster labels
    :param cluster_type: select 'gmm' (default) or 'kmeans' (remnant of earlier work that started with kmeans clustering)
    :return: dummy variables for categorical columns in study table
    '''

    try:
        if cluster_type == 'kmeans':
            input_df = study_table.drop(columns=['gmm_cluster_label'])
        if cluster_type == 'gmm':
            input_df = study_table.drop(columns=['kmeans_cluster_label'], errors='ignore')
        if cluster_type == 'both':
            input_df = study_table
        # for the 2nd stage model - drop both of the above
        if cluster_type == 'none':
            input_df = study_table.drop(columns=['kmeans_cluster_label','gmm_cluster_label'], errors='ignore')
    except:
        input_df = study_table


    # fill wimd with median value
    if 'wimd_2014_decile' in input_df.columns:
        median_wimd = input_df.wimd_2014_decile.median()
        input_df = (
            input_df.assign(wimd_2014_decile = lambda x: x.wimd_2014_decile.fillna(median_wimd))
        )

    # get dummies where required
    target_cols = ['ethnicity','kmeans_cluster_label', 'gmm_cluster_label']
    # this identifies any column in the input_df columns that needs to turned into a dummy
    # (as per target_cols list)
    dum_cols = [col for col in input_df.columns if any(col_t in col for col_t in target_cols)]
    df = pd.get_dummies(data=input_df, columns=dum_cols)

    if df.isnull().sum().sum() != 0:
        print('Study table still has nulls that have not been handled')
    else:
        pass

    X = df.drop(columns=['alf_pe','outcome_indicator','pseudo_id','outcome_or_pseudo_date'])
    y = df.outcome_indicator

    return X,y


def train_advanced_prediction_model(study_table, refit_parameter, model_type='lr', cluster_type='gmm'):
    '''
    Trains a cross-validated grid search for chosen model (as per scikit-learn)
    :param study_table: final study table that includes baseline features and cluster labels
    :param refit_parameter: scorer to identify best parameters for refitting estimator (as per sci-kit learn 3.3 metrics and scoring)
    :param model_type: logistic regression: 'lr', naive bayes: 'nb', xgboost: 'xgb', multilayer perceptron: 'mlp'
    :param cluster_type: 'gmm' (default) or 'kmeans': as above remnant of earlier work
    :return: Returns 2 items:
            1. scikit-learn gridsearch object
            2. the scaler used - for use with test data if needed
    '''

    # this function is defined above
    X,y = create_dummies_for_study_table(study_table, cluster_type)

    X_cols = X.columns

    # scale
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    # model type
    if model_type=='lr':
        model = LogisticRegression(max_iter=1000)
        param_grid=[{
            'penalty' : ['l1'],
            'max_iter':[1000],
            'C':[100,10,1,0.1,0.01],
            'class_weight':['balanced'],
            'solver':['liblinear']}
        ]

    if model_type=='nb':
        param_grid={'alpha':[0.5,1]}
        model = MultinomialNB()

    # XGB
    if model_type =='xgb':
        model = XGBClassifier(use_label_encoder=False, eval_metric='aucpr') #area under the precision recall curve
        #cross validate
        param_grid=[
            {'eta' : [0.01,0.05], # aka learning rate
             'max_depth':[3,6],
             'subsample': [0.75,1],
             'n_estimators':[400,800,1000],
             'scale_pos_weight':[5,7,10]
             }
        ]

    # MLP
    if model_type == 'mlp':
        model = MLPClassifier()
        param_grid=[
            {'hidden_layer_sizes':[(150,100,50)],
             'alpha' : [0.0001,0.005],
             'learning_rate':['constant','adaptive'],
             'max_iter': [200,500]
             }
        ]

    scoring=['accuracy','f1','neg_log_loss','neg_brier_score',
             'roc_auc','recall','precision', 'average_precision']
    gs = GridSearchCV(
        model,
        param_grid=param_grid,
        scoring = scoring,
        cv = 5,
        refit=refit_parameter,
        verbose=1,
        return_train_score=True,
        n_jobs=12
    ).fit(X,y)

    return gs, scaler



def train_all_model_types(study_table, refit_parameter):
    '''
    Using above gridsearch function, trains predictors for each class of model
    :param study_table: final study table that includes baseline features and cluster labels
    :param refit_parameter: scorer to identify best parameters for refitting estimator (as per sci-kit learn 3.3 metrics and scoring)
    :return:dictionary of scikit-learn GridSearchCV objects for each model type
    '''

    gmm_cv_models_dict = {}

    for model_type in ['lr','nb','xgb','mlp']:
        performance_df, fit_scaler = train_advanced_prediction_model(
            study_table=study_table,
            refit_parameter = refit_parameter,
            model_type=model_type,
            cluster_type='gmm')
        gmm_cv_models_dict[model_type] = performance_df

    #add scaler - note it doesn't matter that iteration has its own scaler as these should be identical (same data)
    gmm_cv_models_dict['scaler'] = fit_scaler

    return gmm_cv_models_dict



def plot_performance_all_models(cv_models_dict):
    '''
    Takes a dictionary of scikit-learn GridSearchCV objects and plots their comparative performance
    :param cv_models_dict: dictionary of models from function 'train_all_model_types'
    :return: Barplot comparing all results from model dictionary above
    '''

    # retrieve all cv results for only the best models for each classifier
    all_best_results = pd.DataFrame()

    # get names of all models used in dictionary (need to remove the scaler)
    models = cv_models_dict.keys()
    models = [value for value in models if value != 'scaler']

    for m in models:
        best = cv_models_dict[m].best_index_
        results = cv_models_dict[m].cv_results_

        key_list =[]
        result_list = []
        model_list = []
        for key in results.keys():
            result_list.append(results[key][best])
            key_list.append(key)
            model_list.append(m)

        model_specific_results = (pd.DataFrame([key_list,result_list,model_list])).T
        all_best_results = pd.concat([all_best_results,model_specific_results])

    all_best_results.columns=['value','result','model']

    plot_data = (all_best_results
                 .query('value.str.contains("split")')
                 .pipe(lambda x: x.assign(result = x.result.abs())
                       )
                 )
    plot_data['value'] = plot_data['value'].str.slice(start=7)

    # this keeps only results for test performance and removes train performance
    plot_data = plot_data.query('value.str.contains("test")')

    hue_order = ['mlp','xgb','lr','nb']

    palette = ['blue','grey','darkgrey','lightgrey']

    # VALUES THAT YOU WANT TO BE LOWER
    #     fig, (ax1,ax2) = plt.subplots(2,1,figsize=(9,8), sharey=False, gridspec_kw={'height_ratios':[2,1]})
    fig = plt.figure()
    fig.set_figheight(7.5)
    fig.set_figwidth(9)
    ax1 = plt.subplot2grid(shape=(3,6),loc=(0,0),colspan=6,rowspan=2)
    ax2 = plt.subplot2grid(shape=(3,6),loc=(2,0),colspan=6,rowspan=1)
    #     ax3 = plt.subplot2grid(shape=(3,6),loc=(0,5),colspan=1,rowspan=3)

    # VALUES THAT YOU WANT TO BE HIGHER
    ax1_vals = ['recall','f1','roc_auc','precision']
    pattern = '|'.join(ax1_vals)
    subplot_data = plot_data[plot_data.value.str.contains(pattern)]
    subplot_data = subplot_data[~subplot_data.value.str.contains('average_precision')]
    ax1 = sns.barplot(y='value', x='result', data = subplot_data, hue='model', hue_order = hue_order,
                      palette=palette, orient='h', ax=ax1)
    ax1.tick_params(axis='x',rotation=0, labelsize=16)
    ax1.tick_params(axis='y', labelsize=16)
    ax1.set_ylabel('', fontsize=1)
    ax1.set_xlabel('', fontsize=1)
    ax1.legend(fontsize=12, loc='upper left')
    labels = ['F1 Score', 'AUROC', 'Recall', 'Precision']
    ax1.set_yticklabels(labels)


    ax2_vals = ['brier','neg_log_loss'] # drop log_loss
    pattern = '|'.join(ax2_vals)
    subplot_data = plot_data[plot_data.value.str.contains(pattern)]
    ax2 = sns.barplot(y='value', x='result', data = subplot_data, hue='model', hue_order = hue_order,
                      palette=palette, orient ='h', ax=ax2)
    ax2.tick_params(axis='x',rotation=0, labelsize=18)
    ax2.tick_params(axis='y', labelsize=18,colors='saddlebrown')
    ax2.set_xlabel('Score', fontsize=22)
    ax2.set_ylabel('', fontsize=1)
    ax2.legend(fontsize=12)
    ax2.set_xlim(0,1)
    # specify tick labels
    #     labels = [item.get_text() for item in ax2.get_xticklabels()]
    labels = ['Brier Score', 'Log Loss']
    ax2.set_yticklabels(labels)


    # capitalize legend labels
    for ax in [ax1,ax2]:
        handles, labels = ax.get_legend_handles_labels()
        new_labels = [item.upper() for item in labels]
        ax.legend(handles=handles,labels=new_labels,fontsize=16,frameon=False,labelspacing=.15)

    # HIDE legend in ax2
    ax2.get_legend().remove()

    # top arrrows
    text1 = 'Higher\nScore\nBetter'
    y_pos = ax1.get_ylim()[0]/2
    ax1.text(
        .9, y_pos, s=text1, va='center',ha='center',rotation=0, size = 18, color ='black',
        bbox = dict(boxstyle='rarrow,pad=0.2',fc='white',ec='black',lw=2)
    )
    text1 = 'Lower\nScore\nBetter'
    y_pos = ax2.get_ylim()[0]/2 - .1
    ax2.text(
        0.9, y_pos, s=text1, va='center',ha='center',rotation=0, size = 18, color ='saddlebrown',
        bbox = dict(boxstyle='larrow,pad=0.2',fc='white',ec='saddlebrown',lw=2)
    )

    # hide spines
    for ax in [ax1,ax2]:
        ax.spines.right.set_visible(False)
        ax.spines.left.set_visible(False)
        ax.spines.top.set_visible(False)

    ax1.spines.bottom.set_visible(False)


    # remove x labels on top plot
    ax1.get_xaxis().set_visible(False)

    fig.tight_layout(h_pad=4,pad=0)



'''Comparative Modelling Including Additional Features'''

# Demonstrate that GPE-Clusters add value

def make_comparator_xgb(study_table):
    '''
    Train 2x XGB models to demonstrate utility of cluster
    :param study_table: final study table that includes baseline features and cluster labels
    :return: dictionary with cross validation results and feature names for XGBoost models trained with and without clusters
    '''

    xgb_with_or_without_dict = {}

    # declare sparse feature set
    sparse_baseline_features = [
        'alf_pe','pseudo_id','outcome_indicator',
        'outcome_or_pseudo_date', 'kmeans_cluster_label',
        'gmm_cluster_label', 'gndr_cd', 'ethnicity',
        'wimd_2014_decile', 'study_start_age', 'visits_within_1yr'
    ]

    # get study table with limited feature set only
    study_table_ = study_table.filter(items=sparse_baseline_features)

    for cluster_type in ['none','gmm']:
        # this function is defined above
        X,y = create_dummies_for_study_table(study_table_,cluster_type=cluster_type)
        X_cols = X.columns

        # scale
        scaler = MinMaxScaler()
        scaler.fit(X)
        X = scaler.transform(X)

        # XGB - note that hyperparameters are chosen as per best XGB model above
        model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='aucpr',
            eta = 0.01, # aka learning rate
            max_depth = 3,
            subsample = 1,
            n_estimators = 400,
            scale_pos_weight=5

        )

        scoring=['accuracy','f1','neg_log_loss','neg_brier_score',
                 'roc_auc','recall','precision', 'average_precision']

        cv_results = cross_validate(model, X,y, cv=5,
                                    scoring = scoring,
                                    n_jobs = 12,
                                    verbose=1,
                                    return_estimator=True)

        xgb_with_or_without_dict[f'{cluster_type}_cv_dict'] = cv_results
        xgb_with_or_without_dict[f'{cluster_type}_features'] = X_cols

    return xgb_with_or_without_dict


def xgb_head_to_head_plot(head_to_head_dict):
    '''
    Plot performance of above head-to-head comparison
    :param head_to_head_dict: dictionary output from 'make_comparator_xgb'
    :return: plot
    '''
    gmm_roc = head_to_head_dict['gmm_cv_dict']['test_roc_auc']
    nocluster_roc = head_to_head_dict['none_cv_dict']['test_roc_auc']
    gmm_recall = head_to_head_dict['gmm_cv_dict']['test_recall']
    nocluster_recall = head_to_head_dict['none_cv_dict']['test_recall']
    gmm_precision = head_to_head_dict['gmm_cv_dict']['test_precision']
    nocluster_precision = head_to_head_dict['none_cv_dict']['test_precision']
    gmm_brier = np.abs(head_to_head_dict['gmm_cv_dict']['test_neg_brier_score'])
    nocluster_brier = np.abs(head_to_head_dict['none_cv_dict']['test_neg_brier_score'])

    results_df = pd.DataFrame(
        {
            'gmm_roc':gmm_roc,
            'nocluster_roc':nocluster_roc,
            'gmm_recall':gmm_recall,
            'nocluster_recall':nocluster_recall,
            'gmm_precision':gmm_precision,
            'nocluster_precision':nocluster_precision,
            'gmm_brier':gmm_brier,
            'nocluster_brier':nocluster_brier
        }
    )
    results_df = pd.melt(results_df.T.reset_index(),id_vars=['index'])
    results_df = results_df.rename(columns={'index':'model'}).drop(columns=['variable'])
    results_df = (results_df
                  .assign(data = results_df.model.str.split('_').str.get(0),
                          metric = results_df.model.str.split('_').str.get(1)
                          )
                  .drop(columns='model')
                  )
    # print numeric results
    print(results_df.groupby(['data','metric']).describe())

    fig, ax = plt.subplots(1,1,figsize=(15,9))
    ax = sns.boxplot(ax=ax,data=results_df,y='value',x='metric',hue='data',showfliers=False)
    ax.legend_.set_title(None)
    ax.set_ylim(0,1)
    ax.set_xlabel('Metric', fontsize=22)
    ax.set_ylabel('Score', fontsize=22)
    ax.tick_params(axis='both',labelsize=18)
    plt.setp(ax.get_legend().get_texts(), fontsize='16')


def xgb_head_to_head_feature_importance(xgb_with_or_without_dict):
    '''
    Plot feature importances of above head-to-head comparison
    :param head_to_head_dict: dictionary output from 'make_comparator_xgb'
    :return: plot
    '''
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(15,9))
    axs = [ax1,ax2]

    for feature_set,ax in zip(['gmm','none'],[ax1,ax2]):

        importance_dict ={}
        for fold in range(0,5):
            importance_dict[fold] = xgb_with_or_without_dict[f'{feature_set}_cv_dict']['estimator'][fold].feature_importances_

        col_names = xgb_with_or_without_dict[f'{feature_set}_features']

        importance_df = pd.DataFrame(importance_dict,index=col_names)
        importance_df = (importance_df
                         .assign(mean_importance = lambda x: x.apply(np.mean, axis=1))
                         .sort_values(by='mean_importance', ascending=False)
                         .head(10)
                         .drop(columns=['mean_importance'])
                         .T
                         .reset_index()
                         .rename(columns={'index':'fold'})
                         )
        importance_df = pd.melt(importance_df, id_vars='fold')
        importance_df

        if feature_set == 'gmm':
            title = 'GPE-Cluster'
        if feature_set == 'none':
            title = 'No GPE-Cluster'

        sns.barplot(data = importance_df, y='variable', x = 'value', orient='h', ax=ax)
        ax.set_title(f'Top 10 Most Important Features ({title})', fontsize=20)
        ax.tick_params(axis='both',rotation=0, labelsize=16)
        ax.set_xlabel('Gain',fontsize=16)
        ax.set_xlim(0,1)
        ax.set_ylabel('')

    fig.tight_layout()


# Demonstrate that a sparse model with GPE-Clusters matches a conventional approach

def train_cluster_vs_conventional_model(study_table):
    '''
    :param study_table: final study table that includes baseline features and cluster labels
    :return: sparse model, conventional model
    '''

    # features to include for the sparse model
    sparse_baseline_features = [
        'alf_pe','pseudo_id','outcome_indicator',
        'outcome_or_pseudo_date', 'gmm_cluster_label', 'gndr_cd', 'ethnicity',
        'wimd_2014_decile', 'study_start_age', 'visits_within_1yr'
    ]
    refit_parameter = 'precision'

    # train sparse model with GMM
    sparse_gmm_model_dict, _ = train_advanced_prediction_model(study_table.filter(items=sparse_baseline_features), refit_parameter, model_type='mlp', cluster_type='gmm')
    # train conventional model (no GMM)
    conventional_model_dict, _ = train_advanced_prediction_model(study_table, refit_parameter, model_type='mlp', cluster_type='none')

    return sparse_gmm_model_dict, conventional_model_dict


def plot_cluster_vs_conventional_model(sparse_gmm_model_dict, conventional_model_dict):
    '''
    :param sparse_gmm_model_dict: sparse model from 'train_cluster_vs_conventional_model'
    :param conventional_model_dict: conventional model from 'train_cluster_vs_conventional_model'
    :return: plot of comparative performance
    '''
    models = [sparse_gmm_model_dict, conventional_model_dict]
    model_names = ['GPE-Cluster + Sparse Baseline Features', 'Extended Baseline Features']

    metrics = ['roc_auc','recall','precision']

    result_list = []
    metric_list = []
    model_list = []

    for model,model_name in zip(models,model_names):
        # print(model_name)
        # print('Performance Metric (Mean and CI')
        for metric in metrics:
            vals = model.cv_results_[f'mean_test_{metric}']
            # print(metric, mean_confidence_interval(vals))
            result_list.extend(vals)
            mets = [metric] * len(vals)
            metric_list.extend(mets)
            model_name_ = [model_name] *  len(vals)
            model_list.extend(model_name_)
        print('')

    both_vals = (pd.DataFrame([result_list, metric_list, model_list],
                              index=['result','metric','feature_set'])).T

    # print numeric results
    print('mean')
    print(both_vals.groupby(['feature_set','metric']).mean())
    print('SD')
    print(both_vals.groupby(['feature_set','metric']).std())

    order = ['roc_auc','precision','recall']
    palette = {'GPE-Cluster + Sparse Baseline Features':'orange','Extended Baseline Features':'grey'}
    hue_order = ['GPE-Cluster + Sparse Baseline Features', 'Extended Baseline Features']


    fig,ax = plt.subplots(1,1,figsize = (7,8))
    ax = sns.barplot(data = both_vals, y = 'metric', x='result', hue='feature_set',
                     orient='h', order = order, palette=palette,hue_order=hue_order)
    ax.set_xlabel('Score', fontsize =22)
    ax.tick_params(axis='y',labelsize=18, rotation=90)
    ax.tick_params(axis='x',labelsize=18, rotation=0)
    labels = ['AUROC', 'Precision','Recall']
    ax.set_yticklabels(labels, va='center')
    ax.set_xlim(0,1)
    y_bottom, y_top = ax.get_ylim()
    ax.set_ylim(y_bottom, y_top-0.3)

    ax.set_ylabel('', fontsize=1)
    # ax.set_title('Spars', fontsize = 20)
    ax.set_title("GPE-Cluster + Sparse Baseline Features\n Vs\nExtended Baseline Features", fontsize = 20)
    plt.setp(ax.get_legend().get_texts(), fontsize='16')
    ax.legend_.set_title(None)






'''Supplementary Plots'''

def plot_test_confusion_calibration(study_table, cv_models_dict):
    '''
    Plot both a calibration curve and confusion matrix for models above
    :param study_table: final study table that includes baseline features and cluster labels
    :param cv_models_dict: dictionary of models from function 'train_all_model_types'
    :return: calibration plot with confusion matrices for all models
    '''
    X,y = create_dummies_for_study_table(study_table, 'gmm')
    # the below is critical for handling stage 2 models
    X = X.drop(columns=['model1_preds'], errors='ignore')

    X_cols = X.columns

    # scale
    scaler = cv_models_dict['scaler']
    scaler.fit(X)
    X = scaler.transform(X)
    y = y.to_numpy()

    #RETRIEVE BEST MODELS FROM DICT
    # get names of all models used in dictionary (need to remove the scaler)
    models = cv_models_dict.keys()
    models = [value for value in models if value != 'scaler']
    num_models = len(models)

    fig = plt.figure(figsize=(15,15))
    gs = GridSpec(3,num_models, figure=fig)
    ax1 = fig.add_subplot(gs[0:2, 0:num_models])

    colors = ['pink','navy','green','yellow','black']
    assoc_model = [] # this is used to keep track of what params relate to which model

    for m,color,model_count in zip(models,colors, range(num_models)):
        best_model = cv_models_dict[m].best_estimator_
        base_fpr = np.linspace(0,1,101)
        counter=1

        tprs = []

        kf = KFold(n_splits = 5)
        for train_index, test_index in kf.split(X):
            #note train_index doesn't get used here, this is just a useful way of splitting the data to get CIs
            print(m, ' fold number: ', counter)
            # log labels so that for each value collected, we know which model this refers to
            assoc_model.append(m)
            # get predictions from model
            preds = best_model.predict(X[test_index])
            # calibration plot
            probs = best_model.predict_proba(X[test_index])
            prob_true, prob_pred = calibration_curve(y[test_index], probs[:,1], n_bins=10)
            tpr = np.interp(base_fpr, prob_pred, prob_true)
            tpr[0] = 0.0
            tprs.append(tpr)
            counter = counter+1

        tprs = np.array(tprs)
        mean_tprs = tprs.mean(axis=0)
        std = tprs.std(axis=0)

        tprs_upper = mean_tprs + std
        tprs_lower = mean_tprs - std

        ax1.plot(base_fpr, mean_tprs, label=m,color=color)
        ax1.fill_between(base_fpr, tprs_lower, tprs_upper, alpha=0.1,color=color)

        # plot confusion matrix
        ax = fig.add_subplot(gs[2, model_count])
        plt.rcParams.update({'font.size':16})
        plot_confusion_matrix(best_model, X, y, ax=ax)
        ax.set_title(m.upper())


    #finish plotting calibration curve
    ax1.plot([0,1],[0,1],color='grey',linestyle='--',label=('Perfect Calibration'))
    ax1.set_xlabel('Predicted Probability',fontsize=18)
    ax1.set_ylabel('Proportion of True Positives', fontsize=18)
    ax1.set_title(f'Calibration Curve', fontsize=22)
    ax1.legend()

    fig.tight_layout()

