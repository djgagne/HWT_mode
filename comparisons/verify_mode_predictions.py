import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, brier_score_loss, accuracy_score
from sklearn.calibration import calibration_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob, sys

def plot_histo(data):
    #fig = plt.figure(figsize=(8,8))
    fig, axes = plt.subplots(1, 3, sharex='col', sharey='row', figsize=(10,3))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    axes = axes.flatten()

    for n in range(3):
        ax = axes[n]
        n, bins, patches = ax.hist(data[:,n], bins=np.arange(0,1.01,0.05), facecolor='gray', log=True)
        ax.grid(linewidth=0.25, color='gray')
        ax.set_ylim((1e0,1e3))
        ax.set_xlim((0,1))
        ax.tick_params(top=True, right=True, direction='in', labelsize=8)

    axes[0].set_ylabel('Number of storms')
    axes[1].set_xlabel('Probability')
    plt.savefig('histo_%s.png'%name, dpi=150, bbox_inches='tight')

def plot_rel(data):
    #fig = plt.figure(figsize=(8,8))
    fig, axes = plt.subplots(1, 3, sharex='col', sharey='row', figsize=(10,3))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    axes = axes.flatten()

    for n in range(3):
        ax = axes[n]
        ax.plot([0,1], [0,1], color='k')

        ax.plot(data[n][1], data[n][0], markersize=6, marker='o', linewidth=2.5)
        ax.grid(linewidth=0.25, color='gray')
        ax.set_ylim((0,1))
        ax.set_xlim((0,1))
        ax.tick_params(top=True, right=True, direction='in', labelsize=8)

    axes[0].set_ylabel('Observed relative frequency')
    axes[1].set_xlabel('Forecast Probability')
    plt.savefig('reliability_%s.png'%name, dpi=150, bbox_inches='tight')

def brier_skill_score(y_true, y_pred):
    bs_climo = np.mean((y_true.mean() - y_true) ** 2)
    bs = brier_score_loss(y_true, y_pred)
    return 1 - bs / bs_climo

def classifier_metrics(y_true, model_predictions):
    metric_names = ["AUC", "Brier_Score", "Brier_Skill_Score"]
    metric_funcs = {"AUC": roc_auc_score,
                    "Brier_Score": brier_score_loss,
                    "Brier_Skill_Score": brier_skill_score
                    }

    metrics_dict = {}
    for metric in metric_names:
        metrics = metric_funcs[metric](y_true, model_predictions)
        metrics_dict[metric] = metrics
    return metrics_dict


gmm_predictions = pd.read_parquet('CNN_1_labels_20130101-0000_20131231-0000.parquet')
fnn_predictions = pd.read_csv('dist1.Q1.S1.D1.alllabel.confmin1.ep25.bs32.csv')
#fnn_predictions = pd.read_csv('dist1.Q1.S1.D1.firstlabel.confmin1.ep50.bs32.csv')
cnn_predictions = pd.read_csv('../model_cnn_test2_addstorms2_noaugval_newtrain/predictions_test.csv')

#labels = pd.read_parquet('CNN_1_labels_20130101-0000_20131231-0000.parquet')

#gmm_predictions['time'] = pd.to_datetime(gmm_predictions['time'])
gmm_predictions['run_date'] = pd.to_datetime(gmm_predictions['run_date'])
#cnn_predictions['time'] = pd.to_datetime(cnn_predictions['time'])
cnn_predictions['run_date'] = pd.to_datetime(cnn_predictions['run_date'])

# compute run_date, track_id, and track_step from Step_ID column
test = fnn_predictions['Step_ID'].str.split(pat='_', expand=True)
fnn_predictions['run_date'] = pd.to_datetime(test[3], format='%Y%m%d-%H%M')
fnn_predictions['track_id'] = test[6].astype(int)
fnn_predictions['track_step'] = test[7].astype(int) + 1

# assemble unique storm id
cnn_predictions['unique_id'] = cnn_predictions['run_date'].astype(str) + '.' + cnn_predictions['track_id'].astype(str) + '.' + cnn_predictions['track_step'].astype(str)
fnn_predictions['unique_id'] = fnn_predictions['run_date'].astype(str) + '.' + fnn_predictions['track_id'].astype(str) + '.' + fnn_predictions['track_step'].astype(str)
gmm_predictions['unique_id'] = gmm_predictions['run_date'].astype(str) + '.' + gmm_predictions['track_id'].astype(str) + '.' + gmm_predictions['track_step'].astype(str)

# filter run dates
fnn_predictions = fnn_predictions[fnn_predictions['run_date'] >= '2013-06-25 00:00:00']
gmm_predictions = gmm_predictions[gmm_predictions['run_date'] >= '2013-06-25 00:00:00']
cnn_predictions = cnn_predictions[cnn_predictions['run_date'] >= '2013-06-25 00:00:00']

# only use storms that are labeled (cnn_predictions only include these storms)
cnn_unique_ids = cnn_predictions['unique_id'].values
fnn_predictions = fnn_predictions[fnn_predictions['unique_id'].isin(cnn_unique_ids)]
gmm_predictions = gmm_predictions[gmm_predictions['unique_id'].isin(cnn_unique_ids)]

# sort based on unique id and reset index
cnn_predictions = cnn_predictions.sort_values(by=['unique_id']).reset_index()
fnn_predictions = fnn_predictions.sort_values(by=['unique_id']).reset_index()
gmm_predictions = gmm_predictions.sort_values(by=['unique_id']).reset_index()

# extract label from CNN predictions
categories = {'Q1':0, 'Q2':0, 'S1':1, 'S2':0, 'S3':1, 'D1':2, 'D2':2}
unique_labels = np.unique(list(categories.values()))
labels = cnn_predictions['label']
labels_int = np.zeros(labels.values.shape)
for type in categories.keys(): labels_int[labels==type] = categories[type]

uh_thresh, size_thresh = 75, 75
qlcs_predictions          = (  fnn_predictions['major_axis_length'] >= size_thresh ).values.astype(int)
supercell_predictions     = ( (fnn_predictions['major_axis_length'] < size_thresh) & (fnn_predictions['UP_HELI_MAX_max'] >= uh_thresh) ).values.astype(int)
disorganized_predictions  = ( (fnn_predictions['major_axis_length'] < size_thresh) & (fnn_predictions['UP_HELI_MAX_max'] < uh_thresh) ).values.astype(int)
baseline_predictions = np.array([ qlcs_predictions, supercell_predictions, disorganized_predictions ]).T

# extract predictions
cnn_predictions_all = []
for n in range(5):
    print("cnn_test_%03d_0"%n)
    cnn_predictions_all.append( cnn_predictions[["cnn_test_%03d_0"%n, "cnn_test_%03d_1"%n, "cnn_test_%03d_2"%n]].values )
cnn_predictions = np.mean( np.array(cnn_predictions_all), axis=0 )

fnn_predictions = fnn_predictions[['QLCS\nQ1+Q2+S2', 'Supercell\nS1+S3', 'Disorganized\nD1+D2']].values
gmm_predictions = gmm_predictions[['QLCS_prob', 'Supercell_prob', 'Disorganized_prob']].values

# filter out some probs that slightly exceed 1 for gmm
gmm_predictions = np.where(gmm_predictions>1, 1, gmm_predictions)

# compute metrics for one forecast type
prediction_list = [ cnn_predictions, fnn_predictions, gmm_predictions, ( cnn_predictions + fnn_predictions ) / 2.0, baseline_predictions ]
prediction_names = ['cnn', 'fnn', 'gmm', 'cnnfnn', 'base']
for name, predictions in zip(prediction_names, prediction_list):
    print(name)
    print(np.histogram(np.argmax(predictions, axis=1)))

    cal_data = []
    for m in unique_labels:
        met= classifier_metrics( labels_int==m, predictions[:,m] )
        print(list(met.values()), (labels_int==m).sum())
        cal_data.append( calibration_curve( (labels_int==m).astype(int), predictions[:,m], n_bins=10) )
    print(confusion_matrix(labels_int, np.argmax(predictions, axis=1)))
    
    plot_rel(cal_data)
#plot_histo(predictions)
