import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, brier_score_loss, accuracy_score
from sklearn.calibration import calibration_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob

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
    plt.savefig('histo.png', dpi=150, bbox_inches='tight')

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
    plt.savefig('reliability.png', dpi=150, bbox_inches='tight')

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

expname = 'model_cnn_test2'
#expname = 'model_cnn_test_new'
#expname = 'model_cnn_test2_addstorms_noaugval_patch48'
#expname = 'model_cnn_test2_addstorms_noaugval'
#expname = 'model_cnn_test2_noaugval'
expname  = 'model_cnn_test2_addstorms_rotateevenmore'
expname = 'model_cnn_test2_addstorms_rotateevenmore_noaugval'
expname = 'model_cnn_test2_addstorms_noaugval_fullval'
#expname = 'model_cnn_test2_addstorms_noaugval_cat5'
#expname = 'model_cnn_test2_addstorms2_noaugval'

df = pd.read_csv("./%s/predictions_test.csv"%expname)
categories = {'Q1':0, 'Q2':0, 'S1':1, 'S2':0, 'S3':1, 'D1':2, 'D2':2}
#categories = {'Q1':0, 'Q2':0, 'S1':1, 'S2':1, 'S3':1, 'D1':2, 'D2':2}
#categories = {'Q1':0, 'Q2':1, 'S1':2, 'S2':3, 'S3':4, 'D1':5, 'D2':6}
labels = df['label']
unique_labels = np.unique(list(categories.values()))

labels_int = np.zeros(labels.values.shape)
for type in categories.keys():
    labels_int[labels==type] = categories[type]

# predictions array as shape (examples, 3)
num_models = len(glob.glob('./%s/cnn_test*'%expname))
models = [ 'cnn_test_%03d'%i for i in range(num_models) ]
predictions_avg = 0
for mod in models:
    
    predictions_all = df[[ '%s_%d'%(mod,d) for d in unique_labels ]]
    predictions = np.argmax(predictions_all.values, axis=1)
    print('%d storms in validation set'%predictions.size)
    predictions_avg += predictions_all.values

    # AUC/BS/BSS for three modes
    print(mod)
    for m in unique_labels:
        met = classifier_metrics( labels_int==m, predictions_all.values[:,m] )
        print(list(met.values()), (labels_int==m).sum())

    #cm = confusion_matrix(labels_int, predictions)
    #print(cm)
    #cm = confusion_matrix(labels_int, predictions, normalize='true')
    #print(cm)

predictions_avg = predictions_avg/float(num_models)

print('average of models')
cal_data = []
print(labels_int.shape)
for m in unique_labels:
    met= classifier_metrics( labels_int==m, predictions_avg[:,m] )
    print(list(met.values()), (labels_int==m).sum())
    cal_data.append( calibration_curve( (labels_int==m).astype(int), predictions_avg[:,m], n_bins=10) )
#print(confusion_matrix(labels_int, np.argmax(predictions_avg, axis=1), normalize='true'))
plot_rel(cal_data)
#plot_histo(predictions_avg)
