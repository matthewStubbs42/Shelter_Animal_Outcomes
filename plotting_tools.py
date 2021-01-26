# Analysis, plotting
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import roc_curve, auc
import matplotlib
#matplotlib.use('AGG')
import matplotlib.pyplot as plt
from xgboost import plot_importance

##......................................................................................................##
#                                         Confusion Matrix                                               #
##......................................................................................................##

def plot_confusion_matrix(classes, model, pred=None, labels=None, cmap=plt.cm.Blues, normalize=False, figsize=(9,9)):
    
    cm = confusion_matrix(labels, pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')

#     print(cm)
    plt.figure(figsize = figsize)
    plt.imshow(cm, interpolation='none', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=13)
    plt.yticks(tick_marks, classes, fontsize=13)

    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", fontsize=18)

#     plt.tight_layout()
    plt.ylabel('True label', fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)
    title = model
    
    plt.title(title, fontsize=20)
#     plt.savefig(dump_path + "confusion matrix.png", bbox_inches = 'tight')
    return


##......................................................................................................##
#                                          moving average                                                #
##......................................................................................................##

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

    
##......................................................................................................##
#                                     XGBoost feature importance plot                                    #
##......................................................................................................##
def my_plot_importance(booster, figsize, title, **kwargs): 
    fig, ax = plt.subplots(1,1,figsize=figsize)
    myplt = plot_importance(booster=booster, ax=ax, title=title, max_num_features = 15, **kwargs)
#     plt.savefig('/xgb feature importance.png', bbox_inches='tight')
    return 