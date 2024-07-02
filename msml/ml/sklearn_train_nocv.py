#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 2021
@author: Simon Pelletier
"""

import warnings
import pickle

import numpy as np
import pandas as pd
import os
import csv
import json
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef as MCC
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from msml.utils.utils import plot_confusion_matrix
import mlflow
from sklearn.metrics import roc_curve
# binarize 
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve

np.random.seed(42)

warnings.filterwarnings('ignore')

DIR = 'src/models/sklearn/'


def count_labels(arr):
    """
    Counts elements in array

    :param arr:
    :return:
    """
    elements_count = {}
    for element in arr:
        if element in elements_count:
            elements_count[element] += 1
        else:
            elements_count[element] = 1
    to_remove = []
    for key, value in elements_count.items():
        print(f"{key}: {value}")
        if value <= 2:
            to_remove += [key]

    return to_remove


def get_confusion_matrix(reals, preds, unique_labels):
    acc = np.mean([1 if pred == label else 0 for pred, label in zip(preds, reals)])
    cm = metrics.confusion_matrix(y_true=reals,
                                  y_pred=preds,
                                  labels=np.arange(len(unique_labels)))
    figure = plot_confusion_matrix(cm, unique_labels, acc)

    # cm = np.zeros([len(unique_labels), len(unique_labels)])
    # for real, pred in zip(reals, preds):
    #     confusion_matrix[int(real), int(pred)] += 1
    # indices = [f"{lab}" for lab in unique_labels]
    # columns = [f"{lab}" for lab in unique_labels]
    return figure


def save_confusion_matrix(fig, name, acc, mcc, group, rep):
    # sns_plot = sns.heatmap(df, annot=True, square=True, cmap="YlGnBu",
    #                        annot_kws={"size": 35 / np.sqrt(len(df))})
    # fig = sns_plot.get_figure()
    dirs = '/'.join(name.split('/')[:-1])
    name = name.split('/')[-1]
    plt.title(f'Confusion Matrix (acc={np.round(acc, 3)}, mcc={np.round(mcc, 3)})')
    os.makedirs(f'{dirs}/', exist_ok=True)
    stuck = True
    while stuck:
        try:
            fig.savefig(f"{dirs}/cm_{name}_{group}_{rep}.png")
            stuck = False
        except:
            print('stuck...')
    plt.close()


def save_roc_curve(y_pred_proba, y_test, unique_labels, name, binary, acc):
    dirs = '/'.join(name.split('/')[:-1])
    name = name.split('/')[-1]
    os.makedirs(f'{dirs}', exist_ok=True)
    if binary:
        y_pred_proba = y_pred_proba[::, 1]
        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
        roc_auc = metrics.auc(fpr, tpr)
        roc_score = roc_auc_score(y_true=y_test, y_score=y_pred_proba)

        # create ROC curve
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title(f'ROC curve (acc={np.round(acc, 3)})')
        plt.legend(loc="lower right")
        stuck = True
        while stuck:
            try:
                plt.savefig(f'{dirs}/ROC_{name}.png')
                stuck = False
            except:
                print('stuck...')
        plt.close()
    else:
        # Compute ROC curve and ROC area for each class
        from sklearn.preprocessing import label_binarize
        y_preds = y_pred_proba.argmax(1)
        # y_test = np.concatenate(y_test)
        n_classes = len(unique_labels)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        classes = np.arange(len(unique_labels))

        bin_label = label_binarize(y_test, classes=classes)
        # roc_score = roc_auc_score(y_true=label_binarize(y_test, classes=classes[bin_label.sum(0) != 0]),
        #                           y_score=label_binarize(y_preds, classes=classes[bin_label.sum(0) != 0]),
        #                           multi_class='ovr')
        try:
            roc_score = roc_auc_score(y_true=label_binarize(y_test, classes=classes[y_test.sum(0) != 0]),
                                  y_score=label_binarize(y_preds, classes=classes[bin_label.sum(0) != 0]),
                                  multi_class='ovr')
        except:
            roc_score = roc_auc_score(y_true=label_binarize(y_test, classes=classes),
                                  y_score=label_binarize(y_preds, classes=classes),
                                  multi_class='ovr')
        for i in range(n_classes):
            fpr[i], tpr[i], _ = metrics.roc_curve(label_binarize(y_test, classes=classes)[:, i], y_pred_proba[:, i])
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        # roc for each class
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        plt.title(f'ROC curve (AUC={np.round(roc_score, 3)}, acc={np.round(acc, 3)})')
        # ax.plot(fpr[0], tpr[0], label=f'AUC = {np.round(roc_score, 3)} (All)', color='k')
        for i in range(n_classes):
            ax.plot(fpr[i], tpr[i], label=f'AUC = {np.round(roc_auc[i], 3)} ({unique_labels[i]})')
        ax.legend(loc="best")
        ax.grid(alpha=.4)
        # sns.despine()
        stuck = True
        while stuck:
            try:
                plt.savefig(f'{dirs}/ROC_{name}.png')
                stuck = False
            except:
                print('stuck...')

        plt.close()

    return roc_score

def plot_intervals(results, axs, method="GRU"):
    fpr_mean = np.linspace(0, 1, 100)
    interp_tprs = []
    for i in range(5):
        interp_tpr = np.interp(fpr_mean, results['fpr'][i], results['tpr'][i])
        interp_tpr[0] = 0.0
        interp_tprs.append(interp_tpr)

    tpr_mean = np.mean(interp_tprs, axis=0)
    tpr_std = np.std(interp_tprs, axis=0)

    if "GRU" in method:
        optimal_idx = np.argmax(np.asarray(tpr_mean) - np.asarray(fpr_mean))
        plt.scatter(fpr_mean[optimal_idx], tpr_mean[optimal_idx], color="black", zorder=3, s=10)

    tpr_upper = np.clip(tpr_mean + tpr_std, 0, 1)
    tpr_lower = tpr_mean - tpr_std
    axs.plot(fpr_mean, tpr_mean, lw=2, 
                label=f"{method} AUC = {np.round(np.mean(results['auc']), 2)} ± {np.round(np.std(results['auc']), 2)}")
    axs.fill_between(fpr_mean, tpr_lower, tpr_upper, alpha=.2)
    axs.legend(loc="lower right")

    return axs

def plot_roc(y_pred_proba, y_test, unique_labels, name, binary, acc, run):
    dirs = '/'.join(name.split('/')[:-1])
    name = name.split('/')[-1]
    method = name.split('_')[1]
    group = name.split('_')[2]
    os.makedirs(f'{dirs}', exist_ok=True)
    fig, axs = plt.subplots(1, figsize=(8, 8), dpi=110)

    if binary:

        fpr = []
        tpr = []
        auc = []
        real_values = []
        predictions = []
        
        for i in range(len(y_pred_proba)):
            these_real_values = np.asarray(y_test[i])
            these_predictions = y_pred_proba[i][:, 1]

            # fpr, tpr, thresholds = roc_curve(these_real_values, np.asarray(raw_predictions[v]))
            this_fpr, this_tpr, thresholds = roc_curve(these_real_values, 
                                                       these_predictions, 
                                                       drop_intermediate=False
                                                       )

            fpr += [this_fpr.tolist()]
            tpr += [this_tpr.tolist()]
            auc += [metrics.auc(fpr[-1], tpr[-1])]
            real_values.extend(these_real_values)
            predictions.extend(these_predictions)

    else:
        # micro-averaged ROC curve
        fpr = [[] for _ in range(len(y_pred_proba))]
        tpr = [[] for _ in range(len(y_pred_proba))]
        auc = [[] for _ in range(len(y_pred_proba))]
        real_values = []
        predictions = []
        for i in range(len(y_pred_proba)):
            for j in range(y_pred_proba[0].shape[1]):
                classes = np.arange(len(unique_labels))
                these_real_values = label_binarize(np.asarray(y_test[i]), classes=classes)[:, j]
                these_predictions = y_pred_proba[i][:, j]

                # fpr, tpr, thresholds = roc_curve(these_real_values, np.asarray(raw_predictions[v]))
                this_fpr, this_tpr, thresholds = roc_curve(these_real_values,
                                                           these_predictions,
                                                           drop_intermediate=False
                                                           )

                fpr[i] += [np.array(this_fpr.tolist())]
                tpr[i] += [np.array(this_tpr.tolist())]
                auc[i] += [np.array(metrics.auc(this_fpr, this_tpr))]
                real_values.extend(these_real_values)
                predictions.extend(these_predictions)
            fpr[i] = np.concatenate(fpr[i])
            tpr[i] = np.concatenate(tpr[i])
            # auc[i] = np.mean(auc[i])

    axs = plot_intervals({'fpr': fpr, 'tpr': tpr, 'auc': auc}, axs, method)

    fig.text(0.5, 0.04, '1-Specificity', ha='center', va='center')
    fig.text(0.08, 0.5, "Sensitivity", ha='center', va='center', rotation='vertical')
    # Add line for auc=0.5
    plt.plot([0, 1], [0, 1], 'k--')
    plt.savefig(f'{dirs}/ROC_{group}.png')
    run[f'roc/{group}'].upload(fig)
    run[f'auc/{group}'] = np.mean(auc)
    plt.close()

    fig, axs = plt.subplots(1, figsize=(8, 8), dpi=110)
    fraction_of_positives, mean_predicted_value = \
        calibration_curve(real_values, predictions, n_bins=20)

    plt.plot(mean_predicted_value, fraction_of_positives, "s-",
                label="%s (%1.3f)" % (name, np.mean(acc)))
    # add perfect calibration line
    plt.plot([0, 1], [0, 1], "--", color='k', label='Perfect Calibration')
    fig.text(0.5, 0.04, 'Mean predicted probability', ha='center', va='center')
    fig.text(0.08, 0.5, "Fraction of positives", ha='center', va='center', rotation='vertical')
    plt.title('Calibration plot')
    plt.savefig(f'{dirs}/calib_{group}.png')
    run[f'calib/{group}'].upload(fig)
    plt.close()

    fig, axs = plt.subplots(1, figsize=(8, 8), dpi=110)
    plt.hist(predictions, range=(0, 1), bins=10, label=group,
                histtype="step", lw=2)
    fig.text(0.5, 0.04, 'Mean predicted probability', ha='center', va='center')
    fig.text(0.08, 0.5, "Count", ha='center', va='center', rotation='vertical')
    plt.title(name)
    plt.savefig(f'{dirs}/calib_hist_{group}.png')
    run[f'calib_hist_/{group}'].upload(fig)
    plt.close()
