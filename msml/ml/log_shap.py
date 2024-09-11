import os
import torch
import shap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
import matplotlib.cm as cm
from PIL import Image
import xgboost

def interactions_mean_matrix(shap_interactions, run, group):
    # Get absolute mean of matrices
    mean_shap = np.abs(shap_interactions).mean(0)
    df = pd.DataFrame(mean_shap, index=X.columns, columns=X.columns)

    # times off diagonal by 2
    df.where(df.values == np.diagonal(df), df.values * 2, inplace=True)

    # display
    plt.figure(figsize=(10, 10), facecolor='w', edgecolor='k')
    sns.set(font_scale=1.5)
    sns.heatmap(df, cmap='coolwarm', annot=True, fmt='.3g', cbar=False)
    plt.yticks(rotation=0)
    f = plt.gcf()
    run[f'shap/interactions_matrix/{group}_values'].upload(f)
    plt.close(f)


def make_summary_plot(df, values, group, run, 
                      log_path, category='explainer',
                      mlops='neptune'):
    shap.summary_plot(values, df, show=False)
    f = plt.gcf()
    if mlops == 'neptune':
        run[f'shap/summary_{category}/{group}_values'].upload(f)
    if mlops == 'mlflow':
        os.makedirs(f'{log_path}/shap/summary_{category}', exist_ok=True)
        plt.savefig(f'{log_path}/shap/summary_{category}/{group}_values.png')
        mlflow.log_figure(f, f'{log_path}/shap/summary_{category}/{group}_values.png')

    plt.close(f)


def make_force_plot(df, values, features, group, run, log_path, category='explainer', mlops='mlflow'):
    shap.force_plot(df, values, features=features, show=False)
    f = plt.gcf()
    if mlops == 'neptune':
        run[f'shap/force_{category}/{group}_values'].upload(f)
    if mlops == 'mlflow':
        os.makedirs(f'{log_path}/shap/force_{category}', exist_ok=True)
        plt.savefig(f'{log_path}/shap/force_{category}/{group}_values.png')
        mlflow.log_figure(f, f'{log_path}/shap/force_{category}/{group}_values.png')

    plt.close(f)


def make_deep_beeswarm(df, values, group, run, log_path, category='explainer', mlops='mlflow'):
    shap.summary_plot(values, feature_names=df.columns, features=df, show=False)
    f = plt.gcf()
    if mlops == 'neptune':
        run[f'shap/beeswarm_{category}/{group}_values'].upload(f)
    if mlops == 'mlflow':
        os.makedirs(f'{log_path}/shap/beeswarm_{category}', exist_ok=True)
        plt.savefig(f'{log_path}/shap/beeswarm_{category}/{group}_values.png')
        mlflow.log_figure(f, f'{log_path}/shap/beeswarm_{category}/{group}_values.png')

    plt.close(f)


def make_decision_plot(df, values, misclassified, feature_names, group, run, log_path, category='explainer', mlops='mlflow'):
    shap.decision_plot(df, values, feature_names=list(feature_names), show=False, link='logit', highlight=misclassified)
    f = plt.gcf()
    if mlops == 'neptune':
        run[f'shap/decision_{category}/{group}_values'].upload(f)
        run[f'shap/decision_{category}/{group}_values'].upload(f)
    if mlops == 'mlflow':
        os.makedirs(f'{log_path}/shap/decision_{category}', exist_ok=True)
        plt.savefig(f'{log_path}/shap/decision_{category}/{group}_values.png')
        mlflow.log_figure(f, f'{log_path}/shap/decision_{category}/{group}_values.png')
    plt.close(f)


def make_decision_deep(df, values, misclassified, feature_names, group, run, log_path, category='explainer', mlops='mlflow'):
    shap.decision_plot(df, values, feature_names=list(feature_names), show=False, link='logit', highlight=misclassified)
    f = plt.gcf()
    if mlops == 'neptune':
        run[f'shap/decision_{category}/{group}_values'].upload(f)
        run[f'shap/decision_{category}/{group}_values'].upload(f)
    if mlops == 'mlflow':
        os.makedirs(f'{log_path}/shap/decision_{category}', exist_ok=True)
        plt.savefig(f'{log_path}/shap/decision_{category}/{group}_values.png')
        mlflow.log_figure(f, f'{log_path}/shap/decision_{category}/{group}_values.png')
    plt.close(f)


def make_multioutput_decision_plot(df, values, group, run, log_path, category='explainer', mlops='mlflow'):
    shap.multioutput_decision_plot(values, df, show=False)
    f = plt.gcf()
    if mlops == 'neptune':
        run[f'shap/multioutput_decision_{category}/{group}_values'].upload(f)
    if mlops == 'mlflow':
        os.makedirs(f'{log_path}/shap/multioutput_decision_{category}', exist_ok=True)
        plt.savefig(f'{log_path}/shap/multioutput_decision_{category}/{group}_values.png')
        mlflow.log_figure(f, f'{log_path}/shap/multioutput_decision_{category}/{group}_values.png')
    plt.close(f)


def make_group_difference_plot(values, mask, group, run, log_path, category='explainer', mlops='mlflow'):
    shap.group_difference_plot(values, mask, show=False)
    f = plt.gcf()
    if mlops == 'neptune':
        # run[f'shap/gdiff_{category}/{group}'].upload(f)
        run[f'shap/gdiff_{category}/{group}'].upload(f)
    if mlops == 'mlflow':
        os.makedirs(f'{log_path}/shap/gdiff_{category}', exist_ok=True)
        plt.savefig(f'{log_path}/shap/gdiff_{category}/{group}_values.png')
        mlflow.log_figure(f, f'{log_path}/shap/gdiff_{category}/{group}_values.png')
    plt.close(f)


def make_beeswarm_plot(values, group, run, log_path, category='explainer', mlops='mlflow'):
    shap.plots.beeswarm(values, max_display=20, show=False)
    f = plt.gcf()
    if mlops == 'neptune':
        # run[f'shap/beeswarm_{category}/{group}'].upload(f)
        run[f'shap/beeswarm_{category}/{group}'].upload(f)
    if mlops == 'mlflow':
        os.makedirs(f'{log_path}/shap/beeswarm_{category}', exist_ok=True)
        plt.savefig(f'{log_path}/shap/beeswarm_{category}/{group}_values.png')
        mlflow.log_figure(f, f'{log_path}/shap/beeswarm_{category}/{group}_values.png')
    plt.close(f)


def make_heatmap(values, group, run, log_path, category='explainer', mlops='mlflow'):
    shap.plots.heatmap(values, instance_order=values.values.sum(1).argsort(), max_display=20, show=False)
    f = plt.gcf()
    if mlops == 'neptune':
        # run[f'shap/heatmap_{category}/{group}'].upload(f)
        run[f'shap/heatmap_{category}/{group}'].upload(f)
    if mlops == 'mlflow':
        os.makedirs(f'{log_path}/shap/heatmap_{category}', exist_ok=True)
        plt.savefig(f'{log_path}/shap/heatmap_{category}/{group}_values.png')
        mlflow.log_figure(f, f'{log_path}/shap/heatmap_{category}/{group}_values.png')
    plt.close(f)


def make_heatmap_deep(values, group, run, log_path, category='explainer', mlops='mlflow'):

    shap.plots.heatmap(pd.DataFrame(values), instance_order=values.sum(1).argsort(), max_display=20, show=False)
    f = plt.gcf()
    if mlops == 'neptune':
        # run[f'shap/heatmap_{category}/{group}'].upload(f)
        run[f'shap/heatmap_{category}/{group}'].upload(f)
    if mlops == 'mlflow':
        os.makedirs(f'{log_path}/shap/heatmap_{category}', exist_ok=True)
        plt.savefig(f'{log_path}/shap/heatmap_{category}/{group}_values.png')
        mlflow.log_figure(f, f'{log_path}/shap/heatmap_{category}/{group}_values.png')
    plt.close(f)


def make_barplot(df, y, values, group, run, log_path, category='explainer', mlops='mlflow'):
    clustering = shap.utils.hclust(df, y, metric='correlation')  # cluster_threshold=0.9
    # shap.plots.bar(values, max_display=20, show=False, clustering=clustering)
    shap.plots.bar(values, max_display=20, show=False, clustering=clustering, clustering_cutoff=0.5)
    f = plt.gcf()
    if mlops == 'neptune':
        run[f'shap/bar_{category}/{group}'].upload(f)
    if mlops == 'mlflow':
        os.makedirs(f'{log_path}/shap/bar_{category}', exist_ok=True)
        plt.savefig(f'{log_path}/shap/bar_{category}/{group}_values.png')
        mlflow.log_figure(f, f'{log_path}/shap/bar_{category}/{group}_values.png')
    plt.close(f)


def make_bar_plot(df, values, group, run, log_path, category='explainer', mlops='mlflow'):
    shap.bar_plot(values, max_display=40, feature_names=df.columns, show=False)
    f = plt.gcf()
    if mlops == 'neptune':
        # run[f'shap/barold_{category}/{group}'].upload(f)
        run[f'shap/barold_{category}/{group}'].upload(f)
    if mlops == 'mlflow':
        os.makedirs(f'{log_path}/shap/barold_{category}', exist_ok=True)
        plt.savefig(f'{log_path}/shap/barold_{category}/{group}_values.png')
        mlflow.log_figure(f, f'{log_path}/shap/barold_{category}/{group}_values.png')
    plt.close(f)

def make_dependence_plot(df, values, var, group, run, log_path, category='explainer', mlops='mlflow'):
    shap.dependence_plot(var, values[1], df, show=False)
    f = plt.gcf()
    if mlops == 'neptune':
        run[f'shap/dependence_{category}/{group}'].upload(f)
    if mlops == 'mlflow':
        os.makedirs(f'{log_path}/shap/dependence_{category}', exist_ok=True)
        plt.savefig(f'{log_path}/shap/dependence_{category}/{group}_values.png')
        mlflow.log_figure(f, f'{log_path}/shap/dependence_{category}/{group}_values.png')
    plt.close(f)

def make_images_shap(bins, shap_values, label, run, log_path):
    min_parent_mz = np.unique(np.array([float(x.split('_')[0]) for x in list(shap_values.index)]))
    max_min_parent_mz = np.max(min_parent_mz)
    min_min_parent_mz = np.min(min_parent_mz)
    interval = min_parent_mz[1] - min_parent_mz[0]
    final = {min_parent: pd.DataFrame(
        np.zeros([int(np.ceil(bins['mz_max'] / bins['mz_bin'])) + 1,
                    int(np.ceil(bins['rt_max'] / bins['rt_bin'])) + 1]),
        dtype=np.float32,
        columns=np.arange(0, bins['rt_max'] + bins['rt_bin'], bins['rt_bin']).round(
            bins['rt_rounding']),
        index=np.arange(0, bins['mz_max'] + bins['mz_bin'], bins['mz_bin']).round(
            bins['mz_rounding'])
    ) for min_parent in
                np.arange(int(min_min_parent_mz), int(max_min_parent_mz) + interval, interval)}
    for i in list(final.keys()):
        final[i].index = np.round(final[i].index, bins['mz_rounding'])
        final[i].columns = np.round(final[i].columns, bins['rt_rounding'])
    min_parent = list(final.keys())[0]
    rt = float(final[min_parent][list(final[min_parent].keys())[0]][0])
    # spdtypes = final[min_parent].dtypes[rt]
    # prev_mz = -1
    for i, intensity in enumerate(shap_values):
        min_parent = float(shap_values.index[i].split('_')[0])
        rt = float(shap_values.index[i].split('_')[1])
        mz = float(shap_values.index[i].split('_')[2])
        if mz < 80:
            mz = mz

        rt = np.round(np.round(rt / bins['rt_bin'], 8), bins['rt_bin']) * bins['rt_bin']
        mz = np.round(np.round(mz / bins['mz_bin'], 8), bins['mz_bin']) * bins['mz_bin']
        if bins['rt_rounding'] != 0:
            rt = np.round(rt, bins['rt_rounding'])
        if bins['mz_rounding'] != 0:
            mz = np.round(mz, bins['mz_rounding'])
        mz = np.round(float(mz), bins['mz_rounding'])
        rt = np.round(float(rt), bins['rt_rounding'])
        
        final[min_parent].loc[mz].loc[rt] += intensity
    for i, min_parent in enumerate(list(final.keys())):
        df = np.stack(list(final[min_parent].values))
        df = df / df.max()
        df = pd.DataFrame(df, columns=final[min_parent].columns, index=final[min_parent].index)
        save_images_and_csv3d(log_path, df, label, min_parent)
        if i == 10 and label == 0:
            run[f'shap/images/{min_parent}'].upload(f"{log_path}/shap_images3d/{min_parent}/{label}.png")
        # run[f'shap/images/{min_parent}'].upload(f"{log_path}/shap_images3d//{min_parent}/{label}.csv")

def save_images_and_csv3d(path, final, label, min_parent):
    os.makedirs(f"{path}/shap_csv3d/{min_parent}/", exist_ok=True)
    os.makedirs(f"{path}/shap_images3d/{min_parent}/", exist_ok=True)
    final.to_csv(f"{path}/shap_csv3d/{min_parent}/{label}.csv", index_label='ID')
    im = Image.fromarray(np.uint8(cm.gist_earth(final) * 255))
    im.save(f"{path}/shap_images3d/{min_parent}/{label}.png")
    im.close()
    del im


def log_explainer(model, x_df, labels, group, run, bins, log_path):
    unique_labels = np.unique(labels)
    # The explainer doesn't like tensors, hence the f function
    f = lambda x: model.predict(x)
    try:
        explainer = shap.LinearExplainer(model, x_df, 
                                     max_evals=2 * x_df.shape[1] + 1)
    except:
        explainer = shap.TreeExplainer(model)
        # model.set_param({"device": "cuda:0"})
        # dtrain = xgboost.DMatrix(x_df, label=labels)
        # shap_values = model.predict(dtrain, pred_contribs=True)
        # shap_interaction_values = model.predict(dtrain, pred_interactions=True)
    # except:
    #     explainer = shap.LinearExplainer(model.estimators_[0], x_df, max_evals=2 * x_df.shape[1] + 1)

    # Get the shap values from my test data
    shap_values = explainer(x_df)
    if len(unique_labels) == 2:
        shap_values_df = pd.DataFrame(
            np.c_[shap_values.base_values, shap_values.values], 
            columns=['bv'] + list(x_df.columns)
        )
        # Remove shap values that are 0
        shap_values_df = shap_values_df.loc[:, (shap_values_df != 0).any(axis=0)]
        shap_values_df = shap_values_df.abs()
        shap_values_df = shap_values_df.sum(0)
        total = shap_values_df.sum()
        shap_values_df = shap_values_df / total
        try:
            # Getting the base value
            bv = shap_values_df['bv']
            label = labels[0]

            # Dropping the base value
            shap_values_df = shap_values_df.drop('bv')
            make_images_shap(bins, shap_values_df, label, run, log_path=log_path)
            shap_values_df.to_csv(f"{log_path}/{group}_linear_shap_{label}_abs.csv")
            run[f'shap/linear_{group}_{label}'].upload(f"{log_path}/{group}_linear_shap_{label}_abs.csv")

            shap_values_df.transpose().hist(bins=100, figsize=(10, 10))
            plt.savefig(f"{log_path}/{group}_linear_shap_{label}_hist_abs.png")
            plt.close()
            plt.title(f'base_value: {np.round(bv, 2)}')
            run[f'shap/linear_{group}_{label}_hist'].upload(f"{log_path}/{group}_linear_shap_{label}_hist_abs.png")
            # start x axis at 0
            shap_values_df.abs().sort_values(ascending=False).plot(kind='kde', figsize=(10, 10))
            # shap_values_df.transpose().cumsum().hist(bins=100, figsize=(10, 10))
            plt.xlim(0, shap_values_df.abs().max())
            plt.savefig(f"{log_path}/{group}_linear_shap_{label}_kde_abs.png")
            plt.close()
            plt.title(f'base_value: {np.round(bv, 2)}')
            run[f'shap/linear_{group}_{label}_kde'].upload(f"{log_path}/{group}_linear_shap_{label}_kde_abs.png")
            
            values, base = np.histogram(shap_values_df.abs(), bins=40)
            #evaluate the cumulative
            cumulative = np.cumsum(values)
            # plot the cumulative function
            plt.plot(base[:-1], cumulative, c='blue')
            #plot the survival function
            plt.plot(base[:-1], len(shap_values_df.abs())-cumulative, c='green')

            plt.savefig(f"{log_path}/{group}_linear_shap_{label}_cumulative_abs.png")
            plt.close()
            plt.title(f'base_value: {np.round(bv, 2)}')
            run[f'shap/linear_{group}_{label}_cumulative_abs'].upload(f"{log_path}/{group}_linear_shap_{label}_cumulative_abs.png")

        except:
            pass

    else:
        # save shap_values
        # TODO Verifier que l'ordre est bon
        for i, label in enumerate(unique_labels):
            shap_values_df = pd.DataFrame(
                np.c_[shap_values.base_values[:, i], shap_values.values[:, :, i]], 
                columns=['bv'] + list(x_df.columns)
            )
            # Remove shap values that are 0
            shap_values_df = shap_values_df.loc[:, (shap_values_df != 0).any(axis=0)]
            shap_values_df = shap_values_df.abs()
            shap_values_df = shap_values_df.sum(0)
            total = shap_values_df.sum()
            shap_values_df = shap_values_df / total
            try:
                # Getting the base value
                bv = shap_values_df['bv']

                # Dropping the base value
                shap_values_df = shap_values_df.drop('bv')
                make_images_shap(bins, shap_values_df, label, run, log_path=log_path)
                shap_values_df.to_csv(f"{log_path}/{group}_linear_shap_{label}_abs.csv")
                run[f'shap/linear_{group}_{label}'].upload(f"{log_path}/{group}_linear_shap_{label}_abs.csv")

                shap_values_df.transpose().hist(bins=100, figsize=(10, 10))
                plt.savefig(f"{log_path}/{group}_linear_shap_{label}_hist_abs.png")
                plt.close()
                plt.title(f'base_value: {np.round(bv, 2)}')
                if i == 0:
                    run[f'shap/linear_{group}_{label}_hist'].upload(f"{log_path}/{group}_linear_shap_{label}_hist_abs.png")
                # start x axis at 0
                shap_values_df.abs().sort_values(ascending=False).plot(kind='kde', figsize=(10, 10))
                # shap_values_df.transpose().cumsum().hist(bins=100, figsize=(10, 10))
                plt.xlim(0, shap_values_df.abs().max())
                plt.savefig(f"{log_path}/{group}_linear_shap_{label}_kde_abs.png")
                plt.close()
                plt.title(f'base_value: {np.round(bv, 2)}')
                if i == 0:
                    run[f'shap/linear_{group}_{label}_kde'].upload(f"{log_path}/{group}_linear_shap_{label}_kde_abs.png")
                
                values, base = np.histogram(shap_values_df.abs(), bins=40)
                #evaluate the cumulative
                cumulative = np.cumsum(values)
                # plot the cumulative function
                plt.plot(base[:-1], cumulative, c='blue')
                #plot the survival function
                plt.plot(base[:-1], len(shap_values_df.abs())-cumulative, c='green')

                plt.savefig(f"{log_path}/{group}_linear_shap_{label}_cumulative_abs.png")
                plt.close()
                plt.title(f'base_value: {np.round(bv, 2)}')
                if i == 0:
                    run[f'shap/linear_{group}_{label}_cumulative_abs'].upload(f"{log_path}/{group}_linear_shap_{label}_cumulative_abs.png")

            except:
                pass

    # if x_df.shape[1] <= 1000:
    #     make_barplot(x_df, labels, shap_values[:, :, 0], 
    #                 group, run, 'LinearExplainer', mlops='neptune')
    #     # Summary plot
    #     make_summary_plot(x_df, shap_values[:, :, 0], group, run, 
    #                     'LinearExplainer', mlops='neptune')
    #     make_beeswarm_plot(shap_values[:, :, 0], group, run,
    #                         'LinearExplainer', mlops='neptune')
    #     make_heatmap(shap_values[:, :, 0], group, run,
    #                 'LinearExplainer', mlops='neptune')
    #     # mask = np.array([np.argwhere(x[0] == 1)[0][0] for x in cats])
    #     # make_group_difference_plot(x_df.sum(1).to_numpy(), mask, group, run, 'LinearExplainer', mlops='neptune')
    #     make_bar_plot(x_df, shap_values[0], group, run,
    #                 'LinearExplainer', mlops='neptune')
    #     make_force_plot(x_df, shap_values[0], x_df.columns, 
    #                     group, run, 'LinearExplainer', mlops='neptune')


def log_kernel_explainer(model, x_df, misclassified, 
                         labels, group, run, cats, log_path):
    unique_labels = np.unique(labels)

    # Convert my pandas dataframe to numpy
    data = x_df.to_numpy(dtype=np.float32)
    data = shap.kmeans(data, 20).data
    # The explainer doesn't like tensors, hence the f function
    explainer = shap.KernelExplainer(model.predict, data)

    # Get the shap values from my test data
    df = pd.DataFrame(data, columns=x_df.columns)
    shap_values = explainer.shap_values(df)
    # shap_interaction = explainer.shap_interaction_values(X_test)
    shap_values_df = pd.DataFrame(np.concatenate(shap_values), columns=x_df.columns)
    for i, label in enumerate(unique_labels):
        if i == len(shap_values):
            break
        shap_values_df.iloc[i].to_csv(f"{log_path}/{group}_kernel_shap_{label}.csv")
    # shap_values = pd.DataFrame(np.concatenate(s))
    # Summary plot
    make_summary_plot(x_df, shap_values, group, run, 'Kernel')

    make_bar_plot(x_df, shap_values_df.iloc[1], group, run, 'localKernel')

    make_decision_plot(explainer.expected_value[0], shap_values[0], misclassified, x_df.columns, group, run, 'Kernel')

    mask = np.array([np.argwhere(x[0] == 1)[0][0] for x in cats])
    make_group_difference_plot(x_df.sum(1).to_numpy(), mask, group, run, 'Kernel')


def log_shap(run, ae, best_lists, cols, bins, log_path):
    # explain all the predictions in the test set
    # explainer = shap.KernelExplainer(svc_linear.predict_proba, X_train[:100])
    os.makedirs(log_path, exist_ok=True)
    for group in ['valid', 'test']:
        if group not in best_lists:
            continue
        X = best_lists[group]['inputs']
        X_test_df = pd.DataFrame(X, columns=list(cols))

        # explainer = shap.DeepExplainer(ae, X_test)
        # explanation = shap.Explanation(X_test, feature_names=X_test_df.columns)
        # explanation.values = explanation.values.detach().cpu().numpy()
        misclassified = [pred != label for pred, label in zip(best_lists[group]['preds'],
                                                              best_lists[group]['labels'])]
        # log_deep_explainer(ae, X_test_df, misclassified, np.concatenate(best_lists[group]['labels']),
        #                    group, run, best_lists[group]['cats'], log_path, mlops, device
        #                    )
        # TODO Problem with not enough memory...
        try:
            log_explainer(ae, X_test_df, best_lists[group]['labels'],
                    group, run, bins, log_path)
        except:
            pass
        # log_kernel_explainer(ae, X_test_df, misclassified,
        #                         best_lists[group]['labels'], group, run, 
        #                         best_lists[group]['labels'], log_path
        #                         )

