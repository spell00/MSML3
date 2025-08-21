# %% [markdown]
# # Ordination Analysis
# 
# This notebook visualizes the data using different ordination techniques, with options to use either all features or filtered features from XGBoost/SHAP.

# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pickle
import json

# %% [markdown]
# ## Load Data and Model Results

# %%
def load_experiment_data(exp_path):
    """Load experiment data and results."""
    # Load data
    with open(os.path.join(exp_path, 'data.pkl'), 'rb') as f:
        data = pickle.load(f)
    
    # Load uniques
    with open(os.path.join(exp_path, 'uniques.pkl'), 'rb') as f:
        uniques = pickle.load(f)
    
    # Load feature importances
    xgb_importance = pd.read_csv(os.path.join(exp_path, 'ords_filtered/xgboost_feature_importance.csv'))
    shap_importance = pd.read_csv(os.path.join(exp_path, 'ords_filtered/shap_feature_importance.csv'))
    
    return data, uniques, xgb_importance, shap_importance

# %% [markdown]
# ## Ordination Functions

# %%
def plot_ordination(X, labels, batches, title, feature_type='all'):
    """Plot PCA and LDA ordinations."""
    # PCA
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X)
    pcs_df = pd.DataFrame(data=pcs, columns=['PC1', 'PC2'])
    pcs_df['label'] = labels
    pcs_df['batch'] = batches
    
    # LDA
    lda = LDA(n_components=2)
    lds = lda.fit_transform(X, labels)
    lds_df = pd.DataFrame(data=lds, columns=['LD1', 'LD2'])
    lds_df['label'] = labels
    lds_df['batch'] = batches
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    
    # PCA by label
    sns.scatterplot(data=pcs_df, x='PC1', y='PC2', hue='label', ax=axes[0,0])
    axes[0,0].set_title(f'PCA by Label ({feature_type} features)\nExplained variance: {pca.explained_variance_ratio_.sum():.2%}')
    
    # PCA by batch
    sns.scatterplot(data=pcs_df, x='PC1', y='PC2', hue='batch', ax=axes[0,1])
    axes[0,1].set_title(f'PCA by Batch ({feature_type} features)')
    
    # LDA by label
    sns.scatterplot(data=lds_df, x='LD1', y='LD2', hue='label', ax=axes[1,0])
    axes[1,0].set_title(f'LDA by Label ({feature_type} features)')
    
    # LDA by batch
    sns.scatterplot(data=lds_df, x='LD1', y='LD2', hue='batch', ax=axes[1,1])
    axes[1,1].set_title(f'LDA by Batch ({feature_type} features)')
    
    plt.tight_layout()
    return fig

# %% [markdown]
# ## Visualize Different Feature Sets

# %%
def visualize_all_ordinations(data, uniques, xgb_importance, shap_importance, threshold=0):
    """Create ordination plots for all features, XGBoost features, and SHAP features."""
    # All features
    fig_all = plot_ordination(
        data['inputs']['all'],
        data['labels']['all'],
        data['batches']['all'],
        'All Features'
    )
    
    # XGBoost features
    xgb_features = xgb_importance[xgb_importance['importance'] > threshold]['feature'].tolist()
    fig_xgb = plot_ordination(
        data['inputs']['all'][xgb_features],
        data['labels']['all'],
        data['batches']['all'],
        'XGBoost Features'
    )
    
    # SHAP features
    shap_features = shap_importance[shap_importance['shap_importance'] > threshold]['feature'].tolist()
    fig_shap = plot_ordination(
        data['inputs']['all'][shap_features],
        data['labels']['all'],
        data['batches']['all'],
        'SHAP Features'
    )
    
    return fig_all, fig_xgb, fig_shap

# %% [markdown]
# ## Example Usage

# %%
# Load data
exp_path = 'results/multi/mz10/rt10/ms2/200spd/thr0.0/all/b14-b13-b12-b11-b10-b9-b8-b7-b6-b5-b4-b3-b2-b1_binary0_-1_gkf0_ovr0_mz0-10000rt0-320_na_h/xgboost'
data, uniques, xgb_importance, shap_importance = load_experiment_data(exp_path)

# Create visualizations
fig_all, fig_xgb, fig_shap = visualize_all_ordinations(data, uniques, xgb_importance, shap_importance)

# Save figures
fig_all.savefig('ordinations_all_features.png')
fig_xgb.savefig('ordinations_xgboost_features.png')
fig_shap.savefig('ordinations_shap_features.png') 