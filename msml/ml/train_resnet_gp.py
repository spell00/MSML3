#!/usr/bin/python3
NEPTUNE_API_TOKEN = "YOUR-API-KEY"
NEPTUNE_PROJECT_NAME = "YOUR-PROJECT-NAME"
NEPTUNE_MODEL_NAME = "YOUR-MODEL-NAME"

import matplotlib
# from bernn.utils.pool_metrics import log_pool_metrics

matplotlib.use('Agg')
CUDA_VISIBLE_DEVICES = ""

import uuid
import shutil

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import copy
import torch
from torch import nn
import os

from sklearn import metrics
from tensorboardX import SummaryWriter
from ax.service.managed_loop import optimize
from sklearn.metrics import matthews_corrcoef as MCC
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
# from bernn.ml.train.params_gp import *
from models.data_getters import get_data, get_bacteria_images_ms2, split_images_ms2
from torchvision.models import resnet18, ResNet18_Weights
# from aedacnn import SHAPAutoEncoder2 as SHAPAutoEncoder
from models.loggings import TensorboardLoggingAE, log_metrics, log_input_ordination, \
    LogConfusionMatrix, log_plots, log_neptune, log_shap, log_mlflow, make_data
from models.dataset import get_images_loaders, get_images_loaders_no_pool
from models.utils import get_optimizer, to_categorical, get_empty_dicts, get_empty_traces, \
    log_traces, get_best_values, add_to_logger, add_to_neptune, \
    add_to_mlflow
import mlflow
import warnings
from datetime import datetime
# import params_gp
from torch.nn import functional as F
import math
from utils import scale_data
from torch.autograd import Function


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Initialize dimensions
        self.d_model = d_model # Model's dimension
        self.num_heads = num_heads # Number of attention heads
        self.d_k = d_model // num_heads # Dimension of each head's key, query, and value
        
        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model) # Query transformation
        self.W_k = nn.Linear(d_model, d_model) # Key transformation
        self.W_v = nn.Linear(d_model, d_model) # Value transformation
        self.W_o = nn.Linear(d_model, d_model) # Output transformation
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output

class CausalSelfAttention(nn.Module):

    def __init__(self, num_heads: int, embed_dimension: int, bias: bool=False, is_causal: bool=False, dropout:float=0.0):
        super().__init__()
        assert embed_dimension % num_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(embed_dimension, 3 * embed_dimension, bias=bias)
        # output projection
        self.c_proj = nn.Linear(embed_dimension, embed_dimension, bias=bias)
        # regularization
        self.dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.embed_dimension = embed_dimension
        # Perform causal masking
        self.is_causal = is_causal

    def forward(self, x):
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        query_projected = self.c_attn(x)

        batch_size = query_projected.size(0)
        embed_dim = query_projected.size(2)
        head_dim = embed_dim // (self.num_heads * 3)

        query, key, value = query_projected.chunk(3, -1)
        query = query.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)

        if self.training:
            dropout = self.dropout
            is_causal = self.is_causal
        else:
            dropout = 0.0
            is_causal = False

        y = F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=dropout, is_causal=is_causal)
        y = y.transpose(1, 2).view(batch_size, -1, self.num_heads * head_dim)

        y = self.resid_dropout(self.c_proj(y))
        return y

# Define a simple Transformer model
class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, dropout):
        super(SimpleTransformer, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Linear(input_dim, embed_dim)
        
        # Transformer layers
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='relu'
        )
        
        # Output layer
        # self.classifier = nn.Linear(embed_dim * seq_length, num_classes)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        batch_size, seq_length, _ = x.shape
        
        # Embedding
        x = self.embedding(x)  # Shape: (batch_size, seq_length, embed_dim)
        
        # Transformer expects input shape: (seq_length, batch_size, embed_dim)
        x = x.permute(1, 0, 2)
        
        # Passing through transformer
        x = self.transformer(x, x)
        
        # Flatten and pass through the output layer
        x = x.permute(1, 0, 2).contiguous().view(batch_size, -1)
        # x = self.fc(x)
        
        return x

class CNN1D(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding, dropout):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=output_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.adaptative_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.adaptative_pool(x)
        return x

class FocalLoss(nn.Module):
    """
    binary focal loss
    """

    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.weight = torch.Tensor([alpha, 1-alpha])
        self.nllLoss = nn.NLLLoss(weight=self.weight)
        self.gamma = gamma

    def forward(self, input, target):
        softmax = F.softmax(input, dim=1)
        log_logits = torch.log(softmax)
        fix_weights = (1 - softmax) ** self.gamma
        logits = fix_weights * log_logits
        return self.nllLoss(logits, target.argmax(1))

class Predictor(nn.Module):

    def __init__(self, model: str, n_classes: int, n_batches: int, params: dict):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        self.classifier = nn.Linear(512, n_classes)
        self.dann_discriminator = nn.Linear(512, n_batches)
        self.transforms = weights.transforms()
        if model == 'resnet18':
            self.model = resnet18(weights=weights, progress=False)
            self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        elif model == 'transformer':
            self.model = MultiHeadAttention(512, 8)
        elif model == 'causal':
            self.transforms = None
            num_heads = 8
            heads_per_dim = 64
            embed_dimension = num_heads * heads_per_dim
            dtype = torch.float16
            self.model = CausalSelfAttention(num_heads=num_heads, embed_dimension=embed_dimension, bias=False, is_causal=True, dropout=0.1).to("cuda").to(dtype).eval()
        elif model == 'simple':
            self.transforms = None
            # Model parameters
            input_dim = 34
            # embed_dim = 128
            # num_heads = 4
            # num_layers = 2
            seq_length = 142

            # Instantiate the model
            self.model = SimpleTransformer(input_dim, params['embed_dim']*params['num_heads'], params['num_heads'], params['num_layers'], params['dropout']).to("cuda")
            self.classifier = nn.Linear(params['embed_dim']*params['num_heads'] * seq_length, n_classes)
            self.dann_discriminator = nn.Linear(params['embed_dim']*params['num_heads'] * seq_length, n_batches)
        elif model == 'cnn1d':
            self.transforms = None
            self.model = CNN1D(34, 64, 3, 1, 1, params['dropout'])
            self.classifier = nn.Linear(64, n_classes)
            self.dann_discriminator = nn.Linear(64, n_batches)



    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.transforms is not None:
            x = self.transforms(input)
        else:
            x = input
        enc = self.model(x)
        # x = self.classifier(x.view(x.size(0), -1))
        return enc.squeeze(-1).squeeze(-1)

# import StratifiedGroupKFold
# from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

# from fastapi import BackgroundTasks, FastAPI
# from threading import Thread

# app = FastAPI()

warnings.filterwarnings("ignore")

random.seed(1)
torch.manual_seed(1)
np.random.seed(1)

def keep_top_features(data, path, args):
    """
    Keeps the top features according to the precalculated scores
    Args:
        data: The data to be used to keep the top features

    Returns:
        data: The data with only the top features
    """
    top_features = pd.read_csv(f'{path}/{args.best_features_file}', sep=',')
    for group in ['all', 'train', 'valid', 'test']:
        data['inputs'][group] = data['inputs'][group].loc[:, top_features.iloc[:, 0].values[:args.n_features]]

    return data

def binarize_labels(data, controls):
    """
    Binarizes the labels to be used in the classification loss
    Args:
        labels: The labels to be binarized
        controls: The control labels

    Returns:
        labels: The binarized labels
    """
    for group in ['all']:
        data['cats'][group] = np.array([1 if x not in controls else 0 for x in data['labels'][group]])
        data['labels'][group] = np.array(['blanc' if x not in controls else 'not' for x in data['labels'][group]])
    return data


class TrainAE:

    def __init__(self, args, fix_thres=-1, load_tb=False, keep_models=True, log_inputs=True,
                 log_tb=False, log_neptune=False, log_mlflow=True):
        """

        Args:
            args: contains multiple arguments passed in the command line
            log_path (str): Path where the tensorboard logs are saved
            path (str): Path to the data (in .csv format)
            fix_thres (float): If 1 > fix_thres >= 0 then the threshold is fixed to that value.
                       any other value means the threshold won't be fixed and will be
                       learned as an hyperparameter
            load_tb (bool): If True, loads previous runs already saved
            keep_models (bool): Whether or not to save the models trained
                                (can take a lot of space if training a lot of models)
            log_inputs (bool): Whether or not to log graphs or batch effect metrics
                                of the scaled inputs
            log_tb (bool): Whether or not to use tensorboard.
            log_mlflow (bool): Wether or not to use mlflow.
        """
        self.hparams_names = None
        self.best_acc = 0
        self.best_mcc = -1
        self.best_closs = np.inf
        self.logged_inputs = False
        self.log_tb = log_tb
        self.log_neptune = log_neptune
        self.log_mlflow = log_mlflow
        self.args = args
        self.log_inputs = log_inputs
        self.keep_models = keep_models
        self.fix_thres = fix_thres
        self.load_tb = load_tb
        self.foldername = None

        self.verbose = 1

        self.n_cats = None
        self.data = None
        self.unique_labels = None
        self.unique_batches = None

        self.pools = False
        if self.args.test:
            self.n_neurons = 5208
        else:
            self.n_neurons = 4704
        
        self.data, self.unique_labels, self.unique_batches = get_bacteria_images_ms2(self.args.path, args)
        if args.model == 'cnn1d':
            self.data['inputs']['all'] = self.data['inputs']['all'].reshape([
                self.data['inputs']['all'].shape[0],
                self.data['inputs']['all'].shape[2],
                self.data['inputs']['all'].shape[1]
            ])

        if self.args.controls != '':
            self.data = binarize_labels(self.data, self.args.controls)

    def make_samples_weights(self):
        """
        Makes the samples weights for the training set. The weights are calculated as the inverse of the frequency of
        each class in the training set.
        Returns:

        """
        self.n_batches = len(self.unique_batches)
        self.class_weights = {
            label: 1 / (len(np.where(label == self.data['labels']['train'])[0]) /
                        self.data['labels']['train'].shape[0])
            for label in self.unique_labels if
            label in self.data['labels']['train'] and label not in ["MCI-AD", 'MCI-other', 'DEM-other', 'NPH']}
        self.unique_unique_labels = list(self.class_weights.keys())
        for group in ['train', 'valid', 'test']:
            inds_to_keep = np.array([i for i, x in enumerate(self.data['labels'][group]) if x in self.unique_labels])
            self.data['inputs'][group] = self.data['inputs'][group][inds_to_keep]
            try:
                self.data['names'][group] = self.data['names'][group].iloc[inds_to_keep]
            except:
                self.data['names'][group] = self.data['names'][group][inds_to_keep]

            self.data['labels'][group] = self.data['labels'][group][inds_to_keep]
            self.data['cats'][group] = self.data['cats'][group][inds_to_keep]
            self.data['batches'][group] = self.data['batches'][group][inds_to_keep]

        self.samples_weights = {
            group: [self.class_weights[label] if label not in ["MCI-AD", 'MCI-other', 'DEM-other', 'NPH'] else 0 for
                    name, label in
                    zip(self.data['names'][group],
                        self.data['labels'][group])] if group == 'train' else [
                1 if label not in ["MCI-AD", 'MCI-other', 'DEM-other', 'NPH'] else 0 for name, label in
                zip(self.data['names'][group], self.data['labels'][group])] for group in
            ['train', 'valid', 'test']}
        self.scaler = None

    def train(self, params):
        """
        Args:
            params: Contains the hyperparameters to be optimized

        Returns:
            best_closs: The best classification loss on the valid set

        """
        start_time = datetime.now()
        # Fixing the hyperparameters that are not optimized
        if args.dloss not in ['revTriplet', 'revDANN', 'DANN',
                              'inverseTriplet', 'normae'] or 'gamma' not in params:
            # gamma = 0 will ensure DANN is not learned
            params['gamma'] = 0
        if 1 > self.fix_thres >= 0:
            # fixes the threshold of 0s tolerated for a feature
            params['thres'] = self.fix_thres
        else:
            params['thres'] = 0

        print(params)

        # Assigns the hyperparameters getting optimized
        smooth = params['smoothing']
        scale = params['scaler']
        self.gamma = params['gamma']
        wd = params['wd']
        lr = params['lr']
        margin = params['margin']
        # dropout = params['dropout']
        scaler_name = params['scaler']

        self.args.scaler = scale
        self.args.disc_b_warmup = params['disc_b_warmup']

        optimizer_type = 'adam'
        metrics = {'pool_metrics': {}}
        # self.log_path is where tensorboard logs are saved
        self.foldername = str(uuid.uuid4())

        self.complete_log_path = f'logs/ae_classifier_holdout/{self.foldername}'
        loggers = {'cm_logger': LogConfusionMatrix(self.complete_log_path)}
        print(f'See results using: tensorboard --logdir={self.complete_log_path} --port=6006')

        hparams_filepath = self.complete_log_path + '/hp'
        os.makedirs(hparams_filepath, exist_ok=True)
        self.args.model_name = 'ae_classifier_holdout'
        if self.log_tb:
            loggers['tb_logging'] = TensorboardLoggingAE(hparams_filepath, params,
                                                         tw=self.args.tied_weights,
                                                         dloss=self.args.dloss,
                                                         tl=0,  # to remove, useless now
                                                         pseudo=self.args.predict_tests,
                                                         berm='none',  # to remove, useless now
                                                         args=self.args)
        else:
            model = None
            run = None

        if self.log_neptune:
            # Create a Neptune run object
            run = neptune.init_run(
                project=NEPTUNE_PROJECT_NAME,
                api_token=NEPTUNE_API_TOKEN,
            )  # your credentials
            model = neptune.init_model_version(
                model=NEPTUNE_MODEL_NAME,
                project=NEPTUNE_PROJECT_NAME,
                api_token=NEPTUNE_API_TOKEN,
                # your credentials
            )
            run["dataset"].track_files(f"{self.args.path}/{self.args.csv_file}")
            run["metadata"].track_files(
                f"{self.args.path}/subjects_experiment_ATN_verified_diagnosis.csv"
            )
            # Track metadata and hyperparameters by assigning them to the run
            model["inputs_type"] = run["inputs_type"] = args.csv_file.split(".csv")[0]
            model["best_unique"] = run["best_unique"] = args.best_features_file.split(".tsv")[0]
            model["tied_weights"] = run["tied_weights"] = args.tied_weights
            model["random_recs"] = run["random_recs"] = args.random_recs
            model["dloss"] = run["dloss"] = args.dloss
            model["predict_tests"] = run["predict_tests"] = args.predict_tests
            model["threshold"] = run["threshold"] = args.threshold
            model["rec_loss_type"] = run["rec_loss_type"] = args.rec_loss
            model["strategy"] = run["strategy"] = args.strategy
            model["bad_batches"] = run["bad_batches"] = args.bad_batches
            model["remove_zeros"] = run["remove_zeros"] = args.remove_zeros
            model["parameters"] = run["parameters"] = params
            model["csv_file"] = run["csv_file"] = args.csv_file
            model["model_name"] = run["model_name"] = 'ae_classifier_holdout'
            model["n_meta"] = run["n_meta"] = args.n_meta
            model["n_emb"] = run["n_emb"] = args.embeddings_meta
            model["groupkfold"] = run["groupkfold"] = args.groupkfold
            model["embeddings_meta"] = run["embeddings_meta"] = args.embeddings_meta
            model["foldername"] = run["foldername"] = self.foldername
            model["use_mapping"] = run["use_mapping"] = args.use_mapping
            model["dataset_name"] = run["dataset_name"] = args.dataset
            model["n_agg"] = run["n_agg"] = args.n_agg
        else:
            model = None
            run = None

        if self.log_mlflow:
            mlflow.set_experiment(
                self.args.exp_id,
            )
            try:
                mlflow.start_run()
            except:
                mlflow.end_run()
                mlflow.start_run()
            mlflow.log_params({
                "inputs_type": args.csv_file.split(".csv")[0],
                "best_unique": args.best_features_file.split(".tsv")[0],
                "tied_weights": args.tied_weights,
                "random_recs": args.random_recs,
                "dloss": args.dloss,
                "predict_tests": args.predict_tests,
                "threshold": args.threshold,
                "rec_loss_type": args.rec_loss,
                "bad_batches": args.bad_batches,
                "remove_zeros": args.remove_zeros,
                "parameters": params,
                "scaler": params['scaler'],
                "csv_file": args.csv_file,
                "model_name": args.model_name,
                "n_meta": args.n_meta,
                "n_emb": args.embeddings_meta,
                "groupkfold": args.groupkfold,
                "foldername": self.foldername,
                "use_mapping": args.use_mapping,
                "dataset_name": args.dataset,
                "n_agg": args.n_agg,
                "num_heads": params['num_heads'],
                "embed_dim": params['embed_dim']*params['num_heads'],
                "num_layers": params['num_layers'],
                "dropout": params['dropout'],
                "lr": params['lr'],
                "wd": params['wd'],
                "margin": params['margin'],
                "gamma": params['gamma'],
                "smoothing": params['smoothing'],
                "disc_b_warmup": params['disc_b_warmup'],
                "model": args.model
            })
        else:
            model = None
            run = None
        seed = 0
        combinations = []
        h = 0
        best_closses = []
        best_mccs = []
        while h < self.args.n_repeats:
            print(f'Rep: {h}')
            epoch = 0
            self.best_loss = np.inf
            self.best_closs = np.inf
            self.best_dom_loss = np.inf
            self.best_dom_acc = np.inf
            self.best_acc = 0
            self.best_mcc = -1
            self.warmup_b_counter = 0
            self.warmup_disc_b = False

            # TODO Should not have to load the data everytime
            # if self.args.dataset == 'bacteria':
            self.data = split_images_ms2(self.data, args, seed=seed)                
            self.data, scaler = scale_data(scaler_name, self.data)
            # else:
            #     self.data, self.unique_labels, self.unique_batches = get_data(self.args.path, args, seed=seed)
            #     self.pools = self.args.pool
            #     self.data = keep_top_features(self.data, self.args.path, self.args)
            if self.args.controls != '':
                self.unique_labels = np.unique(
                    np.concatenate((
                        np.unique(self.data['labels']['train']),
                        np.unique(self.data['labels']['valid']),
                        np.unique(self.data['labels']['test'])
                    ))
                )
                self.unique_batches = np.unique(
                    np.concatenate((
                        np.unique(self.data['batches']['train']),
                        np.unique(self.data['batches']['valid']),
                        np.unique(self.data['batches']['test'])
                    ))
                )

            self.n_cats = len(self.unique_labels)  # + 1  # for pool samples
            if self.args.groupkfold:
                combination = list(np.concatenate((np.unique(self.data['batches']['train']),
                                                np.unique(self.data['batches']['valid']),
                                                np.unique(self.data['batches']['test']))))
                seed += 1
                if combination not in combinations:
                    combinations += [combination]
                else:
                    continue
            h += 1
            self.make_samples_weights()
            # event_acc is used to verify if the hparams have already been tested. If they were,
            # the best classification loss is retrieved and we go to the next trial
            event_acc = EventAccumulator(hparams_filepath)
            event_acc.Reload()
            if len(event_acc.Tags()['tensors']) > 2 and self.load_tb:
                # try:
                #     best_acc = get_best_acc_from_tb(event_acc)
                # except:
                pass
            else:
                # If thres > 0, features that are 0 for a proportion of samples smaller than thres are removed
                # data = self.keep_good_features(thres)

                # Transform the data with the chosen scaler
                # data = copy.deepcopy(self.data)
                # data, self.scaler = scale_data(scale, data, self.args.device)

                # feature_selection = get_feature_selection_method('mutual_info_classif')
                # mi = feature_selection(data['inputs']['train'], data['cats']['train'])
                # for g in list(data['inputs'].keys()):
                #     data['inputs'][g] = data['inputs'][g].round(4)
                # Gets all the pytorch dataloaders to train the models
                # if self.pools:
                # loaders = get_images_loaders(data, self.args.random_recs, self.samples_weights, self.args.dloss, None, None, bs=64)
                # else:
                loaders = get_images_loaders_no_pool(self.data, self.args.random_recs, 
                                                     self.samples_weights, self.args)
                print(self.n_batches, self.n_cats)
                ae = Predictor(
                    model=self.args.model,
                    n_classes=self.n_cats, 
                    n_batches=self.n_batches,
                    params=params
                ).to(self.args.device)
                # if self.args.embeddings_meta > 0:
                #     n_meta = self.n_meta
                # shap_ae = SHAPAutoEncoder(n_classes=self.n_cats, n_batches=self.n_batches).to(self.args.device)
                shap_ae = None
                # shap_ae.mapper.to(self.args.device)
                loggers['logger_cm'] = SummaryWriter(f'{self.complete_log_path}/cm')
                loggers['logger'] = SummaryWriter(f'{self.complete_log_path}/traces')
                sceloss, celoss, mseloss, self.triplet_loss = self.get_losses(scale, smooth, margin, args)
                optimizers = {
                    'ae': get_optimizer(ae, lr, wd, optimizer_type),
                    'b': get_optimizer(ae.dann_discriminator, 1e-2, 0, optimizer_type)
                }
                # optimizer_ae = get_optimizer(ae, lr, wd, optimizer_type)
                # optimizer_b = get_optimizer(ae.dann_discriminator, 1e-2, 0, optimizer_type)
                values, best_values, _, best_traces = get_empty_dicts()

                early_stop_counter = 0
                best_vals = values

                for epoch in range(0, self.args.n_epochs):
                    if early_stop_counter == self.args.early_stop:
                        if self.verbose > 0:
                            print('EARLY STOPPING.', epoch)
                        break
                    lists, traces = get_empty_traces()

                    closs, _, _ = self.loop('train', optimizers, ae, sceloss,
                                            loaders['train'], lists, traces,gamma=self.gamma, nu=1)
            
                    if torch.isnan(closs):
                        print('\n\n\nNAN LOSS\n\n\n')
                        if self.log_mlflow:
                            mlflow.log_param('finished', 0)
                            mlflow.end_run()
                        return self.best_loss
                    ae.eval()

                    # Below is the loop for all sets
                    with torch.no_grad():
                        for group in list(self.data['inputs'].keys()):
                            if group == 'all':
                                continue
                            closs, lists, traces = self.loop(group, optimizers, ae, sceloss,
                                                             loaders[group], lists, traces, nu=0, gamma=0)

                    traces = self.get_mccs(lists, traces)
                    values = log_traces(traces, values)
                    if self.log_tb:
                        try:
                            add_to_logger(values, loggers['logger'], epoch)
                        except:
                            print("Problem with add_to_logger!")
                    if self.log_neptune:
                        add_to_neptune(values, run, epoch)
                    if self.log_mlflow:
                        add_to_mlflow(values, epoch)
                    if np.mean(values['valid']['mcc'][-self.args.n_agg:]) > self.best_mcc and len(
                            values['valid']['mcc']) > self.args.n_agg:
                        print(f"Best Classification Mcc Epoch {epoch}, "
                              f"Acc: {values['test']['acc'][-1]}"
                              f"Mcc: {values['test']['mcc'][-1]}"
                              f"Classification train loss: {values['train']['closs'][-1]},"
                              f" valid loss: {values['valid']['closs'][-1]},"
                              f" test loss: {values['test']['closs'][-1]}")
                        self.best_mcc = np.mean(values['valid']['mcc'][-self.args.n_agg:])
                        torch.save(ae.state_dict(), f'{self.complete_log_path}/model_{h}_state.pth')
                        torch.save(ae, f'{self.complete_log_path}/model_{h}.pth')
                        best_values = get_best_values(values.copy(), ae_only=False, n_agg=self.args.n_agg)
                        best_vals = values.copy()
                        best_vals['rec_loss'] = self.best_loss
                        best_vals['dom_loss'] = self.best_dom_loss
                        best_vals['dom_acc'] = self.best_dom_acc
                        early_stop_counter = 0

                    if values['valid']['acc'][-1] > self.best_acc:
                        print(f"Best Classification Acc Epoch {epoch}, "
                              f"Acc: {values['test']['acc'][-1]}"
                              f"Mcc: {values['test']['mcc'][-1]}"
                              f"Classification train loss: {values['train']['closs'][-1]},"
                              f" valid loss: {values['valid']['closs'][-1]},"
                              f" test loss: {values['test']['closs'][-1]}")

                        self.best_acc = values['valid']['acc'][-1]
                        early_stop_counter = 0

                    if values['valid']['closs'][-1] < self.best_closs:
                        print(f"Best Classification Loss Epoch {epoch}, "
                              f"Acc: {values['test']['acc'][-1]} "
                              f"Mcc: {values['test']['mcc'][-1]} "
                              f"Classification train loss: {values['train']['closs'][-1]}, "
                              f"valid loss: {values['valid']['closs'][-1]}, "
                              f"test loss: {values['test']['closs'][-1]}")
                        self.best_closs = values['valid']['closs'][-1]
                        early_stop_counter = 0
                    else:
                        early_stop_counter += 1

                    if self.args.predict_tests and (epoch % 10 == 0):
                        loaders = get_images_loaders(self.data, data, self.args.random_recs, self.args.triplet_dloss, ae,
                                              ae.classifier, bs=self.args.bs)

                best_mccs += [self.best_mcc]

                best_lists, traces = get_empty_traces()
                # Loading best model that was saved during training
                ae.load_state_dict(torch.load(f'{self.complete_log_path}/model_{h}_state.pth'))
                # Need another model because the other cant be use to get shap values
                # shap_ae.load_state_dict(torch.load(f'{self.complete_log_path}/model_{h}_state.pth'))
                # ae.load_state_dict(sd)
                ae.eval()
                # .eval()
                with torch.no_grad():
                    for group in list(self.data['inputs'].keys()):
                        if group == 'all':
                            continue
                        closs, best_lists, traces = self.loop(group, None, ae, sceloss,
                                                              loaders[group], best_lists,
                                                              traces, nu=0, gamma=0)
                if self.log_neptune:
                    model["model"].upload(f'{self.complete_log_path}/model_{h}_state.pth')
                    model["validation/closs"].log(self.best_closs)
                best_closses += [self.best_closs]
                # logs things in the background. This could be problematic if the logging takes more time than each iteration of repetitive holdout
                # daemon = Thread(target=self.log_rep, daemon=True, name='Monitor',
                #                 args=[best_lists, best_vals, best_values, traces, model, metrics, run, cm_logger, ae,
                #                       shap_ae, h,
                #                       epoch])
                # daemon.start()
                self.log_rep(best_lists, best_vals, 
                             best_values, traces, 
                             model, metrics, run, loggers, ae,
                             shap_ae, h, epoch)
                del ae, shap_ae

        # Logging every model is taking too much resources and it makes it quite complicated to get information when
        # Too many runs have been made. This will make the notebook so much easier to work with
        if np.mean(best_mccs) > self.best_mcc:
            try:
                if os.path.exists(
                        f'logs/best_models/ae_classifier_holdout/{self.args.dataset}/{self.args.dloss}'):
                    shutil.rmtree(
                        f'logs/best_models/ae_classifier_holdout/{self.args.dataset}/{self.args.dloss}',
                        ignore_errors=True)
                # os.makedirs(f'logs/best_models/ae_classifier_holdout/{self.args.dloss}', exist_ok=True)
                shutil.copytree(f'{self.complete_log_path}',
                                f'logs/best_models/ae_classifier_holdout/{self.args.dataset}/{self.args.dloss}')
                # print("File copied successfully.")

            # If source and destination are same
            except shutil.SameFileError:
                # print("Source and destination represents the same file.")
                pass
            self.best_mcc = np.mean(best_mccs)

        # Logs confusion matrices in the background. Also runs RandomForestClassifier on encoded and reconstructed
        # representations. This should be shorter than the actual calculation of the model above in the function,
        # otherwise the number of threads will keep increasing.
        # daemon = Thread(target=self.logging, daemon=True, name='Monitor', args=[run, cm_logger])
        # daemon.start()
        if self.log_mlflow:
            mlflow.log_param('finished', 1)
        self.logging(run, loggers['cm_logger'])

        if not self.keep_models:
            # shutil.rmtree(f'{self.complete_log_path}/traces', ignore_errors=True)
            # shutil.rmtree(f'{self.complete_log_path}/cm', ignore_errors=True)
            # shutil.rmtree(f'{self.complete_log_path}/hp', ignore_errors=True)
            shutil.rmtree(f'{self.complete_log_path}', ignore_errors=True)
        print('Duration: {}'.format(datetime.now() - start_time))
        best_closs = np.mean(best_closses)
        if best_closs < self.best_closs:
            self.best_closs = best_closs
            print("Best closs!")

        # It should not be necessary. To remove once certain the "Too many files open" error is no longer a problem
        plt.close('all')

        return self.best_mcc

    def log_rep(self, best_lists, best_vals, best_values, traces, model, metrics, run, loggers, ae, shap_ae, h,
                epoch):
        best_traces = self.get_mccs(best_lists, traces)

        self.log_predictions(best_lists, run, h)

        if self.args.log_metrics:
            if self.log_tb:
                try:
                    # logger, lists, values, model, unique_labels, mlops, epoch, metrics, n_meta_emb=0, device='cuda'
                    metrics = log_metrics(loggers['logger'], best_lists, best_vals, ae,
                                          self.unique_labels,
                                          self.unique_batches, epoch, mlops="tensorboard",
                                          metrics=metrics, n_meta_emb=self.args.embeddings_meta,
                                          device=self.args.device)
                except BrokenPipeError:
                    print("\n\n\nProblem with logging stuff!\n\n\n")
            if self.log_neptune:
                try:
                    metrics = log_metrics(run, best_lists, best_vals, ae,
                                          self.unique_labels, self.unique_batches,
                                          epoch=epoch, mlops="neptune",
                                          metrics=metrics, n_meta_emb=self.args.embeddings_meta,
                                          device=self.args.device)
                except BrokenPipeError:
                    print("\n\n\nProblem with logging stuff!\n\n\n")
            if self.log_mlflow:
                try:
                    metrics = log_metrics(None, best_lists, best_vals, ae,
                                          self.unique_labels,
                                          self.unique_batches, epoch, mlops="mlflow",
                                          metrics=metrics, n_meta_emb=self.args.embeddings_meta,
                                          device=self.args.device)
                except BrokenPipeError:
                    print("\n\n\nProblem with logging stuff!\n\n\n")

        if self.args.log_metrics and self.pools:
            try:
                if self.log_neptune:
                    enc_data = make_data(best_lists, 'encoded_values')
                    metrics = log_pool_metrics(enc_data['inputs'], enc_data['batches'], enc_data['labels'],
                                               self.unique_unique_labels, run, epoch, metrics, 'enc', 'neptune')
                    rec_data = make_data(best_lists, 'rec_values')
                    metrics = log_pool_metrics(rec_data['inputs'], rec_data['batches'], rec_data['labels'],
                                               self.unique_unique_labels, run, epoch, metrics, 'rec', 'neptune')
                if self.log_mlflow:
                    enc_data = make_data(best_lists, 'encoded_values')
                    metrics = log_pool_metrics(enc_data['inputs'], enc_data['batches'], enc_data['labels'],
                                               self.unique_unique_labels, run, epoch, metrics, 'enc', 'mlflow')
                    rec_data = make_data(best_lists, 'rec_values')
                    metrics = log_pool_metrics(rec_data['inputs'], rec_data['batches'], rec_data['labels'],
                                               self.unique_unique_labels, run, epoch, metrics, 'rec', 'mlflow')
                if self.log_tb:
                    enc_data = make_data(best_lists, 'encoded_values')
                    metrics = log_pool_metrics(enc_data['inputs'], enc_data['batches'], enc_data['labels'],
                                               self.unique_unique_labels, loggers['logger'], epoch, metrics, 'enc',
                                               'tensorboard')
                    rec_data = make_data(best_lists, 'rec_values')
                    metrics = log_pool_metrics(rec_data['inputs'], rec_data['batches'], rec_data['labels'],
                                               self.unique_unique_labels, loggers['logger'], epoch, metrics, 'rec',
                                               'tensorboard')

            except BrokenPipeError:
                print("\n\n\nProblem with logging stuff!\n\n\n")

        loggers['cm_logger'].add(best_lists)
        if h == 1:
            if self.args.log_plots:
                if self.log_tb:
                    # TODO Add log_shap
                    loggers.add(loggers['logger_cm'], epoch, best_lists,
                               self.unique_labels, best_traces, 'tensorboard')
                    log_plots(loggers['logger_cm'], best_lists, 'tensorboard', epoch)
                    if log_shap is not None:
                        log_shap(loggers['logger_cm'], shap_ae, best_lists, self.columns, self.args.n_meta, 'mlflow',
                                self.complete_log_path,
                                self.args.device)
                if self.log_neptune:
                    if log_shap is not None:
                        log_shap(run, shap_ae, best_lists, self.columns, self.args.embeddings_meta, 'neptune',
                                self.complete_log_path,
                                self.args.device)
                    log_plots(run, best_lists, 'neptune', epoch)
                if self.log_mlflow:
                    if log_shap is not None:
                        log_shap(None, shap_ae, best_lists, self.columns, self.args.embeddings_meta, 'mlflow',
                                self.complete_log_path,
                                self.args.device)
                    log_plots(None, best_lists, 'mlflow', epoch)

        if self.log_neptune:
            run["recs"].track_files(f'{self.complete_log_path}/recs.csv')
            run["encs"].track_files(f'{self.complete_log_path}/encs.csv')
            # if self.log_mlflow:
            #     mlflow.log_artifact(f'{self.complete_log_path}/recs.csv')
            #     mlflow.log_artifact(f'{self.complete_log_path}/encs.csv')
            # try:
            #     enc_data = make_data(best_lists, 'encoded_values')
            #     if self.log_tb and self.pools:
            #         metrics = log_pool_metrics(enc_data['inputs'], enc_data['batches'], enc_data['labels'], self.unique_labels, ['logger_cm'], epoch,
            #                                    metrics,
            #                                    'enc', 'tensorboard')
            #     if self.log_neptune and self.pools:
            #         metrics = log_pool_metrics(enc_data['inputs'], enc_data['batches'], enc_data['labels'], run, epoch, metrics, 'enc',
            #                                    'neptune')
            #         metrics = log_pool_metrics(enc_data['inputs'], enc_data['batches'], enc_data['labels'], model, epoch, metrics,
            #                                    'enc',
            #                                    'neptune')
            #     if self.log_mlflow and self.pools:
            #         metrics = log_pool_metrics(enc_data['inputs'], enc_data['batches'], enc_data['labels'], run, epoch, metrics, 'enc',
            #                                    'mlflow')
            #         metrics = log_pool_metrics(enc_data['inputs'], enc_data['batches'], enc_data['labels'], model, epoch, metrics,
            #                                    'enc',
            #                                    'mlflow')
            #
            #     if self.pools:
            #         rec_data = make_data(best_lists, 'rec_values')
            #         if self.log_tb:
            #             metrics = log_pool_metrics(rec_data['inputs'], rec_data['batches'], rec_data['labels'], loggers['logger_cm'], epoch,
            #                                        metrics,
            #                                        'rec', 'tensorboard')
            #         if self.log_neptune:
            #             metrics = log_pool_metrics(rec_data['inputs'], rec_data['batches'], rec_data['labels'], run, epoch, metrics, 'rec',
            #                                        'neptune')
            #             metrics = log_pool_metrics(rec_data['inputs'], rec_data['batches'], rec_data['labels'], model, epoch, metrics,
            #                                        'rec',
            #                                        'neptune')
            #         if self.log_mlflow:
            #             metrics = log_pool_metrics(rec_data['inputs'], rec_data['batches'], rec_data['labels'], run, epoch, metrics, 'rec',
            #                                        'mlflow')
            #             metrics = log_pool_metrics(rec_data['inputs'], rec_data['batches'], rec_data['labels'], model, epoch, metrics,
            #                                        'rec',
            #                                        'mlflow')

            ### TODO once log_tb and log_neptune are repaired, remove the block below

            # best_values['enc b_euclidean/tot_eucl'] = metrics['pool_metrics_enc']['all'][
            #     '[b_euclidean/tot_eucl]']
            # best_values['rec b_euclidean/tot_eucl'] = metrics['pool_metrics_rec']['all'][
            #     '[b_euclidean/tot_eucl]']
            # best_values['enc qc_aPCC'] = metrics['pool_metrics_enc']['all']['qc_aPCC']
            # best_values['rec qc_aPCC'] = metrics['pool_metrics_rec']['all']['qc_aPCC']
            # best_values['enc qc_dist/tot_eucl'] = metrics['pool_metrics_enc']['all']['[qc_dist/tot_eucl]']
            # best_values['rec qc_dist/tot_eucl'] = metrics['pool_metrics_rec']['all']['[qc_dist/tot_eucl]']
            # best_values['enc batch_entropy'] = metrics['pool_metrics_enc']['test']['qc_aPCC']
            # best_values['rec batch_entropy'] = metrics['pool_metrics_rec']['test']['qc_aPCC']

            ### TODO once log_tb and log_neptune are repaired, remove the block above

            best_values['pool_metrics'] = {}
            if self.args.log_metrics:
                best_values['batches'] = metrics['batches']
                best_values['pool_metrics']['enc'] = metrics['pool_metrics_enc']
                best_values['pool_metrics']['rec'] = metrics['pool_metrics_rec']
        # TODO change logging with tensorboard and neptune. The previous
        if self.log_tb:
            loggers['tb_logging'].logging(best_values, metrics)
        if self.log_neptune:
            log_neptune(run, best_values)
        if self.log_mlflow:
            log_mlflow(best_values, h)

        # except BrokenPipeError:
        #     print("\n\n\nProblem with logging stuff!\n\n\n")

    def logging(self, run, cm_logger):

        if self.log_neptune:
            cm_logger.plot(run, 0, self.unique_labels, 'neptune')
            # cm_logger.get_rf_results(run, self.args)
            run.stop()
        if self.log_mlflow:
            cm_logger.plot(None, 0, self.unique_labels, 'mlflow')
            # cm_logger.get_rf_results(run, self.args)
            mlflow.end_run()
        # cm_logger.close()
        # logger.close()

    def log_predictions(self, best_lists, run, step):
        cats, labels, preds, scores, names = [{'train': [], 'valid': [], 'test': []} for _ in range(5)]
        for group in ['train', 'valid', 'test']:
            cats[group] = np.concatenate(best_lists[group]['cats'])
            labels[group] = np.concatenate(best_lists[group]['labels'])
            scores[group] = torch.softmax(torch.Tensor(np.concatenate(best_lists[group]['preds'])), 1)
            preds[group] = scores[group].argmax(1)
            names[group] = np.concatenate(best_lists[group]['names'])
            pd.DataFrame(np.concatenate((labels[group].reshape(-1, 1), scores[group],
                                         np.array([self.unique_labels[x] for x in preds[group]]).reshape(-1, 1),
                                         names[group].reshape(-1, 1)), 1)).to_csv(
                f'{self.complete_log_path}/{group}_predictions.csv')
            if self.log_neptune:
                run[f"{group}_predictions"].track_files(f'{self.complete_log_path}/{group}_predictions.csv')
                run[f'{group}_AUC'] = metrics.roc_auc_score(y_true=cats[group], y_score=scores[group],
                                                            multi_class='ovr')
            if self.log_mlflow:
                try:
                    mlflow.log_metric(f'{group}_AUC',
                                    metrics.roc_auc_score(y_true=cats[group], y_score=scores[group], multi_class='ovr'),
                                    step=step)
                except:
                    pass

    def loop(self, group, optimizers, ae, celoss, loader, lists, traces, gamma, nu=1):
        """

        Args:
            group: Which set? Train, valid or test
            optimizer_ae: Object that contains the optimizer for the autoencoder
            ae: AutoEncoder (pytorch model, inherits nn.Module)
            celoss: torch.nn.CrossEntropyLoss instance
            triplet_loss: torch.nn.TripletMarginLoss instance
            loader: torch.utils.data.DataLoader
            lists: List keeping informations on the current run
            traces: List keeping scores on the current run
            nu: indicates if learning is happening

        Returns:

        """
        # If group is train and nu = 0, then it is not training. valid can also have sampling = True
        # if group in ['train', 'valid'] and nu != 0:
        #     sampling = True
        # else:
        #     sampling = False
        classif_loss = None
        for i, batch in enumerate(loader):
            data, meta_inputs, names, labels, domain, to_rec, not_to_rec, pos_batch_sample, \
                neg_batch_sample, meta_pos_batch_sample, meta_neg_batch_sample = batch
            # data[torch.isnan(data)] = 0
            if nu == 1 and self.args.dloss == 'DANN' and group == 'train' and self.args.bdisc:
                self.forward_discriminate(optimizers['b'], ae, celoss, data, domain)
            data = data.to(self.args.device).float()
            meta_inputs = meta_inputs.to(self.args.device).float()
            to_rec = to_rec.to(self.args.device).float()

            # If n_meta > 0, meta data added to inputs

            if self.args.n_meta > 0:
                data = torch.cat((data, meta_inputs), 1)
                to_rec = torch.cat((to_rec, meta_inputs), 1)
            not_to_rec = not_to_rec.to(self.args.device).float()
            enc = ae(data)

            # If embedding_meta > 0, meta data added to embeddings
            if self.args.embeddings_meta:
                preds = ae.classifier(torch.cat((enc, meta_inputs), 1))
            else:
                preds = ae.classifier(enc)

            try:
                cats = to_categorical(labels.long(), self.n_cats).to(self.args.device).float()
                classif_loss = celoss(preds, cats)
            except:
                cats = torch.Tensor([self.n_cats + 1 for _ in labels])
                classif_loss = torch.Tensor([0])

            reverse = ReverseLayerF.apply(enc, 1)
            if self.args.dloss == 'DANN':
                domain_preds = ae.dann_discriminator(reverse)
            else:
                domain_preds = ae.dann_discriminator(enc)
            if self.args.dloss not in ['revTriplet', 'inverseTriplet']:
                dloss, domain = self.get_dloss(celoss, domain, domain_preds)
            elif self.args.dloss == 'revTriplet':
                pos_batch_sample = pos_batch_sample.to(self.args.device).float()
                neg_batch_sample = neg_batch_sample.to(self.args.device).float()
                meta_pos_batch_sample = meta_pos_batch_sample.to(self.args.device).float()
                meta_neg_batch_sample = meta_neg_batch_sample.to(self.args.device).float()
                if self.args.n_meta > 0:
                    pos_batch_sample = torch.cat((pos_batch_sample, meta_pos_batch_sample), 1)
                    neg_batch_sample = torch.cat((neg_batch_sample, meta_neg_batch_sample), 1)
                pos_enc = ae(pos_batch_sample)
                neg_enc = ae(neg_batch_sample)
                dloss = self.triplet_loss(reverse,
                                     ReverseLayerF.apply(pos_enc, 1),
                                     ReverseLayerF.apply(neg_enc, 1)
                                     )
            elif self.args.dloss == 'inverseTriplet':
                pos_batch_sample, neg_batch_sample = neg_batch_sample.to(self.args.device).float(), pos_batch_sample.to(
                    self.args.device).float()
                meta_pos_batch_sample, meta_neg_batch_sample = meta_neg_batch_sample.to(
                    self.args.device).float(), meta_pos_batch_sample.to(self.args.device).float()
                if self.args.n_meta > 0:
                    pos_batch_sample = torch.cat((pos_batch_sample, meta_pos_batch_sample), 1)
                    neg_batch_sample = torch.cat((neg_batch_sample, meta_neg_batch_sample), 1)
                pos_enc = ae(pos_batch_sample)
                neg_enc = ae(neg_batch_sample)
                dloss = self.triplet_loss(enc, pos_enc, neg_enc)
                # domain = domain.argmax(1)


            lists[group]['set'] += [np.array([group for _ in range(len(domain))])]
            lists[group]['domains'] += [np.array([self.unique_batches[d] for d in domain.detach().cpu().numpy()])]
            lists[group]['domain_preds'] += [domain_preds.detach().cpu().numpy()]
            lists[group]['preds'] += [preds.detach().cpu().numpy()]
            lists[group]['classes'] += [labels.detach().cpu().numpy()]
            # lists[group]['encoded_values'] += [enc.view(enc.shape[0], -1).detach().cpu().numpy()]
            lists[group]['names'] += [names]
            lists[group]['cats'] += [cats.detach().cpu().numpy()]
            lists[group]['gender'] += [data.detach().cpu().numpy()[:, -1]]
            lists[group]['age'] += [data.detach().cpu().numpy()[:, -2]]
            lists[group]['atn'] += [str(x) for x in data.detach().cpu().numpy()[:, -5:-2]]
            # lists[group]['inputs'] += [data.view(rec.shape[0], -1).detach().cpu().numpy()]
            lists[group]['encoded_values'] += [enc.detach().cpu().numpy()]
            # lists[group]['rec_values'] += [rec.detach().cpu().numpy()]
            try:
                lists[group]['labels'] += [np.array(
                    [self.unique_labels[x] for x in labels.detach().cpu().numpy()])]
            except:
                pass
            try:
                traces[group]['acc'] += [np.mean([0 if pred != dom else 1 for pred, dom in
                                              zip(preds.detach().cpu().numpy().argmax(1),
                                                  labels.detach().cpu().numpy())])]
            except:
                pass
            traces[group]['top3'] += [np.mean([1 if label.item() in pred.tolist()[::-1][:3] else 0 for pred, label in
                                               zip(preds.argsort(1), labels)])]

            traces[group]['closs'] += [classif_loss.item()]
            traces[group]['mcc'] += [np.round(
                MCC(labels.detach().cpu().numpy(), preds.detach().cpu().numpy().argmax(1)), 3)
            ]
            if group in ['train'] and nu != 0:
                optimizers['ae'].zero_grad()
                w = 1
                total_loss = w * classif_loss + gamma * dloss
                try:
                    total_loss.backward()
                except:
                    pass
                # nn.utils.clip_grad_norm_(ae.classifier.parameters(), max_norm=1)
                optimizers['ae'].step()

        return classif_loss, lists, traces

    def forward_discriminate(self, optimizer_b, ae, celoss, data, domain):
        # Freezing the layers so the batch discriminator can get some knowledge independently
        # from the part where the autoencoder is trained. Only for DANN
        self.freeze_all_but_dlayers(ae)
        optimizer_b.zero_grad()
        # data, meta_inputs, names, labels, domain, to_rec, not_to_rec, pos_batch_sample, \
        #     neg_batch_sample, meta_pos_batch_sample, meta_neg_batch_sample = batch
        # data[torch.isnan(data)] = 0
        data = data.to(self.args.device).float()
        with torch.no_grad():
            enc = ae(data)
        with torch.enable_grad():
            domain_preds = ae.dann_discriminator(enc)
            bclassif_loss = celoss(domain_preds,
                                    to_categorical(domain.long(), self.n_batches).to(self.args.device).float())
            bclassif_loss.backward()
            optimizer_b.step()
        self.unfreeze_layers(ae)

    def freeze_all_but_dlayers(self, ae):
        """
        Freezes all layers except the batch discriminator layers
        Args:
            ae: AutoEncoder object. It inherits torch.nn.Module

        Returns:
            ae: The same AutoEncoder object, but with all layers frozen except the batch discriminator layers

        """
        for name, param in ae.named_parameters():
            if 'dann_discriminator' not in name:
                param.requires_grad = False
        return ae

    def unfreeze_layers(self, ae):
        """
        Unfreeze all layers
        Args:
            ae: AutoEncoder object. It inherits torch.nn.Module

        Returns:
            ae: The same AutoEncoder object, but with all frozen layers. Only the classifier layers are not frozen.

        """
        for param in ae.parameters():
            param.requires_grad = True
        return ae

    def get_dloss(self, celoss, domain, domain_preds):
        """
        This function is used to get the domain loss
        Args:
            celoss: PyTorch CrossEntropyLoss instance object
            domain: one-hot encoded domain classes []
            domain_preds: Matrix containing the predicted domains []

        Returns:
            dloss: Domain loss
            domain: True domain (batch) values
        """
        if args.dloss in ['revTriplet', 'revDANN', 'DANN', 'inverseTriplet', 'normae']:
            domain = domain.to(self.args.device).long().to(self.args.device)
            dloss = celoss(domain_preds, domain)
        else:
            dloss = torch.zeros(1)[0].float().to(self.args.device)
        if args.dloss == 'normae':
            dloss = -dloss
        return dloss, domain

    def get_losses(self, scale, smooth, margin, args):
        """
        Getter for the losses.
        Args:
            scale: Scaler that was used, e.g. normalizer or binarize
            smooth: Parameter for label_smoothing
            margin: Parameter for the TripletMarginLoss

        Returns:
            sceloss: CrossEntropyLoss (with label smoothing)
            celoss: CrossEntropyLoss object (without label smoothing)
            mseloss: MSELoss object
            triplet_loss: TripletMarginLoss object
        """
        if args.classif_loss == 'focal':
            sceloss = FocalLoss().to(self.args.device)
            celoss = FocalLoss().to(self.args.device)
        else:
            sceloss = nn.CrossEntropyLoss()
            celoss = nn.CrossEntropyLoss()
        if self.args.rec_loss == 'mse':
            mseloss = nn.MSELoss()
        elif self.args.rec_loss == 'l1':
            mseloss = nn.L1Loss()
        if scale == "binarize":
            mseloss = nn.BCELoss()
        if args.dloss == 'revTriplet':
            triplet_loss = nn.TripletMarginLoss(margin, p=2, swap=True)
        else:
            triplet_loss = nn.TripletMarginLoss(0, p=2, swap=False)

        return sceloss, celoss, mseloss, triplet_loss


    @staticmethod
    def get_mccs(lists, traces):
        """
        Function that gets the Matthews Correlation Coefficients. MCC is a statistical tool for model evaluation.
        It is a balanced measure which can be used even if the classes are of very different sizes.
        Args:
            lists:
            traces:

        Returns:
            traces: Same list as in the inputs arguments, except in now contains the MCC values
        """
        for group in ['train', 'valid', 'test']:
            try:
                preds, classes = np.concatenate(lists[group]['preds']).argmax(1), np.concatenate(
                    lists[group]['classes'])
            except:
                pass
            traces[group]['mcc'] = MCC(preds, classes)

        return traces


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument('--batch_columns', type=str, default='2,3')
    parser.add_argument('--random_recs', type=int, default=0)  # TODO no longer used
    parser.add_argument('--predict_tests', type=int, default=0)  # TODO no longer used
    parser.add_argument('--early_stop', type=int, default=100)
    parser.add_argument('--threshold', type=float, default=0.)
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--n_trials', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--rec_loss', type=str, default='mse')  # TODO no longer used
    parser.add_argument('--tied_weights', type=int, default=0)  # TODO no longer used
    parser.add_argument('--use_mapping', type=int, default=1, help="Use batch mapping for reconstruct")  # TODO no longer used
    parser.add_argument('--freeze_ae', type=int, default=0)  # TODO no longer used
    parser.add_argument('--freeze_c', type=int, default=0)  # TODO no longer used
    parser.add_argument('--bdisc', type=int, default=1)  # TODO no longer used
    parser.add_argument('--n_repeats', type=int, default=5)
    parser.add_argument('--dloss', type=str, default='inverseTriplet')  # one of revDANN, DANN, inverseTriplet, revTriplet  # TODO no longer used
    parser.add_argument('--csv_file', type=str, default='matrix.csv')
    parser.add_argument('--best_features_file', type=str, default='')  # best_unique_genes.tsv
    parser.add_argument('--bad_batches', type=str, default='')  # 0;23;22;21;20;19;18;17;16;15
    parser.add_argument('--remove_zeros', type=int, default=1)  # TODO no longer used?
    parser.add_argument('--n_meta', type=int, default=0)  # TODO no longer used
    parser.add_argument('--embeddings_meta', type=int, default=0)  # TODO no longer used
    parser.add_argument('--features_to_keep', type=str, default='features_proteins.csv')
    parser.add_argument('--groupkfold', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='bacteria')
    parser.add_argument('--exp_id', type=str, default='reviewer_exp')
    parser.add_argument('--bs', type=int, default=8, help='Batch size')
    parser.add_argument('--n_agg', type=int, default=1, 
                        help='Number of trailing values to get stable valid values')
    parser.add_argument('--n_layers', type=int, default=1, help='N layers for classifier')  # TODO no longer used
    parser.add_argument('--log1p', type=int, default=0, help='log1p the data?')
    parser.add_argument('--strategy', type=str, default='CU_DEM', 
                        help='only for alzheimer dataset')  # TODO no longer used
    parser.add_argument('--pool', type=int, default=1, help='only for alzheimer dataset')  # TODO no longer used, but could make a comeback
    parser.add_argument('--log_plots', type=int, default=0, help='')
    parser.add_argument('--log_metrics', type=int, default=0, help='')
    parser.add_argument('--controls', type=str, default='', 
                        help='Which samples are the controls. Empty for not binary')
    parser.add_argument('--n_features', type=int, default=-1, help='')
    parser.add_argument('--triplet_dloss', type=int, default=0, help='')  # TODO no longer used
    parser.add_argument('--new_size', type=int, default=299, help='') # Use 0 for no resize
    parser.add_argument('--path', type=str, default="resources/bacteries_2024/matrices/mz10/rt10/mzp10/rtp10/thr0.0/200spd/ms2/combat0/shift0/none/loginloop/none/all_B14-B13-B12-B11-B10-B9-B8-B7-B6-B5-B4-B3-B2-B1_gkf0_mz0-1200rt0-320_5splits/images", help='')
    parser.add_argument('--test', type=int, default=0, help='')
    parser.add_argument('--classif_loss', type=str, default='focal', help='')
    parser.add_argument('--model', type=str, default='resnet18', help='')

    args = parser.parse_args()
    try:
        mlflow.create_experiment(
            args.exp_id,
            # artifact_location=Path.cwd().joinpath("mlruns").as_uri(),
            # tags={"version": "v1", "priority": "P1"},
        )
    except:
        print(f"\n\nExperiment {args.exp_id} already exists\n\n")

    # args.batch_columns = [int(x) for x in args.batch_columns.split(',')]

    train = TrainAE(args, fix_thres=-1, load_tb=False, keep_models=False,
                    log_inputs=False, log_tb=False, log_neptune=False,
                    log_mlflow=True)

    # train.train()
    # List of hyperparameters getting optimized
    parameters = [
        {"name": "lr", "type": "range", "bounds": [1e-6, 1e-2], "log_scale": True},
        {"name": "wd", "type": "range", "bounds": [1e-6, 1e-2], "log_scale": True},
        {"name": "smoothing", "type": "range", "bounds": [0., 0.2]},
        {"name": "margin", "type": "range", "bounds": [0., 10.]},
        {"name": "disc_b_warmup", "type": "range", "bounds": [1, 2]},
        {"name": "dropout", "type": "range", "bounds": [0.0, 0.5]},
        {"name": "scaler", "type": "choice", "values": ['minmax2', 'none', 'zscale']},
        {"name": "num_layers", "type": "range", "bounds": [1, 3]},
        {"name": "num_heads", "type": "range", "bounds": [2, 10]},
        {"name": "embed_dim", "type": "range", "bounds": [3, 300]},
    ]
    

    # Some hyperparameters are not always required. They are set to a default value in Train.train()
    if args.dloss in ['revTriplet', 'revDANN', 'DANN', 'inverseTriplet', 'normae']:
        # gamma = 0 will ensure DANN is not learned
        parameters += [{"name": "gamma", "type": "range", "bounds": [1e-2, 1e2], "log_scale": True}]

    best_parameters, values, experiment, model = optimize(
        parameters=parameters,
        evaluation_function=train.train,
        objective_name='closs',
        minimize=True,
        total_trials=args.n_trials,
        random_seed=41,

    )

    # fig = plt.figure()
    # render(plot_contour(model=model, param_x="learning_rate", param_y="weight_decay", metric_name='Loss'))
    # fig.savefig('test.jpg')
    # print('Best Loss:', values[0]['loss'])
    # print('Best Parameters:')
    # print(json.dumps(best_parameters, indent=4))
