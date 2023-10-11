import pdb
import typing
import pandas as pd
import lightgbm as lgb
from lightgbm.callback import early_stopping
from lightgbm import LGBMRegressor
import xgboost as xgb
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV, RidgeCV
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
import os
from dateutil.relativedelta import relativedelta
import numpy as np
from data_centre.helpers import coin_ls
from data_centre.data import Reader
import concurrent.futures
from hottbox.pdtools import pd_to_tensor
from itertools import product
from scipy.stats import t
import sqlite3
from sklearn.metrics import silhouette_score
from model.feature_engineering_room import FeatureAR, FeatureRiskMetricsEstimator, FeatureHAR,\
    FeatureHAREq, FeatureUniversal #FeatureHARCDR, FeatureHARCSR
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import random
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import metrics
import warnings
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm

_data_centre_dir = os.path.abspath(__file__).replace('/model/lab.py', '/data_centre/databases')

"""Functions used to facilitate computation within classes."""
qlike_score = lambda x: ((x.iloc[:, 0].div(x.iloc[:, -1]))-np.log(x.iloc[:, 0].div(x.iloc[:, -1]))-1).mean()


def flatten(ls: list()) -> list:
    flat_ls = []
    for sub_ls in ls:
        flat_ls += sub_ls
        return flat_ls


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, nn: bool = True, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.nn = nn
        self.best_model = None
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            if self.nn:
                self.save_checkpoint(val_loss, model)
            else:
                self.best_model = model
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.nn:
                self.save_checkpoint(val_loss, model)
            else:
                self.best_model = model
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if self.nn:
            torch.save(model.state_dict(), self.path)
        else:
            self.best_model = model
        self.val_loss_min = val_loss


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int=None, hidden_dim: int=None, output_dim: int=None, num_layers: int=None):
        super(Autoencoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        encoder_layers = [nn.Linear(in_features=input_dim, out_features=self.hidden_dim), nn.ReLU()]
        for i in range(num_layers):
            if i == num_layers-1:
                encoder_layers.append(nn.Linear(in_features=self.hidden_dim, out_features=output_dim))
            else:
                encoder_layers.append(nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim))
            encoder_layers.append(nn.LeakyReLU())
        self.encoder = nn.Sequential(*encoder_layers[:-1])
        decoder_layers = []
        for layer in encoder_layers[:-1]:
            if isinstance(layer, torch.nn.modules.linear.Linear):
                out_features = layer.in_features
                in_features = layer.out_features
                decoder_layer = nn.Linear(in_features=in_features, out_features=out_features)
            else:
                decoder_layer = layer
            decoder_layers.append(decoder_layer)
        self.decoder = nn.Sequential(*decoder_layers[::-1])

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class Time2Vec(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Time2Vec, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.biases = nn.Parameter(torch.Tensor(output_dim))

        # Initialize weights and biases
        nn.init.xavier_uniform_(self.weights)
        torch.nn.init.zeros_(self.biases)

    def forward(self, x):
        # Calculate Time2Vec embeddings
        embeddings = torch.sin(x.unsqueeze(-1) * self.weights + self.biases)

        return embeddings


# class LSTM(nn.Module):
#     def __init__(self, input_dim: int, hidden_dim: int, dropout_prob: float, num_layers: int,
#                  output_dim: int):
#         super(LSTM, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#         # LSTM layer with dropout
#         self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=self.num_layers, batch_first=True,
#                             dropout=dropout_prob)
#         # Fully connected layer
#         self.fc = nn.Linear(hidden_dim, output_dim)
#         # Initialisation
#         nn.init.xavier_uniform_(self.lstm.weight_ih_l0)
#         nn.init.orthogonal_(self.lstm.weight_hh_l0)
#         nn.init.constant_(self.lstm.bias_ih_l0, 0)
#         nn.init.constant_(self.lstm.bias_hh_l0, 0)
#
#     def forward(self, x):
#         # Initialize hidden state with zeros
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
#
#         # Forward pass through LSTM layer
#         out, _ = self.lstm(x, (h0, c0))
#
#         # Only take the output from the final time step
#         out = out[:, -1, :]
#
#         # Pass through the fully connected layer
#         out = self.fc(out)
#
#         return out


class EarlyStopping:
    def __init__(self, tolerance=0, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False
        self.best_loss = np.inf

    def __call__(self, loss):
        if self.best_loss > loss:
            self.best_loss = loss
            self.counter +=1
            if self.counter == self.tolerance:
                self.early_stop = True
        else:
            self.counter = 0


# Define the LSTNet model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=dropout)
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
        # Initialisation
        nn.init.xavier_uniform_(self.lstm.weight_ih_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)
        nn.init.constant_(self.lstm.bias_ih_l0, 0)
        nn.init.constant_(self.lstm.bias_hh_l0, 0)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class TransformersLSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, output_size, max_length):
        super(TransformersLSTM, self).__init__()
        # self.hidden_size = hidden_size
        # self.max_length = max_length

        # Define the LSTM encoder
        self.encoder = nn.LSTM(input_dim, hidden_size)

        # Define the attention layer
        self.attention = nn.Linear(hidden_size * 2, max_length)

        # Define the decoder LSTM
        self.decoder = nn.LSTM(hidden_size * 2, hidden_size)

        # Define the output layer
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, target_seq):
        # Initialize hidden states and cell states for encoder and decoder
        encoder_hidden = torch.zeros(1, 1, self.hidden_size)
        encoder_cell = torch.zeros(1, 1, self.hidden_size)

        decoder_hidden = torch.zeros(1, 1, self.hidden_size)
        decoder_cell = torch.zeros(1, 1, self.hidden_size)

        # Get the length of the input sequence
        input_length = input_seq.size(0)

        # Initialize the attention scores
        attention_weights = torch.zeros(self.max_length)

        # Initialize the output sequence
        output_seq = torch.zeros(target_seq.size())

        for i in range(input_length):
            # Encoder LSTM
            _, (encoder_hidden, encoder_cell) = self.encoder(input_seq[i].view(1, 1, -1),
                                                             (encoder_hidden, encoder_cell))

            # Attention Mechanism
            energy = self.attention(torch.cat((encoder_hidden[0], decoder_hidden[0]), 1))
            attention_weights[i] = energy

            # Calculate attention weights
            attention_weights = torch.softmax(attention_weights, dim=0)

            # Calculate the weighted sum of encoder hidden states
            context = torch.sum(attention_weights[i] * encoder_hidden, dim=0)

            # Decoder LSTM
            _, (decoder_hidden, decoder_cell) = self.decoder(context.view(1, 1, -1), (decoder_hidden, decoder_cell))

            # Output layer
            output_seq[i] = self.out(decoder_hidden)

        return output_seq


class DMTest:


    _db_connect_y = sqlite3.connect(database=os.path.abspath(f'{_data_centre_dir}/y.db'), check_same_thread=False)

    def __init__(self):
        pass

    def table(self, L: str, training_scheme: str) -> pd.DataFrame:
        """
            Method computing a matrix with DM statistics for every (i,j) as entries
        """
        regression_per_training_scheme_dd = {'SAM': ['linear', 'lasso', 'ridge', 'elastic'],
                                             'ClustAM': ['lasso', 'ridge', 'elastic'],
                                             'CAM': ['lasso', 'ridge', 'elastic']}
        models_dd = dict()
        for scheme in ['SAM', 'ClustAM', 'CAM']:
            for regression in regression_per_training_scheme_dd[scheme]:
                for model in ['risk_metrics', 'ar', 'har', 'har_eq']:
                    query = f'SELECT "timestamp", ("y"/"y_hat") * ( "y"-"y_hat") AS "e", "symbol" '\
                            f'FROM (SELECT "timestamp", "y", "y_hat", "symbol", "model", "L", "training_scheme",' \
                            f'"regression" FROM y_{L} WHERE "training_scheme" = \"{scheme}\" AND ' \
                            f'"regression" = \"{regression}\" AND "model" = \"{model}\");'
                    dm_table = pd.read_sql(query, con=DMTest._db_connect_y, chunksize=10_000)
                    dm_table = pd.concat(list(dm_table)).set_index('timestamp')
                    dm_table.index = pd.to_datetime(dm_table.index, utc=True)
                    dm_table.sort_index(inplace=True)
                    dm_table = \
                        dm_table.groupby(by=[dm_table.index, dm_table.symbol], group_keys=True).last().reset_index()
                    models_dd[(scheme, L, regression, model)] = \
                        pd.pivot(dm_table, index='timestamp', columns='symbol', values='e')
        dm_ls = list()
        for model in models_dd.keys():
            row = list()
            for model2 in models_dd.keys():
                tmp = models_dd[model].sub(models_dd[model2]).mean(axis=1)
                try:
                    row.append(tmp.mean()/tmp.std())
                except ZeroDivisionError:
                    row.append('NaN')
            dm_ls.append(row)
        dm = pd.DataFrame(dm_ls, columns=models_dd.keys(), index=models_dd.keys()).round(4)
        return dm.loc[:, dm.columns.get_level_values(0).isin([training_scheme])]


class Commonality:

    _reader_obj = Reader()
    _rv = _reader_obj.rv_read(variance=True)
    _mkt = _rv.mean(axis=1)
    _commonality_connect_db = sqlite3.connect(database=os.path.abspath(f'{_data_centre_dir}/commonality.db'))
    _linear = LinearRegression()
    _commonality_per_freq_dd = dict([(symbol, None) for symbol in _rv.columns])
    _factory_transformation_dd = {'log': {'transformation': np.log, 'inverse': np.exp},
                                  None: {'transformation': lambda x: x, 'inverse': lambda x: x}}
    _timeseries_split = TimeSeriesSplit()

    def __init__(self, L: str, transformation: str = None):
        self._L = L
        self._transformation = transformation
        Commonality._rv = \
            Commonality._factory_transformation_dd[self._transformation]['transformation'](Commonality._rv)
        self._rv_fitted = pd.DataFrame(data=np.nan, index=Commonality._rv.index, columns=Commonality._rv.columns)
        self._rv_fitted.columns = self._rv_fitted.columns.str.replace('USDT', 'USDT_hat')

    def raw_commonality(self) -> None:
        commonality_df = pd.DataFrame(data=np.nan, index=Commonality._rv.resample('M').groups.keys(),
                                      columns=Commonality._rv.columns) #self._L
        if 'D' not in self._L:
            D = int(self._L[:-1])*30*288 if 'M' in self._L else int(self._L[:-1])*7*288
        else:
            D = 288
        for i in range(0, Commonality._rv.shape[1]):
            X_train = Commonality._rv.copy()
            X_train.replace(-1*np.inf, np.nan, inplace=True)
            X_train.ffill(inplace=True)
            y_train = X_train.iloc[:, i]
            X_train_m = sm.add_constant(X_train.mean(axis=1))
            rres = RollingOLS(endog=y_train, exog=X_train_m, window=D).fit()
            params = rres.params.copy()
            y_hat = (params * X_train_m).sum(axis=1)
            y_hat.replace(0, np.nan, inplace=True)
            self._rv_fitted.iloc[:, i] = y_hat
        self._rv_fitted = \
            Commonality._factory_transformation_dd[self._transformation]['inverse'](self._rv_fitted)
        Commonality._rv = \
            Commonality._factory_transformation_dd[self._transformation]['inverse'](Commonality._rv)
        self._rv_fitted.dropna(inplace=True)
        Commonality._rv = Commonality._rv.loc[self._rv_fitted.index, :]
        for i in range(0, Commonality._rv.shape[1]):
            tmp = pd.DataFrame(data={Commonality._rv.iloc[:, i].name: Commonality._rv.iloc[:, i],
                                     self._rv_fitted.iloc[:, i].name: self._rv_fitted.iloc[:, i]})
            tmp = tmp.groupby(by=pd.Grouper(level=0, freq='M'))
            r2 = tmp.apply(lambda x: 1-(1-r2_score(x.iloc[:, 0], x.iloc[:, -1]))*(x.shape[0]-1)/(x.shape[0]-2))
            commonality_df.iloc[:, i] .loc[r2.index] = r2.values
        commonality_df.to_sql(f'raw_commonality_{self._L}', con=Commonality._commonality_connect_db,
                              if_exists='replace')
        print(f'[Insertion]: Raw commonality table {self._L} has been inserted into the database')

    def commonality(self) -> None:
        query = f'SELECT name FROM sqlite_master WHERE type="table";'
        check_table = pd.read_sql(query, con=Commonality._commonality_connect_db)
        if f'raw_commonality_{self._L}' not in check_table.name:
            self.raw_commonality()
        commonality_df = \
            pd.read_sql(f'SELECT * FROM raw_commonality_{self._L}', con=Commonality._commonality_connect_db,
                        index_col='index')
        commonality_df.index = pd.to_datetime(commonality_df.index)
        commonality_df = commonality_df.mean(axis=1)
        commonality_df.columns = [self._L]
        commonality_df = pd.melt(pd.DataFrame(commonality_df), ignore_index=False, var_name='L', value_name='values')
        commonality_df['L'] = self._L
        commonality_df.to_sql('commonality', con=Commonality._commonality_connect_db, if_exists='append')
        print(f'[Insertion]: Commonality table has been inserted into the database')

