import concurrent.futures
import pdb
import typing
from typing import Tuple
import pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os
from dateutil.relativedelta import relativedelta
import numpy as np
from data_centre.data import Reader, DBQuery
import sqlite3
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from datetime import datetime
import plotly.express as px
from statsmodels.tsa.api import VAR
from statsmodels import tsa
import pytz
import plotly.io as pio
import itertools
pandas2ri.activate()
var = importr("vars")
spillover = importr("Spillover")
pio.kaleido.scope.mathjax = None
TITLE_FONT_SIZE = 40
LABEL_AXIS_FONT_SIZE = 20
LEGEND_FONT_SIZE = 18
HEIGHT = 600
WIDTH = 800
FORMAT = '%Y-%m-%d'
DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'

_data_centre_dir = os.path.abspath(__file__).replace('/model/lab.py', '/data_centre/databases')
L_dd = {'1W': '7D', '1M': '30D', '6M': '180D'}


"""Functions used to facilitate computation within classes."""


def qlike_score(x: pd.DataFrame, log: bool = False) -> float:
    qlike = x.iloc[:, 0].div(x.iloc[:, -1]) - np.log(x.iloc[:, 0].div(x.iloc[:, -1]))-1 if not log else \
        np.exp(x.iloc[:, 0]).div(np.exp(x.iloc[:, -1]))-x.iloc[:, 0].subtract(x.iloc[:, -1])-1
    return qlike


def lags(df: pd.DataFrame, L_train: typing.List[str]):
    tmp = df.copy().reset_index(0, names=['symbol'])
    col_name = tmp.columns[-1]
    for l, L in enumerate(L_train):
        if l == 0:
            tmp[f'{col_name}_30T'] = tmp['RV'].shift()
        elif L in ['1W', '1M', '6M']:
            L_tmp = tmp[['RV']].resample(L).mean().shift()
            L_tmp.columns = L_tmp.columns.str.replace('RV', f'RV_{L}')
            tmp = pd.concat([tmp, L_tmp], axis=1)
        else:
            tmp[f'{col_name}_{L}'] = \
                df.droplevel(axis=0, level=0).rolling((pd.to_timedelta(L)//pd.to_timedelta('5T'))).mean().shift().values
    tmp = tmp.ffill().bfill()
    tmp = tmp.set_index('symbol', append=True).swaplevel(-1, 0)
    tmp.index.rename(['symbol', 'timestamp'], inplace=True)
    return tmp


def split_train_valid_set(df: pd.DataFrame) -> pd.core.indexes.datetimes.DatetimeIndex:
    dates = df.index.get_level_values(1).unique()
    train_idx = len(dates)//5
    return dates[:-train_idx]


def scaling(kind: str, df: pd.DataFrame) -> typing.Tuple[pd.DataFrame]:
    second, third = pd.DataFrame(), pd.DataFrame()
    if kind == 'standardise':
        first = df.groupby(by=pd.Grouper(level=0)).apply(lambda x: x.mean())
        second = df.groupby(by=pd.Grouper(level=0)).apply(lambda x: x.std())
    elif kind == 'normalise':
        first = df.groupby(by=pd.Grouper(level=0)).apply(lambda x: x.max())
        second = df.groupby(by=pd.Grouper(level=0)).apply(lambda x: x.min())
        third = first.sub(second)
    elif kind == 'maxabs':
        first = df.groupby(by=pd.Grouper(level=0)).apply(lambda x: x.abs().max())
    return first, second, third


def inverse_scaling(kind: str, df: pd.DataFrame, first: pd.DataFrame, second: pd.DataFrame,
                    third: pd.DataFrame) -> typing.Tuple[pd.DataFrame]:
    if kind == 'standardise':
        return df.mul(second[['RV']]).add(first[['RV']])
    elif kind == 'normalise':
        return df.mul(third[['RV']]).add(second[['RV']])
    elif kind == 'maxabs':
        return df.mul(first[['RV']])


def training_freq(date: datetime, freq: str = '1D'):
    if freq == '1D':
        return True
    if freq == '1W':
        return date.isocalendar()[1] != (date - relativedelta(days=1)).isocalendar()[1]
    elif freq == '1M':
        return date.month != (date - relativedelta(days=1)).month
    elif freq == '6M':
        return (date == datetime(date.year, 7, 1).date()) | (date == datetime(date.year, 1, 1).date())


class DMTest:

    query_obj = DBQuery()
    reader_obj = Reader()

    def __init__(self):
        self._L = None

    @property
    def L(self):
        return self._L

    @L.setter
    def L(self, L: str):
        self._L = L

    @staticmethod
    def table(h: str = '30min') -> None:
        """
            Method computing a matrix with DM statistics for every (i,j) as entries
        """
        table = DMTest.query_obj.best_model_for_all_windows_query()
        # table = DMTest.query_obj.query_data(table, table='y')
        rv = DMTest.reader_obj.rv_read()
        dm_test = dict()
        final_dm_results = dict()
        SAM = dict()
        for idx, L in enumerate(['1W', '1M', '6M']):
            tmp = DMTest.query_obj.query_data(query=DMTest.query_obj.training_query(L=L, training_scheme='SAM'),
                                              table='y')
            tmp['trading_session'] = tmp['trading_session'].fillna('NaN')
            tmp['top_book'] = tmp['top_book'].fillna('NaN')
            if L != '6M':
                tmp = tmp.pivot_table(columns=['model', 'regression', 'trading_session', 'top_book', 'symbol'],
                                      index='timestamp', values='y_hat')
            else:
                tmp = tmp.pivot(columns=['model', 'regression', 'trading_session', 'top_book', 'symbol'],
                                index='timestamp', values='y_hat')
            SAM[L.lower()] = tmp
            best_model = table.query(f'L == "{L}"')
            best_model = \
                DMTest.query_obj.forecast_query(L=best_model['L'].values[0],
                                                training_scheme=best_model['training_scheme'].values[0],
                                                model=best_model['model'].values[0],
                                                trading_session=best_model['trading_session'].values[0] if
                                                np.isnan(best_model['trading_session'].values[0]) else
                                                int(best_model['trading_session'].values[0]),
                                                top_book=best_model['top_book'].values[0] if
                                                np.isnan(best_model['top_book'].values[0]) else
                                                int(best_model['top_book'].values[0]),
                                                regression=best_model['regression'].values[0])
            best_model = pd.pivot(best_model[['y_hat', 'symbol']], columns='symbol', values='y_hat')
            best_model.index = pd.to_datetime(best_model.index)
            if 'resampled_rv' not in vars():
                resampled_rv = rv.resample(h).sum()
            for _, SAM_table in SAM.items():
                for tag in SAM_table.columns.droplevel(-1).unique():
                    tmp = SAM_table.loc[:, tag]
                    tmp.columns.name = None
                    tmp.index.name = None
                    tmp.index = pd.to_datetime(tmp.index, utc=True)
                    dm_test[tag] = \
                        (tmp.sub(resampled_rv) ** 2).sub(best_model).dropna(axis=1, how='all').mean(axis=1).dropna()
            dm_test = pd.DataFrame(dm_test).dropna()
            final_dm_results[idx] = dm_test.mean().div(dm_test.std())
        final_dm_results = pd.concat(final_dm_results).unstack(0).rename(
            columns={idx: f'$\mathcal{{M}}^{{{L.lower()}}}$' for idx,
            L in enumerate(['1W', '1M', '6M'])}).fillna(0).round(6)
        final_dm_results = final_dm_results.reset_index(level=list(range(0, 4)))
        final_dm_results['level_0'] = final_dm_results['level_0'].str.upper()
        final_dm_results['level_1'] = final_dm_results['level_1'].str.upper()
        final_dm_results['level_1'] = final_dm_results['level_1'].str.replace('LINEAR', 'LR')
        final_dm_results['level_1'] = final_dm_results['level_1'].str.replace('ELASTIC', 'Elastic')
        final_dm_results['level_1'] = final_dm_results['level_1'].str.replace('RIDGE', 'Ridge')
        final_dm_results['level_1'] = final_dm_results['level_1'].str.replace('LIGHTGBM', 'LightGBM')
        final_dm_results['level_0'] = final_dm_results['level_0'].str.replace('_EQ', '-eq')
        final_dm_results['level_0'] = final_dm_results['level_0'].str.replace('RISK_METRICS', 'RiskMetrics')
        final_dm_results['level_2'] = final_dm_results['level_2'].str.replace('NaN', '')
        final_dm_results['level_3'] = final_dm_results['level_3'].str.replace('NaN', '')
        final_dm_results.loc[(final_dm_results['level_0'] == 'HAR-eq') &
                             (final_dm_results['level_3'].isin(['0', '1'])), 'level_3'] = 'VIXM'
        final_dm_results['level_0'] = ['-'.join((row['level_0'], row['level_3'])) if
                                       row['level_3'] != '' else row['level_0'] for _, row in
                                       final_dm_results[['level_0', 'level_3']].iterrows()]
        final_dm_results = final_dm_results.drop(['level_2', 'level_3'], axis=1)
        final_dm_results = final_dm_results.assign(training_scheme='SAM').rename(columns={'level_0': 'model',
                                                                                          'level_1': 'regression'})
        final_dm_results = final_dm_results.set_index(['training_scheme', 'model', 'regression']).sort_index(level=1)
        print(final_dm_results.applymap(lambda x: f'\\textbf{{{x}}}' if x < 0 else x).to_latex())


class Commonality:

    _reader_obj = Reader()
    _rv = _reader_obj.rv_read()
    _mkt = _rv.mean(axis=1)
    _commonality_connect_db = sqlite3.connect(database=os.path.abspath(f'{_data_centre_dir}/commonality.db'))
    _linear = LinearRegression()
    _commonality_per_freq_dd = dict([(symbol, None) for symbol in _rv.columns])
    _factory_transformation_dd = {'log': {'transformation': np.log, 'inverse': np.exp},
                                  None: {'transformation': lambda x: x, 'inverse': lambda x: x}}
    _commonality_df = None
    _figures_dir = os.path.abspath(__file__).replace('/model/lab.py', '/figures')

    def __init__(self, title_figure: bool, L: str = None, transformation: str = 'log',
                 type_commonality: str = 'adjr2'):
        self._title_figure = title_figure
        self._type_commonality = type_commonality
        self._L = L
        self._transformation = transformation
        Commonality._rv = \
            Commonality._factory_transformation_dd[self._transformation]['transformation'](Commonality._rv)
        self._rv_fitted = pd.DataFrame(data=np.nan, index=Commonality._rv.index, columns=Commonality._rv.columns)
        Commonality._commonality_df = pd.DataFrame(data=np.nan, index=list(Commonality._rv.resample('D').groups.keys()),
                                                   columns=Commonality._rv.columns)

    @property
    def L(self):
        return self._L

    @property
    def type_commonality(self):
        return self._type_commonality

    @L.setter
    def L(self, L: str):
        self._L = L

    @type_commonality.setter
    def type_commonality(self, type_commonality: str):
        self._type_commonality = type_commonality

    @staticmethod
    def ols_closed_formula(df: pd.DataFrame) -> pd.DataFrame:
        y = df['RV'].values
        X = df.loc[:, ~df.columns.str.fullmatch('RV')].values
        XtX = np.matmul(X.transpose(), X)
        Xty = np.matmul(X.transpose(), y)
        inv_XtX = np.linalg.inv(XtX)
        b = pd.DataFrame(np.matmul(inv_XtX, Xty), index=[r'$RV_{M}$', 'intercept'],
                         columns=[df.index.get_level_values(0).unique()[0]]).transpose()
        return b

    def adj_r2_derive(self) -> pd.DataFrame:
        dates = np.unique(self._rv.index.date).tolist()
        start_idx = dates.index(dates[0] + relativedelta(days={'1W': 7, '1M': 30, '6M': 180}[self._L]))
        tmp = pd.Series(index=dates[start_idx:])
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(lambda x: x, x=date) for date in dates[start_idx:]]
            for future in concurrent.futures.as_completed(futures):
                date = future.result()
                adj_r2 = self._rv.loc[
                         (date - relativedelta(days={'1W': 7, '1M': 30, '6M': 180}[self._L])).strftime(FORMAT):
                         (date + relativedelta(days=1)).strftime(FORMAT)]
                mkt = adj_r2.mean(axis=1)
                mkt.name = '$RV_{M}$'
                adj_r2 = pd.melt(adj_r2, var_name='symbol', value_name='RV', ignore_index=False)
                adj_r2 = adj_r2.join(mkt)
                adj_r2 = adj_r2.assign(intercept=1.0)
                adj_r2.index.name = 'timestamp'
                adj_r2 = adj_r2.reset_index().set_index(['symbol', 'timestamp']).sort_index(
                    level=['symbol', 'timestamp'])
                adj_r2_group = adj_r2.groupby(by=[pd.Grouper(level='symbol')])
                coef = adj_r2_group.apply(lambda x: self.ols_closed_formula(x)).droplevel(axis=0,
                                                                                          level=-1).reset_index()
                coef = adj_r2.reset_index(level=0).merge(coef, left_on='symbol', right_on='symbol').filter(regex=f'_y')
                coef.columns = coef.columns.str.replace('_y', '')
                coef = coef.assign(symbol=adj_r2.index.get_level_values(0), timestamp=adj_r2.index.get_level_values(1))
                coef = coef.set_index(['symbol', 'timestamp'])
                adj_r2.loc[:, 'RV_hat'] = adj_r2.loc[:, coef.columns].mul(coef).sum(axis=1)
                adj_r2 = adj_r2.filter(regex=f'RV').drop(axis=1, columns='$RV_{M}$')
                adj_r2 = self._factory_transformation_dd[self._transformation]['transformation'](adj_r2)
                adj_r2 = adj_r2.groupby(by=[pd.Grouper(level='symbol'), pd.Grouper(level='timestamp', freq='D')])
                adj_r2 = \
                    adj_r2.apply(lambda x: 1-((1 - (r2_score(x.iloc[:, 0], x.iloc[:, -1]))) * (x.shape[0] - 1) / (
                            x.shape[0] - 2)))
                tmp.loc[date.strftime(FORMAT)] = adj_r2.groupby(by=[pd.Grouper(level='symbol')]).mean().mean()
        return tmp

    def adj_r2(self) -> None:
        adj_r2_dd = dict()
        for L in ['1W', '1M', '6M']:
            if L != self._L:
                self.L = L
            print(f'[Commonality]: Computation (Adjusted R2, {self._L.lower()}) has started...')
            adj_r2_dd[L] = self.adj_r2_derive()
            print(f'[Commonality]: (Adjusted R2, {self._L.lower()}) has been computed.')
        adj_r2 = pd.DataFrame(adj_r2_dd).dropna()
        adj_r2.index = pd.to_datetime(adj_r2.index, utc=True)
        adj_r2 = adj_r2.resample('1W').mean()
        adj_r2.columns = [f'${L.lower()}$' for L in adj_r2.columns]
        adj_r2 = pd.melt(adj_r2, var_name=r'$L_{train}$', value_name='adj_r2', ignore_index=False)
        fig = px.line(data_frame=adj_r2, y='adj_r2', color=r'$L_{train}$',
                      category_orders={r'$L_{train}$': [r'$1w$', r'$1m$', r'$6m$']})
        if self._title_figure:
            fig.update_layout(title=dict(text='Commonality (type 1)'))
        fig.update_yaxes(title=r'$\text{Adjusted } R^2$')
        fig.update_xaxes(title='Date', tickangle=45)
        fig.update_layout(title=dict(font=dict(size=TITLE_FONT_SIZE)),
                          width=WIDTH, height=HEIGHT, font=dict(size=LABEL_AXIS_FONT_SIZE),
                          legend=dict(orientation='h', y=1, x=.75, title=None))
        fig.show()

    def absorption_ratio_derive(self, all_eigenvectors: bool = True) -> pd.DataFrame:
        dates = np.unique(self._rv.index.date).tolist()
        start_idx = dates.index(dates[0]+relativedelta(days={'1W': 7, '1M': 30, '6M': 180}[self._L]))
        tmp = pd.Series(index=dates[start_idx:])
        for idx, date in enumerate(dates[start_idx:]):
            cov = \
                self._rv.loc[
                (date-relativedelta(days={'1W': 7, '1M': 30, '6M': 180}[self._L])).strftime(FORMAT):
                (date-relativedelta(days=1)).strftime(FORMAT)].cov()
            eigen, rank = np.linalg.eig(cov.values), np.linalg.matrix_rank(cov.values)
            n_eigenvectors = rank if all_eigenvectors else rank//5
            eigenvalues = eigen[0][:n_eigenvectors]
            tmp.loc[date.strftime(FORMAT)] = eigenvalues.sum()/np.diag(cov).sum()
        return tmp

    def absorption_ratio(self, all_eigenvectors: bool = False) -> None:
        absorption_ratio_dd = dict()
        for L in ['1W', '1M', '6M']:
            if L != self._L:
                self.L = L
            print(f'[Commonality]: Computation (Absorption Ratio, {self._L.lower()}) has started...')
            absorption_ratio_dd[L] = self.absorption_ratio_derive(all_eigenvectors)
            print(f'[Commonality]: (Absorption Ratio, {self._L.lower()}) has been computed.')

        absorption_ratio = pd.DataFrame(absorption_ratio_dd).dropna()
        absorption_ratio.index = pd.to_datetime(absorption_ratio.index, utc=True)
        absorption_ratio = absorption_ratio.resample('1W').mean()
        absorption_ratio = \
            pd.melt(absorption_ratio, var_name='$L_{train}$', value_name='abs_ratio', ignore_index=False)
        absorption_ratio['$L_{train}$'] = absorption_ratio['$L_{train}$'].apply(lambda x: f'${x.lower()}$')
        fig = px.line(data_frame=absorption_ratio, y='abs_ratio', color='$L_{train}$')
        if self._title_figure:
            fig.update_layout(title=dict(text='Commonality (type 2)'))
        fig.update_yaxes(title='Absorption ratio')
        fig.update_xaxes(title='Date', tickangle=45)
        fig.update_layout(title=dict(font=dict(size=TITLE_FONT_SIZE)),
                          width=WIDTH, height=HEIGHT, font=dict(size=LABEL_AXIS_FONT_SIZE),
                          legend=dict(title=None, orientation='h', y=1, x=.75))
        fig.show()

    def commonality(self) -> None:
        if self._type_commonality == 'absorption_ratio':
            self.absorption_ratio()
        elif self._type_commonality == 'adjusted_r2':
            self.adj_r2()
        else:
            raise TypeError('Provide a valid commonality type: Absorption ratio or adjusted R2.')


class DL_Model:

    def __init__(self):
        self._scaling_first = None
        self._scaling_second = None
        self._scaling_third = None
        self._scaling_name = None

    @property
    def scaling_first(self) -> pd.DataFrame:
        return self._scaling_first

    @scaling_first.setter
    def scaling_first(self, scaling_first: pd.DataFrame) -> pd.DataFrame:
        self._scaling_first = scaling_first

    @property
    def scaling_second(self) -> pd.DataFrame:
        return self._scaling_second

    @scaling_second.setter
    def scaling_second(self, scaling_second: pd.DataFrame) -> None:
        self._scaling_second = scaling_second

    @property
    def scaling_third(self) -> pd.DataFrame:
        return self._scaling_third

    @scaling_first.setter
    def scaling_third(self, scaling_third: pd.DataFrame) -> None:
        self._scaling_third = scaling_third

    @property
    def scaling_name(self) -> str:
        return self._scaling_name

    @scaling_name.setter
    def scaling_name(self, scaling_name: str) -> None:
        self._scaling_name = scaling_name


class LSTM_NNModel(nn.Module, DL_Model):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, lr: float,
                 dropout_rate: float, batch_size: int, mlp_hidden_size: int, mlp_num_layers: int, output_size: int = 1):
        super(DL_Model, self).__init__()
        super(LSTM_NNModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout_rate)
        if mlp_hidden_size:
            mlp_layers = list()
            mlp_layers.append(nn.Linear(in_features=self.lstm.hidden_size, out_features=mlp_hidden_size))
            for i in range(0, mlp_num_layers-2):
                if i % 2 == 0:
                    mlp_layers.append(nn.ReLU())
                else:
                    mlp_layers.append(nn.Linear(in_features=mlp_hidden_size, out_features=mlp_hidden_size))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Linear(in_features=mlp_hidden_size, out_features=output_size))
            self.last_layer = nn.Sequential(*mlp_layers)
        else:
            self.last_layer = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.lr = lr
        self.batch_size = batch_size
        # Initialisation
        nn.init.xavier_uniform_(self.lstm.weight_ih_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)
        nn.init.constant_(self.lstm.bias_ih_l0, 0)
        nn.init.constant_(self.lstm.bias_hh_l0, 0)
        self.train_loss_dd = dict()
        self.valid_loss_dd = dict()

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(DEVICE)
        out, _ = self.lstm(x, (h0, c0))
        out = self.last_layer(out)
        return out


def reshape_dataframe(date: datetime, df: pd.DataFrame, model: typing.Union[LSTM_NNModel])\
        -> Tuple[datetime, torch.tensor]:
    n = df.index.get_level_values(0).unique().shape[0]
    data = df.loc[df.index.get_level_values(1).date == date, :]
    if model.scaling_name == 'maxabs':
        data.loc[:, data.columns.str.contains('|'.join(['RV', r'^VIXM']))] = \
            data.loc[:, data.columns.str.contains('|'.join(['RV', r'^VIXM']))].div(model.scaling_first)
    data = torch.from_numpy(data.values.reshape((n, data.shape[0]//n, data.shape[-1])))
    return date, data.type(torch.float32)


def train_model(train: DataLoader, valid: DataLoader, EPOCH: int, model: typing.Union[LSTM_NNModel],
                early_stopping_patience: int = 5, save_loss: bool = False) -> typing.Union[LSTM_NNModel]:
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=model.lr)
    best_val_loss = np.inf
    epochs_without_improvement = 0
    for epoch in range(EPOCH):
        for train_batch, valid_batch in zip(train, valid):
            model.train()
            optimizer.zero_grad()
            train_outputs = model(train_batch[:, :, 1:])
            train_loss = criterion(train_outputs[:, :, 0].view(-1, 1), train_batch[:, :, 0].view(-1, 1))
            if save_loss:
                if model.train_loss_dd.get(epoch) is None:
                    model.train_loss_dd[epoch] = [train_loss.item()]
                else:
                    model.train_loss_dd[epoch].append(train_loss.item())
            train_loss.backward()
            optimizer.step()
            valid_outputs = model(valid_batch[:, :, 1:])
            valid_loss = criterion(valid_outputs[:, :, 0].view(-1, 1), valid_batch[:, :, 0].view(-1, 1))
            if save_loss:
                if model.valid_loss_dd.get(epoch) is None:
                    model.valid_loss_dd[epoch] = [valid_loss.item()]
                else:
                    model.valid_loss_dd[epoch].append(valid_loss.item())
        # Early stopping
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= early_stopping_patience:
            print(f'Early stopping after {epoch}/{EPOCH} epochs without improvement.')
            print(f'Training loss: {train_loss.item():.4f}, Valid loss: {valid_loss.item():.4f}')
            break
        if (epoch % 10 == 0) & (epoch > 0):
            print(
                f'Epoch [{epoch}/{EPOCH}], Training loss: {train_loss.item():.4f}, '
                f'Valid loss: {valid_loss.item():.4f}'
            )


class VAR_Model:

    def __init__(self):
        pass

    @staticmethod
    def var_train(idx: int, date: datetime, df: pd.DataFrame, L: str, transformation: str,
                  factory_transformation: dict, r: bool=False, **kwargs) -> \
            typing.Union[typing.Tuple[datetime, ro.vectors.ListVector], typing.Tuple[datetime, None],
            typing.Tuple[datetime, tsa.vector_ar.var_model.VARResultsWrapper]]:
        var_model = None
        if (idx == 0) | (training_freq(date, kwargs['freq'])):
            print(f'[Training model]: Training on {date.strftime("%Y-%m-%d")} has started...')
            train = df.loc[(date - relativedelta(
                days={'1W': 7, '1M': 30, '6M': 180}[L])).strftime('%Y-%m-%d'):(
                                   date - relativedelta(days=1)).strftime('%Y-%m-%d'), :]
            print(f'[Training model]: Training on {date.strftime("%Y-%m-%d")} is now complete.')
            if r:
                pd.DataFrame.iteritems = pd.DataFrame.items
                train_r = \
                    pandas2ri.py2rpy_pandasdataframe(factory_transformation[transformation]['transformation'](train))
                var_model = var.VAR(train_r, p=1, type="const")
            else:
                var_model = VAR(factory_transformation[transformation]['transformation'](train))
                try:
                    var_model = var_model.fit(1)
                except ValueError:
                    var_model = var_model.fit(1, trend='n')
        else:
            print(f'No training on {date.strftime("%Y-%m-%d")}...')
        return date, var_model

    @staticmethod
    def var_forecast(var_model: tsa.vector_ar.var_model.VARResultsWrapper, date: datetime,
                     df: pd.DataFrame, L: str, freq: str, n_head: int = 288) -> pd.DataFrame:
        start_var_idx = (date - relativedelta(days={'1W': 7, '1M': 30, '6M': 180}[L])).strftime(
            '%Y-%m-%d')
        end_var_idx = (date - relativedelta(days=1)).strftime('%Y-%m-%d')
        y = \
            pd.DataFrame(
                data=var_model.forecast(df.loc[start_var_idx:end_var_idx, :].values, n_head),
                index=pd.date_range(start=date, end=date + relativedelta(days=1), freq=freq,
                                    tz=pytz.utc, inclusive='left'),
                columns=df.columns)
        return y


class SpilloverEffect:

    def __init__(self, forecast: pd.DataFrame = None):
        self._g_spillover = None
        self._forecast = forecast if forecast is None else forecast.copy()
        self._spillover_network_dd = dict()
        self._net_transmitter_sender_dd = dict()
        self._spillover_index_dd = dict()

    @property
    def forecast(self) -> pd.DataFrame:
        return self._forecast

    @property
    def g_spillover(self) -> pd.DataFrame:
        self._g_spillover = self.spillover()
        return self._g_spillover

    @property
    def spillover_network_dd(self) -> typing.Dict:
        return self._spillover_network_dd

    @property
    def net_transmitter_sender_dd(self) -> typing.Dict:
        return self._net_transmitter_sender_dd

    @property
    def spillover_index_dd(self) -> typing.Dict:
        return self._spillover_index_dd

    @forecast.setter
    def forecast(self, forecast: pd.DataFrame) -> None:
        self._forecast = forecast

    def spillover(self) -> pd.DataFrame:
        pd.DataFrame.iteritems = pd.DataFrame.items
        train_r = ro.pandas2ri.py2rpy_pandasdataframe(self._forecast)
        vars_r = var.VAR(train_r, p=1, type="const")
        n_ahead = pd.to_timedelta(self._forecast.index[1]-self._forecast.index[0])//pd.to_timedelta('5T')
        g_spillover = spillover.G_spillover(vars_r, n_ahead=n_ahead, standardized="TRUE")
        g_spillover = pd.DataFrame(data=g_spillover, index=self._forecast.columns.tolist() + ['To', 'Net'],
                                   columns=self._forecast.columns.tolist() + ['From']).div(100)
        return g_spillover

    def spillover_matrix(self) -> np.ndarray:
        return self.g_spillover.iloc[:-2, :-1]

    def spillover_type(self) -> np.ndarray:
        return self.g_spillover.iloc[-1, :-1]

    def spillover_index(self) -> float:
        return self.spillover().iloc[-2, -1]
