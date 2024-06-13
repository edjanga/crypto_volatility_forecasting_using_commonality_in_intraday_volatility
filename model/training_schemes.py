import concurrent.futures
import pdb
import pickle
import typing
from typing import Tuple, Union
import lightgbm
import pandas as pd
import lightgbm as lgb
from datetime import datetime
import os
from dateutil.relativedelta import relativedelta
from data_centre.helpers import coin_ls
import sqlite3
from model.feature_engineering_room import FeatureAR, FeatureHAR, FeatureRiskMetrics, FeatureHAREq
import numpy as np
import optuna
from model.lab import training_freq, train_model, split_train_valid_set, lags, scaling, inverse_scaling,\
    LSTM_NNModel, qlike_score, reshape_dataframe, VAR_Model
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from optuna.samplers import RandomSampler
from functools import partial
import pytz
from data_centre.data import Reader
DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'
"""
    List of constant variables used in the script
"""
NUM_BOOST_ROUND = 100
STOPPING_ROUNDS = 5
MAX_ITER = 10
N_TRIALS = 5
N_JOBS = -1
EPOCH = 100
EARLY_STOPPING_PATIENCE = 5


class TrainingScheme(object):
    _reader_obj = Reader()
    _data_centre_dir = os.path.abspath(__file__).replace('/model/training_schemes.py', '/data_centre')
    _db_connect_qlike = sqlite3.connect(database=os.path.abspath(f'{_data_centre_dir}/databases'
                                                                 f'/qlike.db'),
                                        check_same_thread=False)
    _db_connect_y = sqlite3.connect(database=os.path.abspath(f'{_data_centre_dir}/databases/y.db'),
                                    check_same_thread=False)
    _factory_model_type_dd = {'ar': FeatureAR(), 'har': FeatureHAR(), 'risk_metrics': FeatureRiskMetrics(),
                              'har_eq': FeatureHAREq(), None: None}

    _factory_transformation_dd = {'log': {'transformation': np.log, 'inverse': np.exp},
                                  None: {'transformation': lambda x: x, 'inverse': lambda x: x}}
    global coins_copy
    coins = [''.join((coin, 'usdt')).upper() for _, coin in enumerate(coin_ls)]
    global coins_dd
    coins_dd = {coin: {} for coin in coins}
    _training_scheme_y_dd = coins_dd.copy().copy()
    _training_scheme_qlike_dd = coins_dd.copy().copy()
    L_shift_dd = {'5T': pd.to_timedelta('5T') // pd.to_timedelta('5T'),
                  '30T': pd.to_timedelta('30T') // pd.to_timedelta('5T'),
                  '1H': pd.to_timedelta('1H') // pd.to_timedelta('5T'),
                  '6H': pd.to_timedelta('6H') // pd.to_timedelta('5T'),
                  '12H': pd.to_timedelta('12H') // pd.to_timedelta('5T'),
                  '1D': pd.to_timedelta('1D') // pd.to_timedelta('5T'),
                  '1W': pd.to_timedelta('1W') // pd.to_timedelta('5T'),
                  '1M': pd.to_timedelta('30D') // pd.to_timedelta('5T'),
                  '6M': pd.to_timedelta('180D') // pd.to_timedelta('5T')}

    def __init__(self, h: str, L: str, Q: str, universe: typing.List[str], model_type: str=None,
                 s: str=None, b: str='5T', F: typing.List[str] = None):
        """
            h: Forecasting horizon
            s: Sliding window
            F: Feature building lookback
            L: Lookback window for model training
            Q: Model update frequency
        """
        self._s = h if s is None else s
        self._h = h
        self._F = F
        self._L = L
        self._Q = Q
        self._b = b
        self._L = L
        self._universe = universe
        self._feature_importance = dict()
        self._feature_importance_symbol = dict([(symbol, None) for symbol in universe])
        if model_type not in TrainingScheme._factory_model_type_dd.keys():
            raise ValueError(f'{model_type} is not part of the model catalogue. Unknown model type.')
        else:
            self._model_type = model_type

    @property
    def s(self):
        return self._s

    @property
    def h(self):
        return self._h

    @property
    def F(self):
        return self._F

    @property
    def L(self):
        return self._L

    @property
    def Q(self):
        return self._Q

    @property
    def model_type(self):
        return self._model_type

    @property
    def factory_transformation_dd(self):
        return self._factory_transformation_dd

    @s.setter
    def s(self, s: str):
        self._s = s

    @h.setter
    def h(self, h: str):
        self._h = h

    @F.setter
    def F(self, F: typing.List[str]):
        max_feature_building_lookback = \
            max([pd.to_timedelta(lookback) // pd.to_timedelta(self._b) if 'M' not in lookback else
                 pd.to_timedelta(''.join((str(30 * int(lookback.split('M')[0])), 'D'))) // pd.to_timedelta(self._b)
                 for _, lookback in enumerate(F)])
        upper_bound_feature_building_lookback = \
            pd.to_timedelta(''.join((str(30 * int(self.L.split('M')[0])), 'D'))) // pd.to_timedelta(self._b) \
                if 'M' in self._L else pd.to_timedelta(self._L) // pd.to_timedelta(self._b)
        if upper_bound_feature_building_lookback < max_feature_building_lookback:
            raise ValueError('Lookback window for model training is smaller than the furthest lookback window '
                             'in feature building.')
        else:
            self._F = F

    @L.setter
    def L(self, L: str):
        max_feature_building_lookback = \
            max([pd.to_timedelta(lookback) // pd.to_timedelta(self._b) if 'M' not in lookback else
                 pd.to_timedelta(''.join((str(30 * int(lookback.split('M')[0])), 'D'))) // pd.to_timedelta(self._b)
                 for _, lookback in enumerate(self._F)])
        upper_bound_feature_building_lookback = \
            pd.to_timedelta(''.join((str(30 * int(L.split('M')[0])), 'D'))) // pd.to_timedelta(self._b) \
                if 'M' in L else pd.to_timedelta(L) // pd.to_timedelta(self._b)
        if upper_bound_feature_building_lookback < max_feature_building_lookback:
            raise ValueError('Lookback window for model training is smaller than the furthest lookback window '
                             'in feature building.')
        else:
            self._L = L

    @Q.setter
    def Q(self, Q: str):
        self._Q = Q

    @model_type.setter
    def model_type(self, model_type: str):
        self._model_type = model_type

    def build_exog(self, df: pd.DataFrame, transformation: str, regression_type: str = 'linear',
                   **kwargs) -> pd.DataFrame:
        pass

    @staticmethod
    def volatility_period(df: pd.DataFrame) -> pd.Series:
        rv_mkt = pd.DataFrame(df.mean(axis=1).rename('RV_MKT')).resample('1D').sum()
        rv_mkt = rv_mkt.assign(vol_regime=
                               pd.cut(
                                   rv_mkt.RV_MKT,
                                   bins=[0, rv_mkt.RV_MKT.quantile(.45), rv_mkt.RV_MKT.quantile(.9),
                                         rv_mkt.RV_MKT.quantile(1)],
                                   labels=['low', 'normal', 'high']))
        rv_mkt.drop('RV_MKT', axis=1, inplace=True)
        rv_mkt.index = pd.to_datetime(rv_mkt.index, utc=True)
        rv_mkt.index.name = 'timestamp'
        return rv_mkt

    @staticmethod
    def objective(trial: optuna.trial.Trial, X_train: pd.DataFrame, X_valid: pd.DataFrame, y_train: pd.DataFrame,
                  y_valid: pd.DataFrame, init: bool, idx: int):
        regression_type = trial.study.study_name.split('_')[3]
        if regression_type == 'lightgbm':
            date = trial.study.study_name.split('_')[-1]
            date_prev = datetime.strptime(trial.study.study_name.split('_')[-1], '%Y-%m-%d')-relativedelta(days=idx)
            date_prev = date_prev.strftime('%Y-%m-%d')
            tmp_model_dir = '../model/tmp'
            tmp_model_path = '/'.join((tmp_model_dir, trial.study.study_name))
            prev_tmp_model_path = '/'.join((tmp_model_dir, trial.study.study_name.replace(date, date_prev)))
            if not os.path.exists(os.path.relpath(start='.', path=tmp_model_dir)):
                os.makedirs(tmp_model_dir)
            train_loader, valid_loader = lgb.Dataset(X_train, label=y_train), lgb.Dataset(X_valid, label=y_valid)
            param = {
                'objective': 'regression', 'metric': 'mse', 'verbosity': -1,
                'max_depth': trial.suggest_int('max_depth', 1, 3),
                'lr': trial.suggest_categorical(name='lr', choices=[.01, .05, .1]),
                'tree_learner': 'data',
                'feature_fraction': trial.suggest_float('feature_fraction', .1, .8, step=.1, log=False),
                'extra_tree': True, 'boosting_type': 'goss'
            }
            tmp_rres = lgb.train(param, train_set=train_loader, valid_sets=[valid_loader],
                                 num_boost_round=NUM_BOOST_ROUND,
                                 callbacks=[lgb.early_stopping(STOPPING_ROUNDS,
                                                               first_metric_only=True, verbose=True, min_delta=0.0)],
                                 init_model={True: os.path.relpath(start='.', path=prev_tmp_model_path),
                                             False: None}[init])
        elif regression_type == 'lasso':
            param = {'objective': 'regression', 'metric': 'mse', 'verbosity': -1,
                     'alpha': trial.suggest_float('alpha', 1e-4, 1e-1, log=False)
                     }
            tmp_rres = Lasso(alpha=param['alpha'], fit_intercept=True, random_state=123, max_iter=MAX_ITER,
                             warm_start=True, selection='random')
        elif regression_type == 'elastic':
            param = {'objective': 'regression', 'metric': 'mse', 'verbosity': -1,
                     'alpha': trial.suggest_float('alpha', 1e-4, 1e-1, log=False),
                     'l1_ratio': trial.suggest_float('l1_ratio', .01, .99, log=False)
                     }
            tmp_rres = ElasticNet(alpha=param['alpha'], l1_ratio=param['l1_ratio'], fit_intercept=True,
                                  max_iter=MAX_ITER, random_state=123, selection='random')
        elif regression_type == 'ridge':
            param = {'alpha': trial.suggest_float('alpha', 1e-4, 1e-1, log=False)}
            tmp_rres = Ridge(alpha=param['alpha'], max_iter=MAX_ITER, fit_intercept=True)
        if regression_type != 'lightgbm':
            tmp_rres.fit(X_train, y_train)
        loss = mean_squared_error(y_valid, tmp_rres.predict(X_valid))
        trial.set_user_attr(key='best_estimator', value=tmp_rres)
        if regression_type == 'lightgbm':
            if not init:
                trial.user_attrs['best_estimator'].save_model(tmp_model_path)
        return loss

    @staticmethod
    def callback(study, trial):
        if study.best_trial.number == trial.number:
            study.set_user_attr(key='best_estimator', value=trial.user_attrs['best_estimator'])

    def rolling_metrics_per_date(self, date: datetime, symbol: str, regression_type: str,
                                 exog: pd.DataFrame, endog: pd.DataFrame, transformation: str, idx: int = 0,
                                 **kwargs) -> Union[pd.Series, Tuple[datetime, object]]:
        if self._model_type == 'risk_metrics':
            model_obj = self._factory_model_type_dd[self._model_type]
            print(f'[End of Training]: Training on {symbol}-{(date+relativedelta(days=idx)).strftime("%Y-%m-%d")} '
                  f'has been completed.')
            yt_hat = model_obj.predict(y=endog.loc[((date+relativedelta(days=idx)) - relativedelta(
                days={'1W': 7, '1M': 30, '6M': 180}[self._L])).strftime('%Y-%m-%d'):
                                                   (date+relativedelta(days=idx)).strftime('%Y-%m-%d')],
                                       date=(date+relativedelta(days=idx)))
            return yt_hat
        else:
            train_index = list(set(exog.index[(exog.index.date >= date + relativedelta(days=idx) - L_train) &
                                              (exog.index.date < date + relativedelta(days=idx))]))
            train_index.sort()
            test_index = list(set(exog.index[exog.index.date == date + relativedelta(days=idx)]))
            test_index.sort()
            valid_index = train_index[-len(train_index) // 5:]
            train_index = train_index[:-len(train_index) // 5]
            X_valid, y_valid = exog.loc[exog.index.isin(valid_index), :], endog.loc[endog.index.isin(valid_index)]
            X_train, y_train = exog.loc[exog.index.isin(train_index), :], endog.loc[endog.index.isin(train_index)]
            X_test, y_test = exog.loc[exog.index.isin(test_index), :], endog.loc[endog.index.isin(test_index)]
            if regression_type not in ['linear', 'pcr']:
                study_name = f'{self.__class__.__name__}_{self._L}_{transformation}_{regression_type}_' \
                             f'{self._model_type}_{symbol}_{(date+relativedelta(days=idx)).strftime("%Y-%m-%d")}'
                study = optuna.create_study(direction='minimize', study_name=study_name, sampler=RandomSampler(123))
                objective = partial(self.objective, X_train=X_train, X_valid=X_valid, y_train=y_train, y_valid=y_valid,
                                    init=not (idx == 0), idx=idx)
                study.optimize(objective, n_trials=N_TRIALS, callbacks=[self.callback], n_jobs=N_JOBS)
                rres = study.user_attrs['best_estimator']
                return date, rres
            else:
                tmp_model_dir = '../model/tmp'
                rres = LinearRegression()
                if regression_type == 'pcr':
                    rres_pca = PCA()
                    X_train = X_train.replace(np.inf, np.nan)
                    X_train = X_train.fillna(X_train.ewm(span=12, min_periods=1).mean())
                    if idx == 0:
                        tmp_model_path = '_'.join((self.__class__.__name__, self._L, str(transformation),
                                                   regression_type, self._model_type, symbol,
                                                   date.strftime('%Y-%m-%d')))
                        tmp_model_path = os.path.relpath(start='.', path='/'.join((tmp_model_dir, tmp_model_path)))
                        rres_pca.fit(X_train.loc[:, ~X_train.columns.str.contains(symbol)].values)
                        explained_variance_ratio = list(np.cumsum(rres_pca.explained_variance_ratio_) >= .9)
                        n_components = explained_variance_ratio.index(True) + 1
                        rres_pca.n_components = n_components
                        if not os.path.exists(os.path.relpath(start='.', path=tmp_model_dir)):
                            os.makedirs(tmp_model_dir)
                        with open(tmp_model_path, 'wb') as f:
                            pickle.dump(rres_pca, f)
                    else:
                        prev_tmp_model_path = '_'.join((self.__class__.__name__, self._L, str(transformation),
                                                        regression_type, self._model_type, symbol,
                                                        date.strftime('%Y-%m-%d')))
                        prev_tmp_model_path = os.path.relpath(start='.',
                                                              path='/'.join((tmp_model_dir, prev_tmp_model_path)))
                        with open(prev_tmp_model_path, 'rb') as f:
                            rres_pca = pickle.load(f)
                    own_train, own_test = pd.DataFrame(), pd.DataFrame()
                    """If cluster group contains only more than one member"""
                    if self.__class__.__name__ == 'ClustAM':
                        if len(kwargs['cluster']) > 1:
                            own_train, own_test = \
                                X_train.loc[:, X_train.columns.str.contains(symbol)].loc[train_index, :],\
                                    X_test.loc[:, X_test.columns.str.contains(symbol)].loc[test_index, :]
                    X_train = \
                        pd.DataFrame(rres_pca.transform(X_train.loc[:, ~X_train.columns.str.contains(symbol)].values),
                                     index=X_train.index)
                    X_train = pd.concat([X_train, own_train], axis=1)
                rres.fit(X_train.values, y_train)
                pipeline = Pipeline([('pca', rres_pca), ('linear', rres)]) if regression_type == 'pcr' else rres
                print(f'[End of Training]: Training on {symbol}-{date.strftime("%Y-%m-%d")} has been completed.') \
                    if idx == 0 else \
                    print(f'[End of fine tuning]: Fine tuning on '
                          f'{symbol}-{(date+relativedelta(days=idx)).strftime("%Y-%m-%d")} '
                          f'has been completed.')
                return date, pipeline

    def rolling_metrics(self, exog: pd.DataFrame, agg: str, transformation: str,
                        regression_type: str = 'linear', **kwargs) -> typing.Union[None, typing.Tuple[pd.Series]]:
        global L_train
        global endog
        global factor_obj
        if kwargs.get('symbol'):
            symbol = kwargs.get('symbol')
        else:
            symbol = 'RV'
        if (self._model_type == 'har_eq') & (kwargs.get('trading_session') == 1):
            exog.loc[:, ~exog.columns.str.contains('_SESSION')] = \
                self._factory_transformation_dd[transformation]['transformation'](
                    exog.loc[:, ~exog.columns.str.contains('_SESSION')])
        else:
            exog = self._factory_transformation_dd[transformation]['transformation'](exog)
        endog = exog.pop(symbol)
        y = list()
        global dates
        dates = list(np.unique(exog.index.date))[:28]
        L_train = relativedelta(days={'1W': 7, '1M': 30, '6M': 180}[self._L])
        start = dates[0] + L_train
        tmp_dd = dict()
        kwargs_copy = kwargs.copy()
        del kwargs_copy['symbol']
        for date in dates[dates.index(start)::L_train.days]:
            if self._model_type == 'risk_metrics':
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(self.rolling_metrics_per_date, date, symbol, regression_type,
                                               exog, endog, transformation, idx, **kwargs_copy)
                               for idx in range(0, L_train.days)]
                    for future in concurrent.futures.as_completed(futures):
                        yt = future.result()
                        y.append(yt)
            else:
                # if (self.__class__.__name__ == 'ClustAM') & (regression_type == 'pcr'):
                #     pdb.set_trace()
                # else:
                date, model = self.rolling_metrics_per_date(exog=exog, endog=endog, date=date,
                                                            regression_type=regression_type,
                                                            transformation=transformation,
                                                            symbol=symbol, idx=0, **kwargs_copy)
                tmp_dd[date] = model
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = \
                        [executor.submit(self.rolling_metrics_per_date, date, symbol, regression_type, exog, endog,
                                         transformation, idx, **kwargs_copy) for idx in
                         range(1, L_train.days)]
                    for idx, future in enumerate(concurrent.futures.as_completed(futures)):
                        date, model = future.result()
                        tmp_dd[date+relativedelta(days=idx+1)] = model
                tmp_model_path = '_'.join((self.__class__.__name__, self._L, str(transformation),
                                           regression_type, self._model_type, symbol, date.strftime('%Y-%m-%d')))
                tmp_model_dir = '../model/tmp'
                tmp_model_path = '/'.join((tmp_model_dir, tmp_model_path))
                os.remove(os.path.relpath(start='.', path=tmp_model_path))
        model_s = pd.Series(tmp_dd).sort_index()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(lambda x: x, x=date) for date in dates[dates.index(start):]]
            for future in concurrent.futures.as_completed(futures):
                date = future.result()
                if regression_type == 'pcr':
                    own = exog.loc[exog.index.date == date, exog.columns.str.contains(symbol)]
                    tmp_pca = model_s[date][0].transform(exog.loc[exog.index.date == date, ~exog.columns.str.contains(
                        symbol)].values)
                    y_hat = model_s[date][1].predict(np.hstack((tmp_pca, own)))
                else:
                    y_hat = model_s[date].predict(exog.loc[exog.index.date == date, :])
                y.append(pd.Series(name=symbol, data=y_hat,
                                   index=pd.date_range(start=date, end=date + relativedelta(days=1), freq='5T',
                                                       inclusive='left')))
        y = pd.DataFrame(pd.concat(y)).dropna()
        y.index.name = 'timestamp'
        y.index = pd.to_datetime(y.index, utc=True)
        y = pd.melt(y, ignore_index=False, var_name='symbol', value_name=0)
        y = y.join(endog, how='left').sort_index().set_index('symbol', append=True).swaplevel(-1, 0)
        y = TrainingScheme._factory_transformation_dd[transformation]['inverse'](y)
        y = y.groupby(by=[pd.Grouper(level='symbol'), pd.Grouper(level='timestamp', freq=self._h)]).sum()
        tmp = y.groupby(by=[pd.Grouper(level='symbol'), pd.Grouper(level='timestamp', freq=self._h)]) #agg
        qlike = tmp.apply(qlike_score)
        qlike = pd.Series(qlike, name=symbol)
        qlike = pd.melt(pd.DataFrame(qlike), ignore_index=False, value_name='values', var_name='symbol')
        qlike = qlike.assign(metric='qlike')
        y.reset_index(level='symbol', inplace=True)
        y.index = pd.to_datetime(y.index, utc=True)
        y = y.rename(columns={symbol: 'y', 0: 'y_hat'})
        table_dd = {'qlike': qlike, 'y': y}
        training_scheme_tables = {'qlike': self._training_scheme_qlike_dd, 'y': self._training_scheme_y_dd}
        transformation_dd = {'log': 'log', None: 'level'}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = \
                [executor.submit(lambda x, y: (x, y), x=table_name, y=table) for table_name, table in table_dd.items()]
            for future in concurrent.futures.as_completed(futures):
                table_name, table = future.result()
                table = table.assign(symbol=symbol, model=self._model_type, L=self._L,
                                     training_scheme=self.__class__.__name__, regression=regression_type,
                                     transformation=transformation_dd[transformation],
                                     trading_session=kwargs.get('trading_session'), top_book=kwargs.get('top_book'),
                                     h=self._h)
                if table_name == 'feature_importance':
                    table = table.drop('regression', axis=1)
                training_scheme_tables[table_name][symbol] = table

    def add_metrics_per_symbol(self, symbol: str, df: pd.DataFrame, agg: str, transformation: str,
                               regression_type: str = 'linear', **kwargs) -> None:
        kwargs_copy = kwargs.copy()
        del kwargs_copy['freq']
        try:
            rv_mkt = self.volatility_period(df.loc[:, ~df.columns.str.contains('VIXM')])
        except ValueError:
            rv_mkt = self.volatility_period(self._reader_obj.rv_read())
        if (self.__class__.__name__ in ['SAM', 'ClustAM']) & (self._model_type not in ['risk_metrics']):
            exog = self.build_exog(symbol=symbol, df=df, transformation=transformation, **kwargs_copy)
        else:
            exog = df.copy()
        exog = exog.replace(np.inf, np.nan)
        exog = exog.fillna(exog.ewm(span=12, min_periods=1).mean())
        transformation_dd = {'log': 'log', None: 'level'}
        self.rolling_metrics(
            symbol=symbol, regression_type=regression_type, transformation=transformation, agg=agg, exog=exog, **kwargs
        )
        y = self._training_scheme_y_dd[symbol]
        qlike = self._training_scheme_qlike_dd[symbol]
        con_dd = {'qlike': TrainingScheme._db_connect_qlike, 'y': TrainingScheme._db_connect_y}
        table_dd = {'qlike': qlike, 'y': y}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = \
                [executor.submit(lambda x, y: (x, y), x=table_name, y=table) for table_name, table in table_dd.items()]
            for count, future in enumerate(concurrent.futures.as_completed(futures)):
                table_name, table = future.result()
                if table_name != 'y':
                    table = table.drop('symbol', axis=1)
                    table = table.reset_index(level=0)
                if table_name != 'feature_importance':
                    table = table.join(rv_mkt.reindex(table.index).ffill(), how='left')
                table.to_sql(if_exists='append', con=con_dd[table_name], name=f'{table_name}_{self._L}') if \
                    table_name != 'feature_importance' else table.to_sql(if_exists='append', con=con_dd[table_name],
                                                                         name=f'{table_name}')
                if table_name != 'feature_importance':
                    print(f'[Insertion]: '
                          f'Tables for '
                          f'{self.__class__.__name__}_{self._L}_{transformation_dd[transformation]}_{regression_type}_'
                          f'{self._model_type}_{symbol}'
                          f' have been inserted into the database ({count+1})....')
                else:
                    print(f'[Insertion]: '
                          f'Table for {self.__class__.__name__}_{self._L}_{self._model_type}_{symbol}'
                          f' has been inserted into the database ({count+1})....')

    def add_metrics(self, regression_type: str, transformation: str, agg: str, df: pd.DataFrame,
                    **kwargs) -> None:
        pass

    @property
    def db_connect_pca(self):
        return self._db_connect_pca


class SAM(TrainingScheme):

    def build_exog(self, symbol: str, df: pd.DataFrame, transformation: str, **kwargs) -> pd.DataFrame:
        if self._model_type not in ['risk_metrics']:
            exog = \
                TrainingScheme._factory_model_type_dd[self._model_type].builder(F=self._F,
                                                                                df=df, symbol=symbol, **kwargs)
        else:
            exog = df[[symbol]]
        return exog

    def add_metrics(self, df: pd.DataFrame, agg: str, transformation: str, regression_type: str = 'linear',
                    **kwargs) -> None:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.add_metrics_per_symbol, symbol=symbol, transformation=transformation,
                                       regression_type=regression_type, df=df, agg=agg, **kwargs)
                       for symbol in df.columns]


class ClustAM(TrainingScheme):
    silhouette = list()
    _kmeans = KMeans(n_init='auto', random_state=123)
    _cluster_group = None
    _clusters_trained = False

    @staticmethod
    def cluster(df: pd.DataFrame) -> None:
        tmp = df.transpose().copy()
        for k in range(2, len(tmp.index) - 1):
            ClustAM._kmeans.n_clusters = k
            try:
                ClustAM._kmeans.fit(tmp)
            except ValueError:
                tmp = tmp.transpose()
                tmp = tmp.replace(np.inf, np.nan)
                tmp = tmp.fillna(tmp.ewm(span=12, min_periods=1).mean())
                tmp = tmp.transpose()
                ClustAM._kmeans.fit(tmp)
            ClustAM.silhouette.append(silhouette_score(tmp, ClustAM._kmeans.labels_))
        k_best = ClustAM.silhouette.index(max(ClustAM.silhouette)) + 2
        ClustAM._kmeans.n_clusters = k_best

    @staticmethod
    def build_clusters(df: pd.DataFrame) -> None:
        ClustAM.cluster(df)
        tmp = df.transpose().copy()
        try:
            ClustAM._kmeans.fit(tmp)
        except ValueError:
            tmp = tmp.transpose()
            tmp = tmp.replace(np.inf, np.nan)
            tmp = tmp.fillna(tmp.ewm(span=12, min_periods=1).mean())
            tmp = tmp.transpose()
            ClustAM._kmeans.fit(tmp)
        tmp.loc[:, 'labels'] = ClustAM._kmeans.labels_
        ClustAM._cluster_group = tmp.groupby(by='labels')
        ClustAM._clusters_trained = True

    def cluster_members(self, symbol: str) -> typing.Union[str, typing.List[str]]:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(lambda x, y: (x, y), x=cluster, y=members) for
                       cluster, members in ClustAM._cluster_group.groups.items()]
            for future in concurrent.futures.as_completed(futures):
                _, members = future.result()
                if symbol in members:
                    members_ls = list(members)
                    return symbol, members_ls

    def build_exog(self, symbol: typing.Union[str, typing.List[str]], df: pd.DataFrame, transformation: str, **kwargs)\
            -> typing.Union[pd.DataFrame, str]:
        _, member_ls = self.cluster_members(symbol)
        del kwargs['cluster']
        exog = \
            TrainingScheme._factory_model_type_dd[self._model_type].builder(F=self._F, df=df, symbol=member_ls,
                                                                            **kwargs)
        return exog

    def add_metrics(self, regression_type: str, transformation: str, agg: str, df: pd.DataFrame, **kwargs) -> None:
        if (self._model_type == 'har_eq') & (kwargs.get('trading_session') == 0):
            self._universe = self._universe+kwargs['vixm'].columns.tolist()
            df = TrainingScheme._factory_model_type_dd[self._model_type].builder(F=self._F, df=df, symbol=df.columns,
                                                                                 **kwargs)
        if ClustAM._cluster_group is None:
            ClustAM.build_clusters(df.loc[:, self._universe])
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = \
                [executor.submit(lambda x: self.cluster_members(x), x=symbol) for symbol in df.columns]
            for future in concurrent.futures.as_completed(futures):
                symbol, member_ls = future.result()
                self.add_metrics_per_symbol(symbol=symbol, df=df[member_ls], agg=agg, regression_type=regression_type,
                                            transformation=transformation, cluster=member_ls, **kwargs)

    @property
    def clusters_trained(self):
        return self._clusters_trained

    @property
    def cluster_group(self):
        return self._cluster_group


class CAM(TrainingScheme):
    _db_feature_importance = sqlite3.connect(
        database=os.path.abspath(f'{TrainingScheme._data_centre_dir}/databases/feature_importance.db'),
        check_same_thread=False
    )

    def __init__(self, h: str, L: str, Q: str, universe: typing.List[str], model_type: str = None,
                 s: str = None, b: str = '5T', F: [typing.List[str], None] = None):
        super(CAM, self).__init__(h=h, F=F, L=L, Q=Q, universe=universe, model_type=model_type, s=s, b=b)
        self._feature_importance = None
        self._feature_importance_symbol = dict([(symbol, None) for symbol in universe])

    def build_exog(self, df: pd.DataFrame, transformation: str = None, **kwargs) \
            -> pd.DataFrame:
        exog = TrainingScheme._factory_model_type_dd[self._model_type].builder(F=self._F, df=df, symbol=df.columns,
                                                                               **kwargs)
        return exog

    def add_metrics(self, regression_type: str, transformation: str, agg: str, df: pd.DataFrame, **kwargs) -> None:
        if (self._model_type == 'har_eq') & (kwargs.get('trading_session') == 0):
            self._universe = self._universe+kwargs['vixm'].columns.tolist()
            df = self.build_exog(df, transformation, **kwargs)
        # for symbol in self._universe:
        #     print(f'[Data Process]: Process for {symbol} has started...')
        #     self.add_metrics_per_symbol(symbol=symbol, df=df, agg=agg, regression_type=regression_type,
        #                                 transformation=transformation, **kwargs)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.add_metrics_per_symbol, symbol=symbol, transformation=transformation,
                                       regression_type=regression_type, df=df, agg=agg, **kwargs)
                       for symbol in df.columns]

    @property
    def feature_importance(self):
        return self._feature_importance

    @property
    def feature_importance_symbol(self):
        return self._feature_importance_symbol

    @feature_importance.setter
    def feature_importance(self, df: pd.DataFrame):
        self._feature_importance = df


class UAM(TrainingScheme):

    def __init__(self, h: str, L: str, Q: str, universe: typing.List[str], model_type: str = None,
                 s: str = None, b: str = '5T', F: typing.List[str] = None):
        super(UAM, self).__init__(h=h, F=F, L=L, Q=Q, universe=universe, model_type=model_type, s=s, b=b)
        self._train = None
        self._valid = None
        self._universe = None
        self._INPUT_SIZE = None
        self._regression_type = None
        self._label_encoder = None
        self.feature_importance = dict()
        self._transformation = None

    @property
    def universe(self) -> typing.List[str]:
        return self._universe

    @property
    def INPUT_SIZE(self) -> int:
        return self._INPUT_SIZE

    @property
    def regression_type(self) -> typing.Union[str, None]:
        return self._regression_type

    @universe.setter
    def universe(self, universe: typing.List[str]) -> None:
        self._universe = universe

    @INPUT_SIZE.setter
    def INPUT_SIZE(self, INPUT_SIZE: int) -> None:
        self._INPUT_SIZE = INPUT_SIZE

    @regression_type.setter
    def regression_type(self, regression_type: str) -> None:
        self._regression_type = regression_type

    def objective(self, trial: optuna.trial.Trial, train: pd.DataFrame, valid: pd.DataFrame, init: bool) -> float:
        if self._regression_type == 'lightgbm':
            y_train = train.copy().pop('RV')
            y_valid = valid.copy().pop('RV')
            date = trial.study.study_name.split('_')[-1]
            date_prev = datetime.strptime(trial.study.study_name.split('_')[-1], '%Y-%m-%d') - relativedelta(days=1)
            date_prev = date_prev.strftime('%Y-%m-%d')
            tmp_model_dir = '../model/tmp'
            tmp_model_path = '/'.join((tmp_model_dir, trial.study.study_name))
            prev_tmp_model_path = '/'.join((tmp_model_dir, trial.study.study_name.replace(date, date_prev)))
            if not os.path.exists(os.path.relpath(start='.', path=tmp_model_dir)):
                os.makedirs(tmp_model_dir)
            train_loader, valid_loader = lgb.Dataset(train.drop('RV', axis=1), label=y_train),\
                lgb.Dataset(valid.drop('RV', axis=1), label=y_valid)
            param = {
                'objective': 'regression', 'metric': 'mse', 'verbosity': -1,
                'max_depth': trial.suggest_int('max_depth', 1, 3),
                'tree_learner': 'data',
                'extra_tree': True,
                'boosting_type': 'goss',
                'lr': trial.suggest_categorical(name='lr', choices=[.01, .05, .1]),
            }
            model = \
                lgb.train(param,
                          train_set=train_loader, valid_sets=[valid_loader], num_boost_round=NUM_BOOST_ROUND,
                          callbacks=\
                              [lgb.early_stopping(STOPPING_ROUNDS, first_metric_only=True, verbose=True, min_delta=0.0)],
                          categorical_feature=train.columns[train.columns.str.contains('is')].tolist(),
                          init_model={True: os.path.relpath(start='.', path=prev_tmp_model_path),
                                      False: None}[os.path.exists(os.path.relpath(start='.',
                                                                                  path=prev_tmp_model_path))]
                          )
            trial.set_user_attr(key='best_score', value=model.best_score['valid_0']['l2'])
            trial.user_attrs['best_estimator'].save_model(tmp_model_path)
            if os.path.exists(os.path.relpath(start='.', path=prev_tmp_model_path)):
                os.remove(os.path.relpath(start='.', path=prev_tmp_model_path))
        else:
            train_date = train.index.get_level_values(1).unique().tolist()
            date = trial.study.study_name.split('_')[-1]
            date_prev = \
                datetime.strptime(trial.study.study_name.split('_')[-1], '%Y-%m-%d') - \
                relativedelta(days={'1W': 7, '1M': 30, '6M': 180}[self._L])
            date_prev = date_prev.strftime('%Y-%m-%d')
            tmp_model_dir = '../model/tmp'
            tmp_model_path = '/'.join((tmp_model_dir, trial.study.study_name))
            prev_tmp_model_path = '/'.join((tmp_model_dir, trial.study.study_name.replace(date, date_prev)))
            if not os.path.exists(os.path.relpath(start='.', path=tmp_model_dir)):
                os.makedirs(tmp_model_dir)
            params = {
                'hidden_size': trial.suggest_categorical(name='hidden_size', choices=[2 ** i for i in range(3, 6)]),
                'lr': trial.suggest_categorical(name='lr', choices=[1e-3, 5e-3, 1e-4]),
                'num_layers': trial.suggest_int(name='num_layers', low=1, high=5, step=1),
                'non_linear_last_layer': trial.suggest_categorical(name='non_linear_last_layer', choices=[True, False]),
                'batch_size': trial.suggest_categorical(name='batch_size', choices=[1]),
                'scaling': trial.suggest_categorical(name='scaling', choices=['maxabs'])
            }
            if params['non_linear_last_layer']:
                params.update({'mlp_hidden_size': trial.suggest_categorical(name='mlp_hidden_size',
                                                                            choices=[2 ** i for i in range(3, 6)]),
                               'mlp_num_layers': trial.suggest_int(name='mlp_num_layers', low=1, high=3, step=1)})
            if self._regression_type == 'lstm':
                params['dropout_prob'] = trial.suggest_float(name='dropout_prob', low=1e-1,
                                                             high=2e-1, step=1e-2) if params['num_layers'] > 1 else 0
            VALID_SIZE = len(train_date) // 5
            train_date = train_date[:-VALID_SIZE]
            first, second, third = \
                scaling(kind=params['scaling'], df=train.loc[train.index.get_level_values(1).isin(train_date),
                train.columns.str.contains('RV')])
            BATCH_SIZE = len(self._universe)
            valid_dataloader = \
                train.groupby(
                    by=pd.Grouper(level=0), group_keys=False).apply(lambda x, idx: x.iloc[-idx:, :], idx=VALID_SIZE)
            train_dataloader = \
                train.groupby(
                    by=pd.Grouper(level=0), group_keys=False).apply(lambda x, idx: x.iloc[:-idx, :], idx=VALID_SIZE)
            if params['scaling'] == 'standardise':
                train_dataloader.loc[:, train_dataloader.columns.str.contains('RV')] = \
                    train_dataloader.loc[:, train_dataloader.columns.str.contains('RV')].sub(first).div(second)
                valid_dataloader.loc[:, valid_dataloader.columns.str.contains('RV')] = \
                    valid_dataloader.loc[:, valid_dataloader.columns.str.contains('RV')].sub(first).div(second)
            elif params['scaling'] == 'normalise':
                train_dataloader.loc[:, train_dataloader.columns.str.contains('RV')] = \
                    train_dataloader.loc[:, train_dataloader.columns.str.contains('RV')].sub(second).div(third)
                valid_dataloader.loc[:, valid_dataloader.columns.str.contains('RV')] = \
                    valid_dataloader.loc[:, valid_dataloader.columns.str.contains('RV')].sub(second).div(third)
            elif params['scaling'] == 'maxabs':
                train_dataloader.loc[:, train_dataloader.columns.str.contains('RV')] = \
                    train_dataloader.loc[:, train_dataloader.columns.str.contains('RV')].div(first)
                valid_dataloader.loc[:, valid_dataloader.columns.str.contains('RV')] = \
                    valid_dataloader.div(first)
            TRAIN_SIZE = train_dataloader.shape[0] // BATCH_SIZE
            VALID_SIZE = valid_dataloader.shape[0] // BATCH_SIZE
            if self._INPUT_SIZE is None:
                self._INPUT_SIZE = train_dataloader.shape[1]-1
            model = LSTM_NNModel(input_size=self._INPUT_SIZE, hidden_size=params['hidden_size'],
                                 num_layers=params['num_layers'], mlp_hidden_size=params.get('mlp_hidden_size'),
                                 mlp_num_layers=params.get('mlp_num_layers'), lr=params['lr'], output_size=1,
                                 dropout_rate=params['dropout_prob'], batch_size=params['batch_size'])
            if not init:
                try:
                    state = torch.load(os.path.relpath(start='.', path=prev_tmp_model_path))
                    model = model.load_state_dict(state['model_state_dict'])
                    model.eval()
                except FileNotFoundError:
                    pass
                except KeyError:
                    pass
            model.to(device=DEVICE)
            train_dataloader = \
                DataLoader(torch.tensor(train_dataloader.values.reshape(BATCH_SIZE, TRAIN_SIZE, self._INPUT_SIZE + 1),
                                        dtype=torch.float32, device=DEVICE), batch_size=1, shuffle=True,
                           pin_memory=False)
            valid_dataloader = \
                DataLoader(torch.tensor(valid_dataloader.values.reshape(BATCH_SIZE, VALID_SIZE, self._INPUT_SIZE + 1),
                                        dtype=torch.float32, device=DEVICE), shuffle=False, pin_memory=False,
                           batch_size=1)
            train_model(train=train_dataloader, valid=valid_dataloader, EPOCH=EPOCH, model=model,
                        early_stopping_patience=EARLY_STOPPING_PATIENCE,
                        init=training_freq(date=datetime.strptime(date, '%Y-%m-%d').date()))
            y_hat = model(valid_dataloader.dataset[:, :, 1:])
            y = valid_dataloader.dataset[:, :, 0].view(y_hat.shape)
            """Scale back to original scale"""
            valid_tmp = torch.concat([y, y_hat], 2).detach().cpu().numpy()
            valid_tmp = np.concatenate(valid_tmp, 0)
            valid_index = valid.index.get_level_values(1).unique()[-valid_tmp.shape[0]//5:]
            valid_index = valid.loc[valid.index.get_level_values(1).isin(valid_index), :]
            valid_tmp = pd.DataFrame(data=valid_tmp, index=valid_index.index[-valid_tmp.shape[0]:])
            valid_tmp[[0]] = \
                inverse_scaling(df=valid_tmp[[0]].rename(columns={0: 'RV'}), kind=params['scaling'],
                                first=first, second=second, third=third)
            valid_tmp[[1]] = \
                inverse_scaling(df=valid_tmp[[1]].rename(columns={1: 'RV'}), kind=params['scaling'], first=first,
                                second=second, third=third)
            valid_tmp = self._factory_transformation_dd[self._transformation]['inverse'](valid_tmp)
            loss = \
                valid_tmp.groupby(
                    by=[pd.Grouper(level=0), pd.Grouper(level=1, freq='1W')]).apply(
                    lambda x: mean_squared_error(x.iloc[:, 0], x.iloc[:, 1])).sum()
            trial.set_user_attr(key='best_score', value=loss)
            model.scaling_first = first
            model.scaling_second = second
            model.scaling_third = third
            model.scaling_name = params['scaling']
            if init:
                torch.save(model.state_dict(), tmp_model_path)
        trial.set_user_attr(key='best_estimator', value=model)
        return trial.user_attrs['best_score']

    @staticmethod
    def fill_inf(df) -> pd.DataFrame:
        df = df.replace(np.inf, np.nan)
        df = df.fillna(df.ewm(span=12, min_periods=1).mean())
        return df

    def add_metrics_freq(self, transformation: str, df: pd.DataFrame, date: datetime, **kwargs) -> \
            Union[Tuple[datetime, lightgbm.Booster], Tuple[datetime, LSTM_NNModel], Tuple[datetime, None]]:
        start_idx = pd.to_datetime((date - relativedelta(days={'1W': 7, '1M': 30, '6M': 180}[self._L])), utc=True)
        end_idx = pd.to_datetime((date - relativedelta(days=1)), utc=True)
        train = df.loc[(df.index.get_level_values(1) >= start_idx) &
                       (df.index.get_level_values(1) <= end_idx), :].copy()
        train = train.groupby(by=pd.Grouper(level='symbol'), group_keys=False).apply(lambda x: UAM.fill_inf(x))
        train_range = split_train_valid_set(train)
        valid = train.loc[~train.index.get_level_values(1).isin(train_range), :]
        train = train.loc[train.index.get_level_values(1).isin(train_range), :]
        study_name = f'UAM_{self._L}_{transformation}_{self._regression_type}_{self._model_type}_{date}'
        study = optuna.create_study(direction='minimize', study_name=study_name, sampler=RandomSampler(123))
        objective = partial(self.objective, train=train, valid=valid, init=training_freq(date, self._L))
        study.optimize(objective, n_trials=N_TRIALS, callbacks=[self.callback], show_progress_bar=True,
                       n_jobs=N_JOBS)
        model = study.user_attrs['best_estimator']
        print(f'[End of Training]: Training on {date} has been completed.')
        return date, model

    def add_metrics(self, regression_type: str, transformation: str, agg: str, df: pd.DataFrame, **kwargs) -> None:
        self._regression_type = regression_type
        self._transformation = transformation
        try:
            rv_mkt = self.volatility_period(df.loc[:, ~df.columns.str.contains('VIXM')])
        except ValueError:
            rv_mkt = self.volatility_period(self._reader_obj.rv_read())
        y = list()
        df.index.name = 'timestamp'
        if kwargs.get('trading_session') == 0:
            df = pd.concat([df, kwargs['vixm'].dropna()], axis=1)
            df = df.fillna(df.ewm(span=12, min_periods=1).mean()).dropna()
        if (self._universe is None) & (self._regression_type != 'var'):
            self.universe = df.columns.tolist()
        dates = [date for date in np.unique(df.index.date).tolist()]
        start_idx = dates.index((dates[0] + relativedelta(days={'1W': 7, '1M': 30, '6M': 180}[self._L])))
        if regression_type != 'var':
            if (self._L == '1W') & (regression_type in ['lstm']):
                raise ValueError('Increase lookback window to use LSTM.')
            df = pd.melt(df, ignore_index=False, var_name='symbol', value_name='RV').set_index('symbol', append=True)
            df = df.swaplevel(-1, 0)
            df = df.groupby(by=pd.Grouper(level='symbol'), group_keys=False).apply(lambda x, L_train: lags(x, L_train),
                                                                                   L_train=self._F)
            one_hot_encoding = pd.DataFrame(data=pd.get_dummies(df.index.get_level_values(0), prefix='is').astype(int))
            one_hot_encoding.index = df.index
            df.loc[:, df.columns.str.contains('RV')] = \
                self.factory_transformation_dd[transformation]['transformation'](
                    df.loc[:, df.columns.str.contains('RV')])
            df = pd.concat([df, one_hot_encoding], axis=1)
            model_s = pd.Series(data=np.nan, index=dates[start_idx:])
            for idx, date in enumerate(dates[start_idx::{'1W': 7, '1M': 30, '6M': 180}[self._L]]):
                date, model = self.add_metrics_freq(transformation=transformation, df=df, date=date, idx=idx, **kwargs)
                model_s[date] = model
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = \
                        [executor.submit(self.add_metrics_freq, transformation, df,
                                         date+relativedelta(days=idx),
                                         **kwargs) for idx in range(1, {'1W': 7, '1M': 30, '6M': 180}[self._L])]
                    for future in concurrent.futures.as_completed(futures):
                        date, model = future.result()
                        model_s[date] = model
                model_s = model_s.ffill().dropna()
            #     date, model = self.add_metrics_freq(transformation=transformation, df=df, date=date, idx=idx, **kwargs)
            #     model_s[date] = model
            # with concurrent.futures.ThreadPoolExecutor() as executor:
            #     futures = [executor.submit(self.add_metrics_freq, transformation, df, date, idx, **kwargs) for idx, date
            #                in enumerate(dates[start_idx:])]
            #     for future in concurrent.futures.as_completed(futures):
            #         date, model = future.result()
            #         model_s[date] = model
            # model_s = model_s.ffill().dropna()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                if self._regression_type != 'lightgbm':
                    futures = [executor.submit(lambda x, df, model: reshape_dataframe(x, df, model), x=date, df=df,
                                               model=model_s[date]) for date in model_s.index]
                else:
                    futures = [executor.submit(lambda x, data: (x, data), x=date,
                                               data=df.loc[df.index.get_level_values(1).date == date, :])
                               for date in model_s.index]
                for future in concurrent.futures.as_completed(futures):
                    date, data = future.result()
                    if self._regression_type != 'lightgbm':
                        y_hat = model_s[date](data[:, :, 1:]).detach().numpy()
                    else:
                        y_hat = model_s[date].predict(data.iloc[:, 1:])
                    y_hat = pd.DataFrame(data=y_hat.reshape(np.cumprod(y_hat.shape)[
                                                                int(self._regression_type != 'lightgbm')]),
                                         columns=[{True: 'RV', False: 0}[self._regression_type != 'lightgbm']],
                                         index=df.loc[df.index.get_level_values(1).date == date, :].index)

                    if self._regression_type != 'lightgbm':
                        y_hat = y_hat.groupby(by=pd.Grouper(level='symbol'),
                                              group_keys=False).apply(
                            lambda x, model: inverse_scaling(model_s[date].scaling_name, x,
                                                             model.scaling_first, model.scaling_second,
                                                             model.scaling_third), model=model_s[date])\
                            .rename(columns={'RV': 0})
                    y.append(y_hat)
            y = pd.concat(y)
            y = df[['RV']].loc[y.index.unique(), :].join(y, how='inner')
            y = y.rename(columns={'RV': 'y', 0: 'y_hat'})
        else:
            vars_dd = dict()
            var_obj = VAR_Model()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = \
                    [executor.submit(var_obj.var_train, idx, date, df, self._L, self._transformation,
                                     self._factory_transformation_dd, **kwargs) for idx, date
                     in enumerate(dates[start_idx:])]
                for future in concurrent.futures.as_completed(futures):
                    date, var = future.result()
                    vars_dd[date] = var
            vars_dd = pd.DataFrame(vars_dd.items()).set_index(0).sort_index().ffill()
            vars_dd = dict(zip(vars_dd.index, vars_dd.iloc[:, 0]))
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(var_obj.var_forecast, var_model, date, df, self._L, self._b) for date,
                var_model in vars_dd.items()]
                for future in concurrent.futures.as_completed(futures):
                    y.append(future.result())
            y_hat = pd.concat(y).sort_index().unstack()
            y = \
                self.factory_transformation_dd[transformation]['transformation'](
                df.loc[df.index.isin(y_hat.index.get_level_values(1).unique()), :])
            y = y.unstack()
            y = pd.concat([y, y_hat], axis=1)
            y = \
                y.reset_index().rename(columns={'level_0': 'symbol',
                                                'level_1': 'timestamp',
                                                0: 'y',
                                                1: 'y_hat'}).set_index(['symbol', 'timestamp'])
        y = self.factory_transformation_dd[transformation]['inverse'](y).sort_index(axis=0, level=[0, 1])
        y = y.groupby(by=[pd.Grouper(level='symbol'), pd.Grouper(level='timestamp', freq=self._h)]).sum()
        if kwargs.get('trading_session') in [1, None]:
            tmp = y.groupby(by=[pd.Grouper(level='symbol'), pd.Grouper(level='timestamp', freq=agg)])
        else:
            tmp = \
                y.loc[~y.index.get_level_values(0).str.contains('VIXM'), :].groupby(
                    by=[pd.Grouper(level='symbol'), pd.Grouper(level='timestamp', freq=agg)])
        qlike = tmp.apply(qlike_score).reset_index(0).rename(columns={0: 'values'})
        y = y.reset_index(0)
        con_dd = {'qlike': TrainingScheme._db_connect_qlike, 'y': TrainingScheme._db_connect_y}
        table_dd = {'y': y, 'qlike': qlike}
        # if regression_type == 'var':
        #     r2 = tmp.apply(lambda x: r2_score(x.iloc[:, 0], x.iloc[:, -1])).reset_index(0).rename(columns={0: 'values'})
        #     con_dd.update({'r2': TrainingScheme._db_connect_r2})
        #     table_dd.update({'r2': r2})
        transformation_dd = {'log': 'log', None: 'level'}
        del kwargs['freq']
        if 'vixm' in kwargs.keys():
            del kwargs['vixm']
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(lambda x, y: (x, y), x=table_name,
                                       y=table) for table_name, table in table_dd.items()]
            for count, future in enumerate(concurrent.futures.as_completed(futures)):
                table_name, table = future.result()
                table = table.assign(model=self._model_type, L=self._L, training_scheme=self.__class__.__name__,
                                     regression=regression_type, transformation=transformation_dd[transformation],
                                     h=self._h, **kwargs)
                if table_name in ['qlike', 'mse', 'r2']:
                    table = table.assign(metric=table_name)
                elif table_name == 'feature_importance':
                    table = table.drop('regression', axis=1)
                if len(kwargs) == 0:
                    table = table.assign(trading_session=kwargs.get('trading_session'), top_book=kwargs.get('top_book'))
                if table_name != 'feature_importance':
                    table = table.join(rv_mkt.reindex(table.index).ffill(), how='left')
                    table = table.groupby([table.index, 'symbol']).last().reset_index(-1)
                table.to_sql(if_exists='append', con=con_dd[table_name], name=f'{table_name}_{self._L}') if \
                    table_name != 'feature_importance' else table.to_sql(if_exists='append', con=con_dd[table_name],
                                                                         name=f'{table_name}')
                if table_name != 'feature_importance':
                    print(f'[Insertion]: '
                          f'Tables for '
                          f'{self.__class__.__name__}_{self._L}_{transformation_dd[transformation]}_{regression_type}_'
                          f'{self._model_type}'
                          f' have been inserted into the database ({count+1})....')
                else:
                    model_type = {False: self._model_type, True: 'ar'}[self._model_type is None]
                    print(f'[Insertion]: '
                          f'Table for {self.__class__.__name__}_{self._L}_{model_type}'
                          f' has been inserted into the database ({count+1})....')
        # for table_name, table in table_dd.items():
        #     table = table.assign(model=self._model_type, L=self._L, training_scheme=self.__class__.__name__,
        #                          regression=regression_type, transformation=transformation_dd[transformation],
        #                          h=self._h, **kwargs)
        #     if table_name in ['qlike', 'mse', 'r2']:
        #         table = table.assign(metric=table_name)
        #     elif table_name == 'feature_importance':
        #         table = table.drop('regression', axis=1)
        #     if len(kwargs) == 0:
        #         table = table.assign(trading_session=kwargs.get('trading_session'), top_book=kwargs.get('top_book'))
        #     if table_name != 'feature_importance':
        #         table = table.join(rv_mkt.reindex(table.index).ffill(), how='left')
        #         table = table.groupby([table.index, 'symbol']).last().reset_index(-1)
        #     table.to_sql(if_exists='append', con=con_dd[table_name], name=f'{table_name}_{self._L}') if \
        #         table_name != 'feature_importance' else table.to_sql(if_exists='append', con=con_dd[table_name],
        #                                                              name=f'{table_name}')
        #     if table_name != 'feature_importance':
        #         print(f'[Insertion]: '
        #               f'Tables for '
        #               f'{self.__class__.__name__}_{self._L}_{transformation_dd[transformation]}_{regression_type}_'
        #               f'{self._model_type}'
        #               f' have been inserted into the database ({count})....')
        #     else:
        #         model_type = {False: self._model_type, True: 'ar'}[self._model_type is None]
        #         print(f'[Insertion]: '
        #               f'Table for {self.__class__.__name__}_{self._L}_{model_type}'
        #               f' has been inserted into the database ({count})....')
        #     count += 1






