import pdb
import typing
import pandas as pd
from xgboost import XGBRegressor
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
    FeatureHARDummy, FeatureHARCDR, FeatureHARCSR
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import threading
#FeatureHARUniversal, FeatureHARUniversalPuzzle, rv_1w_correction
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import metrics
import warnings
import matplotlib.pyplot as plt


"""Functions used to facilitate computation within classes."""
qlike_score = lambda x: ((x.iloc[:, 0].div(x.iloc[:, -1]))-np.log(x.iloc[:, 0].div(x.iloc[:, -1]))-1).mean()


def flatten(ls: list()) -> list:
    flat_ls = []
    for sub_ls in ls:
        flat_ls += sub_ls
        return flat_ls


class ModelBuilder:
    _filter_dd = {'har': lambda x: f'{x}_', 'har_dummy_markets': lambda x: f'{x}_|session',
                  'har_cdr': lambda x: f'{x}_|{x}_CDR', 'har_csr': lambda x: f'{x}_|{x}_CSR_',
                  'risk_metrics': lambda x: f'{x}_|{x}_RET', 'ar': lambda x: f'{x}_'}
    _transformation_dd = {None: 'level', 'log': 'log'}
    _training_schemes = ['SAM', 'ClustAM', 'UAM']
    _factory_model_type_dd = {'ar': FeatureAR(), 'risk_metrics': FeatureRiskMetricsEstimator(), 'har': FeatureHAR(),
                              'har_dummy_markets': FeatureHARDummy(), 'har_cdr': FeatureHARCDR(),
                              'har_csr': FeatureHARCSR()}
    _params_grid_dd = {'xgboost': {'n_estimators': [100, 500, 1_000],
                                   'lambda': np.linspace(0.1, 1, 10).tolist(),
                                   'max_depth': [int(depth) for depth in np.linspace(1, 6, 6).tolist()],
                                   'eta': np.linspace(0.1, 1, 10).tolist()},
                       'elastic': {'alpha': np.linspace(0.1, 1, 10).tolist(),
                                   'l1_ratio': np.linspace(0.25, .75, 3).tolist()},
                       'lasso': {'alphas': np.linspace(0.1, 1, 10).tolist},
                       'ridge': {'alpha': np.linspace(0.1, 1, 10).tolist()}}
    _pca_obj = PCA()
    _factory_transformation_dd = {'log': {'transformation': np.log, 'inverse': np.exp},
                                  None: {'transformation': lambda x: x, 'inverse': lambda x: x}}
    models = [None, 'ar', 'risk_metrics', 'har', 'har_dummy_markets', 'har_cdr', 'har_csr', 'har_universal']
    models_rolling_metrics_dd = {model: dict([('qlike', {}), ('r2', {}), ('mse', {}),
                                              ('tstats', {}), ('pvalues', {}), ('coefficient', {})])
                                 for _, model in enumerate(models) if model}
    models_forecast_dd = {model: list() for _, model in enumerate(models) if model}
    global coins_copy
    coins = [''.join((coin, 'usdt')).upper() for _, coin in enumerate(coin_ls)]
    global coins_dd
    coins_dd = {coin: {} for coin in coins}
    _db_connect_coefficient = sqlite3.connect(database=os.path.abspath('./data_centre/databases/coefficients.db'),
                                              check_same_thread=False)
    _db_connect_mse = sqlite3.connect(database=os.path.abspath('./data_centre/databases/mse.db'),
                                      check_same_thread=False)
    _db_connect_qlike = sqlite3.connect(database=os.path.abspath('./data_centre/databases/qlike.db'),
                                        check_same_thread=False)
    _db_connect_r2 = sqlite3.connect(database=os.path.abspath('./data_centre/databases/r2.db'), check_same_thread=False)
    _db_connect_y = sqlite3.connect(database=os.path.abspath('./data_centre/databases/y.db'),
                                    check_same_thread=False)
    # db_connect_correlation = sqlite3.connect(database=os.path.abspath('./data_centre/databases/correlation.db'),
    #                                          check_same_thread=False)
    # db_connect_outliers = sqlite3.connect(database=os.path.abspath('./data_centre/databases/outliers.db'),
    #                                       check_same_thread=False)
    # db_connect_pca = sqlite3.connect(database=os.path.abspath('./data_centre/databases/pca.db'),
    #                                  check_same_thread=False)
    _db_connect_tstats = sqlite3.connect(database=os.path.abspath('./data_centre/databases/tstats.db'),
                                         check_same_thread=False)
    _db_connect_pvalues = sqlite3.connect(database=os.path.abspath('./data_centre/databases/pvalues.db'),
                                          check_same_thread=False)
    _training_scheme_r2_dd = {training_scheme: coins_dd.copy() for training_scheme in _training_schemes}
    _training_scheme_y_dd = {training_scheme: coins_dd.copy() for training_scheme in _training_schemes}
    _training_scheme_qlike_dd = {training_scheme: coins_dd.copy() for training_scheme in _training_schemes}
    _training_scheme_mse_dd = {training_scheme: coins_dd.copy() for training_scheme in _training_schemes}
    _training_scheme_tstats_dd = {training_scheme: coins_dd.copy() for training_scheme in _training_schemes}
    _training_scheme_pvalues_dd = {training_scheme: coins_dd.copy() for training_scheme in _training_schemes}
    _training_scheme_coefficient_dd = {training_scheme: coins_dd.copy() for training_scheme in _training_schemes}
    coins_copy = coins[::]
    exog_cross_impact_dd = \
        dict([(model, None) for _, model in enumerate(models) if model not in [None, 'har_universal']])
    exog_cluster_impact_dd = \
        dict([(model, None) for _, model in enumerate(models) if model not in [None, 'har_universal']])
    outliers_dd = {L: list() for _, L in enumerate(['1H', '6H', '12H', '1D', '1W', '1M'])}
    _pairs = list(product(coins, repeat=2))
    _pairs = [(syms[0], syms[-1]) for _, syms in enumerate(_pairs)]
    _kmeans = KMeans(n_init='auto', random_state=123)
    L_shift_dd = {'5T': pd.to_timedelta('5T') // pd.to_timedelta('5T'),
                  '30T': pd.to_timedelta('30T') // pd.to_timedelta('5T'),
                  '1H': pd.to_timedelta('1H') // pd.to_timedelta('5T'),
                  '6H': pd.to_timedelta('6H') // pd.to_timedelta('5T'),
                  '12H': pd.to_timedelta('12H') // pd.to_timedelta('5T'),
                  '1D': pd.to_timedelta('1D') // pd.to_timedelta('5T'),
                  '1W': pd.to_timedelta('1W') // pd.to_timedelta('5T'),
                  '1M': pd.to_timedelta('30D') // pd.to_timedelta('5T')}
    _cluster_dd = dict()
    start_dd = {'1D': relativedelta(days=1), '1W': relativedelta(weeks=1), '1M': relativedelta(months=1)}
    reader_obj = Reader(file='./data_centre/tmp/aggregate2022')

    def __init__(self, h: str, F: typing.List[str], L: str, Q: str, model_type: str=None, s=None, b: str='5T'):
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
        max_feature_building_lookback = \
            max([pd.to_timedelta(lookback) // pd.to_timedelta(self._b) if 'M' not in lookback else
                 pd.to_timedelta(''.join((str(30 * int(lookback.split('M')[0])), 'D'))) // pd.to_timedelta(self._b)
                 for _, lookback in enumerate(self._F)])
        upper_bound_feature_building_lookback = \
            pd.to_timedelta(''.join((str(30 * int(self.L.split('M')[0])), 'D'))) // pd.to_timedelta(self._b) \
                if 'M' in self._L else pd.to_timedelta(self._L) // pd.to_timedelta(self._b)
        if upper_bound_feature_building_lookback < max_feature_building_lookback:
            raise ValueError('Lookback window for model training is smaller than the furthest lookback window '
                             'in feature building.')
        if model_type not in ModelBuilder.models:
            raise ValueError('Unknown model type.')
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

    @staticmethod
    def correlation(cutoff_low: float = .01, cutoff_high: float = .01, insert: bool=True) \
            -> typing.Union[None, pd.DataFrame]:
        correlation_dd = dict()
        reader_obj = Reader(file=os.path.abspath('../data_centre/tmp/aggregate2022'))
        returns = reader_obj.returns_read(raw=False, cutoff_low=cutoff_low, cutoff_high=cutoff_high)
        for L in ['1D', '1W', 'SAM']:
            window = pd.to_timedelta('30D') // pd.to_timedelta('5T') if L == 'SAM'\
                else max(4, pd.to_timedelta(L) // pd.to_timedelta('5T'))
            correlation = \
                returns.rolling(window=window).corr().dropna().droplevel(axis=0, level=1).mean(axis=1)
            correlation = correlation.groupby(by=correlation.index).mean()
            correlation = correlation.resample('1D').mean()
            correlation.name = L
            correlation_dd[L] = correlation
        correlation = pd.DataFrame(correlation_dd)
        correlation = pd.melt(frame=correlation, value_name='value', var_name='lookback_window', ignore_index=False)
        if insert:
            print(f'[Insertion]: Correlation table...............................')
            correlation.to_sql(name='correlation', con=ModelBuilder.db_connect_correlation, if_exists='replace')
            print(f'[Insertion]: Correlation table is not completed.')
            return
        else:
            return correlation

    @staticmethod
    def covariance(cutoff_low: float = .01, cutoff_high: float = .01, insert: bool = True) \
            -> typing.Union[None, pd.DataFrame]:
        covariance_dd = dict()
        reader_obj = Reader(file=os.path.abspath('../data_centre/tmp/aggregate2022'))
        returns = reader_obj.returns_read(raw=False, cutoff_low=cutoff_low, cutoff_high=cutoff_high)
        for L in ['1D', '1W', 'SAM']:
            window = pd.to_timedelta('30D') // pd.to_timedelta('5T') if L == 'SAM' \
                else max(4, pd.to_timedelta(L) // pd.to_timedelta('5T'))
            covariance = \
                returns.rolling(window=window).cov().dropna().droplevel(axis=0, level=1).mean(axis=1)
            covariance = covariance.groupby(by=covariance.index).mean()
            covariance = covariance.resample('1D').mean()
            covariance.name = L
            covariance_dd[L] = covariance
        covariance = pd.DataFrame(covariance_dd)
        covariance = pd.melt(frame=covariance, value_name='value', var_name='lookback_window', ignore_index=False)
        if insert:
            print(f'[Insertion]: Covariance table...............................')
            covariance.to_sql(name='covariance', con=ModelBuilder.db_connect_correlation, if_exists='replace')
            print(f'[Insertion]: Covariance table is not completed.')
            return
        else:
            return covariance

    def rolling_metrics(self, training_scheme: str, symbol: str, exog: pd.DataFrame, agg: str,
                        regression_type: str = 'linear', transformation: str = None) -> typing.Tuple[pd.Series]:
        factory_regression_dd = \
            {'linear': LinearRegression(),
             'xgboost': RandomizedSearchCV(estimator=XGBRegressor(objective='reg:squarederror'),
                                           param_distributions=ModelBuilder._params_grid_dd['xgboost']),
             'elastic': ElasticNetCV(max_iter=5_000), 'lasso': LassoCV(max_iter=5_000), 'ridge': RidgeCV()}
        stats_condition = \
            (self._model_type == 'risk_metrics') & (training_scheme == 'SAM') & (regression_type == 'linear')
        stats_condition = not stats_condition
        feature_obj = ModelBuilder._factory_model_type_dd[self._model_type]
        if training_scheme != 'SAM':
            exog = feature_obj.builder(symbol=exog.columns, df=exog, F=self._F)
        endog = exog.pop(symbol) if self._model_type != 'risk_metrics' else exog.copy()
        endog.replace(0, np.nan, inplace=True)
        endog.ffill(inplace=True)
        y = list()
        if not ((self._model_type == 'risk_metrics') & (training_scheme == 'SAM')):
            L_train = ModelBuilder.L_shift_dd[self._L]
            test_size = pd.to_timedelta('1D')//pd.to_timedelta(self._b)
            split_obj = TimeSeriesSplit(n_splits=round((exog.shape[0]-L_train)//test_size),
                                        max_train_size=L_train, test_size=test_size, gap=0)
        if self._model_type == 'risk_metrics':
            returns2 = endog.copy()
            exog = \
            ModelBuilder._factory_transformation_dd[transformation]['transformation'](
                returns2.ewm(alpha=feature_obj.factor).mean())
            if training_scheme == 'SAM':
                exog.columns = [0]
                y_hat = exog
                y_test = ModelBuilder._factory_transformation_dd[transformation]['transformation']\
                    (ModelBuilder.reader_obj.rv_read(symbol=symbol).iloc[:, 0])
                y.append(ModelBuilder._factory_transformation_dd[transformation]['inverse']
                         (pd.concat([y_test, y_hat], axis=1)))
            else:
                idx_ls = pd.to_datetime(list(exog.groupby(exog.index.date).groups.keys()), utc=True)
                coefficient = pd.DataFrame(index=idx_ls, columns=['const'] + exog.columns.tolist(), data=np.nan)
                rv = \
                    ModelBuilder._factory_transformation_dd[transformation]['transformation'](
                        ModelBuilder.reader_obj.rv_read(symbol=symbol))
                rres = factory_regression_dd[regression_type]
                split_gen = TimeSeriesSplit(max_train_size=L_train, test_size=test_size, gap=0,
                                            n_splits=exog.shape[0] // (L_train + test_size))
                for i, (train_index, test_index) in enumerate(split_gen.split(exog)):
                    X_train, y_train = exog.iloc[train_index, :], rv.iloc[train_index]
                    X_test, y_test = exog.iloc[test_index, :], rv.iloc[test_index]
                    rres.fit(X_train, y_train)
                    y_hat = pd.Series(data=rres.predict(X_test).reshape(1, -1)[0], index=y_test.index)
                    y.append(ModelBuilder._factory_transformation_dd[transformation]['inverse']
                             (pd.concat([y_test, y_hat], axis=1)))
                    """ Coefficients """
                    coefficient.loc[exog.index[test_index[0]], :] = \
                        np.hstack((np.array([rres.intercept_]), rres.coef_))
                """Tstats"""
                X_train = X_train.assign(const=1)
                X_train = X_train[coefficient.columns]
                tstats = pd.DataFrame(data=coefficient.div((np.diag(np.matmul(X_train.values.transpose(),
                                                                              X_train.values)) / np.sqrt(
                    X_train.shape[0]))), index=coefficient.index, columns=coefficient.columns)
                """Pvalues"""
                pvalues = pd.DataFrame(data=2 * (1 - t.cdf(tstats, df=X_train.shape[0] - coefficient.shape[1] - 1)),
                                       index=tstats.index, columns=tstats.columns)
                tstats.dropna(inplace=True)
                pvalues.dropna(inplace=True)
        else:
            """
                Coefficient, tstats and pvalues dataframes
            """
            rres = factory_regression_dd[regression_type]
            idx_ls = pd.to_datetime(list(exog.groupby(exog.index.date).groups.keys()), utc=True)
            coefficient = pd.DataFrame(index=idx_ls, columns=['const']+exog.columns.tolist(), data=np.nan)
            for _, (train_index, test_index) in enumerate(split_obj.split(exog)):
                X_train, y_train = exog.iloc[train_index, :], endog.iloc[train_index]
                X_test, y_test = exog.iloc[test_index, :], endog.iloc[test_index]
                try:
                    rres.fit(X_train, y_train)
                except ValueError as e:
                    print(e)
                    print(symbol)
                    pdb.set_trace()
                y_hat = pd.Series(data=rres.predict(X_test), index=y_test.index)
                y.append(pd.concat([y_test, y_hat], axis=1))
                """ Coefficients """
                coefficient.loc[exog.index[test_index[0]], :] = \
                    np.concatenate((np.array([rres.intercept_]), rres.coef_))
            """Tstats"""
            X_train = X_train.assign(const=1)
            X_train = X_train[coefficient.columns]
            tstats = pd.DataFrame(data=coefficient.div((np.diag(np.matmul(X_train.values.transpose(),
                    X_train.values))/np.sqrt(X_train.shape[0]))), index=coefficient.index, columns=coefficient.columns)
            """Pvalues"""
            pvalues = pd.DataFrame(data=2 * (1 - t.cdf(tstats, df=X_train.shape[0] - coefficient.shape[1] - 1)),
                                   index=tstats.index, columns=tstats.columns)
            tstats.dropna(inplace=True)
            pvalues.dropna(inplace=True)
        y = pd.concat(y).dropna()
        y = y.resample(self._s).sum()
        tmp = y.groupby(by=pd.Grouper(level=0, freq=agg))
        mse = tmp.apply(lambda x: mean_squared_error(x.iloc[:, 0], x.iloc[:, -1]))
        qlike = tmp.apply(qlike_score)
        r2 = tmp.apply(lambda x: r2_score(x.iloc[:, 0], x.iloc[:, -1]))
        mse = pd.Series(mse, name=symbol)
        qlike = pd.Series(qlike, name=symbol)
        r2 = pd.Series(r2, name=symbol)
        mse = pd.melt(pd.DataFrame(mse), ignore_index=False, value_name='values', var_name='symbol')
        mse = mse.assign(metric='mse')
        r2 = pd.melt(pd.DataFrame(r2), ignore_index=False, value_name='values', var_name='symbol')
        r2 = r2.assign(metric='r2')
        qlike = pd.melt(pd.DataFrame(qlike), ignore_index=False, value_name='values', var_name='symbol')
        qlike = qlike.assign(metric='qlike')
        if stats_condition:
            coefficient.ffill(inplace=True)
            coefficient.dropna(inplace=True)
            coefficient = pd.melt(coefficient, ignore_index=False, var_name='features', value_name='values')
            coefficient = coefficient.assign(training_scheme=training_scheme, transformation=transformation,
                                             regression=regression_type, model=self._model_type)
            tstats.ffill(inplace=True)
            tstats.dropna(inplace=True)
            tstats = pd.melt(tstats, ignore_index=False, var_name='features', value_name='values')
            tstats = tstats.assign(training_scheme=training_scheme, transformation=transformation,
                                   regression=regression_type, model=self._model_type)
            pvalues.ffill(inplace=True)
            pvalues.dropna(inplace=True)
            pvalues = pd.melt(pvalues, ignore_index=False, var_name='features', value_name='values')
            pvalues = pvalues.assign(training_scheme=training_scheme, transformation=transformation,
                                     regression=regression_type, model=self._model_type)
        y = y.reset_index()
        y['symbol'] = symbol
        y = y.assign(model=self._model_type)
        y = y.set_index('timestamp')
        y = y.rename(columns={symbol: 'y', 0: 'y_hat'})
        table_dd = {'r2': r2, 'qlike': qlike, 'mse': mse, 'y': y}
        if stats_condition:
            table_dd.update({'tstats': tstats, 'pvalues': pvalues, 'coefficient': coefficient})
        training_scheme_tables = {'r2': ModelBuilder._training_scheme_r2_dd,
                                  'qlike': ModelBuilder._training_scheme_qlike_dd,
                                  'mse': ModelBuilder._training_scheme_mse_dd,
                                  'y': ModelBuilder._training_scheme_y_dd}
        if stats_condition:
            training_scheme_tables.update({'tstats': ModelBuilder._training_scheme_tstats_dd,
                                           'pvalues': ModelBuilder._training_scheme_pvalues_dd,
                                           'coefficient': ModelBuilder._training_scheme_coefficient_dd})
        for table_name, table in table_dd.items():
            table['symbol'] = symbol
            table['model'] = self._model_type
            table['L'] = self._L
            table['training_scheme'] = training_scheme
            table['transformation'] = ModelBuilder._transformation_dd[transformation]
            table['regression'] = regression_type
            table['h'] = self._h
            training_scheme_tables[table_name][training_scheme][symbol] = table

    @staticmethod
    def reinitialise_models_forecast_dd():
        ModelBuilder.models_forecast_dd = {model: list() for _, model in enumerate(ModelBuilder.models) if model}

    def add_metrics_per_symbol(self, symbol: str, training_scheme: str, exog: pd.DataFrame, agg: str,
                               regression_type: str = 'linear', transformation: str = None) -> None:
        self.rolling_metrics(symbol=symbol, training_scheme=training_scheme, exog=exog,
                             regression_type=regression_type, transformation=transformation, agg=agg)
        y = ModelBuilder._training_scheme_y_dd[training_scheme][symbol]
        r2 = ModelBuilder._training_scheme_r2_dd[training_scheme][symbol]
        qlike = ModelBuilder._training_scheme_qlike_dd[training_scheme][symbol]
        mse = ModelBuilder._training_scheme_mse_dd[training_scheme][symbol]
        con_dd = {'r2': ModelBuilder._db_connect_r2, 'qlike': ModelBuilder._db_connect_qlike,
                  'mse': ModelBuilder._db_connect_mse, 'y': ModelBuilder._db_connect_y}
        table_dd = {'r2': r2, 'qlike': qlike, 'mse': mse, 'y': y}
        if (self._model_type != 'risk_metrics') & (regression_type in ['linear', 'lasso', 'ridge', 'elastic']):
            tstats = ModelBuilder._training_scheme_tstats_dd[training_scheme][symbol]
            pvalues = ModelBuilder._training_scheme_pvalues_dd[training_scheme][symbol]
            coefficient = ModelBuilder._training_scheme_coefficient_dd[training_scheme][symbol]
            con_dd.update({'tstats': ModelBuilder._db_connect_tstats, 'pvalues': ModelBuilder._db_connect_pvalues,
                           'coefficient': ModelBuilder._db_connect_coefficient})
            table_dd.update({'tstats': tstats, 'pvalues': pvalues, 'coefficient': coefficient})
        for table_name, table in table_dd.items():
            table.to_sql(if_exists='append', con=con_dd[table_name], name=f'{table_name}_{self._L}')
        print(f'[Insertion]: Tables for {training_scheme}_{self._L}_{transformation}_{regression_type}_{symbol}'
              f' have been inserted into the database....')

    def add_metrics(self, training_scheme: str, regression_type: str = 'linear', transformation: str = None,
                    **kwargs) -> None:
        for symbol in ModelBuilder.coins:
            if training_scheme == 'SAM':
                if self._model_type in ['ar', 'har', 'har_dummy_markets', 'risk_metrics']:
                    exog = \
                        ModelBuilder._factory_model_type_dd[self._model_type].builder(F=self._F, df=kwargs['df'],
                                                                                      symbol=symbol)
                elif self._model_type in ['har_cdr', 'har_csr']:
                    exog = \
                        ModelBuilder._factory_model_type_dd[self._model_type].builder(F=self._F, df=kwargs['df'],
                                                                                      df2=kwargs['df2'], symbol=symbol)
            elif training_scheme == 'ClustAM':
                """Number of cluster selection procedure"""
                silhouette = list()
                tmp = kwargs['df'].transpose().copy()
                for k in range(1, len(ModelBuilder.coins)+1):
                    ModelBuilder._kmeans.n_clusters = k
                    ModelBuilder._kmeans.fit(tmp)
                    silhouette.append(silhouette_score(tmp, ModelBuilder._kmeans.labels_))
                silhouette = silhouette[1:-1]
                k_best = silhouette.index(max(silhouette))+2
                ModelBuilder._kmeans.n_clusters = k_best
                ModelBuilder._kmeans.fit(tmp)
                tmp.loc[:, 'labels'] = ModelBuilder._kmeans.labels_
                cluster_group = tmp.groupby(by='labels')
                for cluster, members in cluster_group.groups.items():
                    if symbol in members:
                        members_ls = members
                exog = kwargs['df'].loc[:, members_ls]
            elif training_scheme == 'UAM':
                exog = kwargs['df']
            self.add_metrics_per_symbol(symbol, training_scheme, exog, kwargs['agg'], regression_type,
                                        transformation)


class DMTest:

    _db_connect_y = sqlite3.connect(database=os.path.abspath('./data_centre/databases/y.db'),
                                    check_same_thread=False)

    def __init__(self):
        pass

    def table(self, L: str, training_scheme: str) -> pd.DataFrame:
        """
            Method computing a matrix with DM statistics for every (i,j) as entries
        """
        regression_per_training_scheme_dd = {'SAM': ['linear', 'lasso', 'ridge', 'elastic'],
                                             'ClustAM': ['lasso', 'ridge', 'elastic'],
                                             'UAM': ['lasso', 'ridge', 'elastic']}
        models_dd = dict()
        for scheme in ['SAM', 'ClustAM', 'UAM']:
            for regression in regression_per_training_scheme_dd[scheme]:
                for model in ['risk_metrics', 'ar', 'har', 'har_dummy_markets']:
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
    _commonality_connect_db = sqlite3.connect(database=os.path.abspath('./data_centre/databases/commonality.db'))
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
        commonality_df = pd.DataFrame(data=np.nan, index=Commonality._rv.resample(self._L).groups.keys(),
                                      columns=Commonality._rv.columns)
        for i in range(0, Commonality._rv.shape[1]):
            X_train = Commonality._rv.copy()
            y_train = X_train.iloc[:, i]
            X_train = X_train.mean(axis=1).values.reshape(-1, 1)
            self._rv_fitted.iloc[:, i] = Commonality._linear.fit(X_train, y_train).predict(X_train)
        self._rv_fitted = \
            Commonality._factory_transformation_dd[self._transformation]['inverse'](self._rv_fitted)
        Commonality._rv = \
            Commonality._factory_transformation_dd[self._transformation]['inverse'](Commonality._rv)
        for i in range(0, Commonality._rv.shape[1]):
            tmp = pd.DataFrame(data={Commonality._rv.iloc[:, i].name: Commonality._rv.iloc[:, i],
                                     self._rv_fitted.iloc[:, i].name: self._rv_fitted.iloc[:, i]})
            tmp = tmp.groupby(by=pd.Grouper(level=0, freq=self._L))
            r2 = tmp.apply(lambda x: 1-(1-r2_score(x.iloc[:, 0], x.iloc[:, -1]))*(x.shape[0]-1)/(x.shape[0]-2))
            commonality_df.iloc[:, i] = r2.values
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


class TensorDecomposition:

    def __init__(self, rv: pd.DataFrame):
        self._rv = rv
        self._rv.index = \
            pd.MultiIndex.from_tuples([(date, time) for date, time in zip(self._rv.index.date, self._rv.index.time)])
        self._rv.index = self._rv.index.set_names(['Date', 'Time'])
        self._rv = self._rv.stack()
        self._rv.index = self._rv.index.set_names(['Date', 'Time', 'Symbol'])
        self.tensor = pd_to_tensor(self._rv, keep_index=True)
        pdb.set_trace()
