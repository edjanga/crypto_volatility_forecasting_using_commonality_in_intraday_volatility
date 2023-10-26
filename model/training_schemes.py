import concurrent.futures
import pdb
import typing
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LinearRegression, LassoCV, Ridge, ElasticNetCV, Lasso, ElasticNet
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error, silhouette_score
from sklearn.cluster import KMeans
from datetime import datetime
import os
from dateutil.relativedelta import relativedelta
from data_centre.helpers import coin_ls
import sqlite3
from model.feature_engineering_room import FeatureAR, FeatureHAR, FeatureHAREq
import numpy as np
import optuna
from model.lab import qlike_score
from optuna.samplers import RandomSampler


class TrainingScheme(object):
    _data_centre_dir = os.path.abspath(__file__).replace('/model/training_schemes.py', '/data_centre')
    _db_connect_coefficient = sqlite3.connect(database=os.path.abspath(f'{_data_centre_dir}/'
                                                                       f'databases/coefficients.db'),
                                              check_same_thread=False)
    _db_connect_mse = sqlite3.connect(database=os.path.abspath(f'{_data_centre_dir}/databases/mse.db'),
                                      check_same_thread=False)
    _db_connect_qlike = sqlite3.connect(database=os.path.abspath(f'{_data_centre_dir}/databases'
                                                                 f'/qlike.db'),
                                        check_same_thread=False)
    _db_connect_r2 = sqlite3.connect(database=os.path.abspath(f'{_data_centre_dir}/databases'
                                                              f'/r2.db'), check_same_thread=False)
    _db_connect_y = sqlite3.connect(database=os.path.abspath(f'{_data_centre_dir}/databases/y.db'),
                                    check_same_thread=False)
    _db_connect_pca = sqlite3.connect(database=os.path.abspath(f'{_data_centre_dir}/databases/pca.db'),
                                      check_same_thread=False)
    _db_connect_tstats = sqlite3.connect(database=os.path.abspath(f'{_data_centre_dir}/databases'
                                                                    f'/tstats.db'),
                                         check_same_thread=False)
    _db_connect_pvalues = sqlite3.connect(database=os.path.abspath(f'{_data_centre_dir}/databases'
                                                                   f'/pvalues.db'),
                                          check_same_thread=False)
    _factory_model_type_dd = {'ar': FeatureAR(), 'har': FeatureHAR(), 'har_eq': FeatureHAREq()}

    _factory_transformation_dd = {'log': {'transformation': np.log, 'inverse': np.exp},
                                  None: {'transformation': lambda x: x, 'inverse': lambda x: x}}
    global coins_copy
    coins = [''.join((coin, 'usdt')).upper() for _, coin in enumerate(coin_ls)]
    global coins_dd
    coins_dd = {coin: {} for coin in coins}
    _training_scheme_r2_dd = coins_dd.copy().copy()
    _training_scheme_y_dd = coins_dd.copy().copy()
    _training_scheme_qlike_dd = coins_dd.copy().copy()
    _training_scheme_mse_dd = coins_dd.copy().copy()
    _training_scheme_tstats_dd = coins_dd.copy().copy()
    _training_scheme_pvalues_dd = coins_dd.copy().copy()
    _training_scheme_coefficient_dd = coins_dd.copy().copy()
    _training_scheme_1st_comp_weights_dd = coins_dd.copy().copy()
    L_shift_dd = {'5T': pd.to_timedelta('5T') // pd.to_timedelta('5T'),
                  '30T': pd.to_timedelta('30T') // pd.to_timedelta('5T'),
                  '1H': pd.to_timedelta('1H') // pd.to_timedelta('5T'),
                  '6H': pd.to_timedelta('6H') // pd.to_timedelta('5T'),
                  '12H': pd.to_timedelta('12H') // pd.to_timedelta('5T'),
                  '1D': pd.to_timedelta('1D') // pd.to_timedelta('5T'),
                  '1W': pd.to_timedelta('1W') // pd.to_timedelta('5T'),
                  '1M': pd.to_timedelta('30D') // pd.to_timedelta('5T'),
                  '6M': pd.to_timedelta('180D') // pd.to_timedelta('5T')}

    def __init__(self, h: str, F: typing.List[str], L: str, Q: str, universe: typing.List[str], model_type: str=None,
                 s: str=None, b: str='5T'):
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

    def stats_condition(self, regression_type: str) -> bool:
        return False if regression_type == 'lightgbm' else True

    @staticmethod
    def df_per_day(df: pd.DataFrame, date: datetime):
        return date, df.loc[(df.index.date >= date - L_train) & (df.index.date <= date)]

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
    def objective(trial):
        regression_type = trial.study.study_name.split('_')[3]
        training_scheme_name = trial.study.study_name.split('_')[0]
        if regression_type == 'lightgbm':
            train_loader, valid_loader = lgb.Dataset(X_train, label=y_train), lgb.Dataset(X_valid, label=y_valid)
            param = {
                'objective': 'regression', 'metric': 'mse', 'verbosity': -1,
                'max_depth': trial.suggest_int('max_depth', 1, 3), 'lr': .1, 'tree_learner': 'data',
                'feature_fraction': trial.suggest_float('feature_fraction', .1, .8, step=.1, log=False),
                'extra_tree': True, 'boosting_type': 'goss'
            }
            tmp_rres = lgb.train(param, train_set=train_loader, valid_sets=[valid_loader],
                                 num_boost_round=10,
                                 callbacks=[lgb.early_stopping(1, first_metric_only=True, verbose=True,
                                                               min_delta=0.0)])
        elif regression_type == 'lasso':
            param = {
                'objective': 'regression', 'metric': 'mse', 'verbosity': -1,
                'alpha': trial.suggest_float('alpha', .01, .99, log=False)
            }
            tmp_rres = Lasso(alpha=param['alpha'], fit_intercept=training_scheme_name != 'UAM', max_iter=10,
                             random_state=123)
        elif regression_type == 'ridge':
            param = {
                'objective': 'regression', 'metric': 'mse', 'verbosity': -1,
                'alpha': trial.suggest_float('alpha', .01, .99, log=False),
            }
            tmp_rres = Ridge(alpha=param['alpha'], fit_intercept=training_scheme_name != 'UAM', random_state=123)
        elif regression_type == 'elastic':
            param = {
                'objective': 'regression', 'metric': 'mse', 'verbosity': -1,
                'alpha': trial.suggest_float('alpha', .01, .99, log=False),
                'l1_ratio': trial.suggest_float('l1_ratio', .01, .99, log=False),
            }
            tmp_rres = ElasticNet(alpha=param['alpha'], l1_ratio=param['l1_ratio'],
                                  fit_intercept=training_scheme_name != 'UAM', max_iter=10, random_state=123)
        if regression_type not in ['lightgbm', 'xgboost']:
            tmp_rres.fit(X_train, y_train)
        loss = mean_squared_error(y_valid, tmp_rres.predict(X_valid))
        trial.set_user_attr(key='best_estimator', value=tmp_rres)
        return loss

    @staticmethod
    def callback(study, trial):
        if study.best_trial.number == trial.number:
            study.set_user_attr(key='best_estimator', value=trial.user_attrs['best_estimator'])

    def rolling_metrics_per_date(self, date: datetime, symbol: str, regression_type: str,
                                 exog: pd.DataFrame, endog: pd.DataFrame, transformation: str,
                                 **kwargs) -> pd.DataFrame:
        train_index = list(set(exog.index[(exog.index.date >= date - L_train) & (exog.index.date < date)]))
        train_index.sort()
        test_index = list(set(exog.index[exog.index.date == date]))
        test_index.sort()
        global X_train
        global y_train
        global X_test
        global y_test
        if regression_type not in ['linear', 'pcr']:
            valid_index = train_index[-len(train_index) // 5:]
            train_index = train_index[:-len(train_index) // 5]
            global X_valid
            global y_valid
            X_valid, y_valid = exog.loc[exog.index.isin(valid_index), :], endog.loc[endog.index.isin(valid_index)]
        X_train, y_train = exog.loc[exog.index.isin(train_index), :], endog.loc[endog.index.isin(train_index)]
        X_test, y_test = exog.loc[exog.index.isin(test_index), :], endog.loc[endog.index.isin(test_index)]
        if regression_type in ['linear', 'pcr']:
            rres_pca = PCA() if regression_type == 'pcr' else None
            rres = LinearRegression()
        if regression_type not in ['lightgbm', 'xgboost']:
            columns_name_dd = {True: [f'PCA_{i + 1}' for i, _ in enumerate(exog.columns)],
                               False: exog.columns.tolist()}
        if regression_type not in ['linear', 'pcr']:
            n_trials = 5
            study_name = f'{self.__class__.__name__}_{self._L}_{transformation}_{regression_type}_' \
                         f'{self._model_type}_{symbol}_{date.strftime("%Y-%m-%d")}'
            study = optuna.create_study(direction='minimize', study_name=study_name, sampler=RandomSampler(123))
            study.optimize(self.objective, n_trials=n_trials, callbacks=[self.callback], n_jobs=-1)
            rres = study.user_attrs['best_estimator']
        else:
            if regression_type == 'pcr':
                rres_pca.fit(X_train)
                explained_variance_ratio = list(np.cumsum(rres_pca.explained_variance_ratio_) >= .9)
                n_components = explained_variance_ratio.index(True) + 1
                rres_pca.n_components = n_components
                own_train, own_test = pd.DataFrame(), pd.DataFrame()
                """If cluster group contains only more than one member"""
                if self.__class__.__name__ == 'ClustAM':
                    if len(kwargs['cluster']) > 1:
                        own_train, own_test = \
                            X_train.filter(regex=f'{symbol}'), X_test.filter(regex=f'{symbol}')
                X_train = \
                    pd.DataFrame(rres_pca.fit_transform(X_train),
                                 columns=columns_name_dd['pcr' == regression_type][:n_components],
                                 index=X_train.index)
                X_train = pd.concat([X_train, own_train], axis=1)
                X_test = pd.DataFrame(rres_pca.transform(X_test),
                                      columns=columns_name_dd['pcr' == regression_type][:n_components],
                                      index=X_test.index)
                X_test = pd.concat([X_test, own_test], axis=1)
            rres.fit(X_train, y_train)
        if self.__class__.__name__ == 'CAM':
            start = dates[0] + L_train
            if start == date:
                self.feature_importance = pd.DataFrame(index=dates[dates.index(start):], columns=rres.feature_name(),
                                                       data=np.nan)
            self.feature_importance.loc[date, rres.feature_name()] = rres.feature_importance(importance_type='split',
                                                                                             iteration=-1)
        y_hat = pd.Series(data=rres.predict(X_test), index=y_test.index)
        return pd.concat([y_test, y_hat], axis=1)

    def rolling_metrics(self, exog: pd.DataFrame, agg: str, transformation: str,
                             regression_type: str = 'linear', **kwargs) -> typing.Union[None, typing.Tuple[pd.Series]]:
        global L_train
        global endog
        global factor_obj
        if kwargs.get('symbol'):
            symbol = kwargs.get('symbol')
        else:
            symbol = 'RV'
        if self._model_type == 'har_eq':
            exog.loc[:, ~exog.columns.str.contains('_SESSION')] = \
                self._factory_transformation_dd[transformation]['transformation'](
                    exog.loc[:, ~exog.columns.str.contains('_SESSION')])
        endog = exog.pop(symbol)
        y = list()
        global dates
        dates = list(np.unique(exog.index.date))
        L_train = relativedelta(minutes=TrainingScheme.L_shift_dd[self._L] * 5)
        start = dates[0] + L_train
        for date in dates[dates.index(start):]:
            y.append(self.rolling_metrics_per_date(exog=exog, endog=endog, date=date, regression_type=regression_type,
                                                   transformation=transformation, **kwargs))
        if self.__class__.__name__ == 'CAM':
            self.feature_importance = pd.melt(self.feature_importance, var_name='feature', value_name='importance',
                                              ignore_index=False)
            self.feature_importance = \
                self.feature_importance.assign(L=self._L, model_type=self._model_type, transformation=transformation,
                                               training_scheme=self.__class__.__name__, symbol=symbol)
        y = pd.concat(y).dropna()
        y.sort_index(inplace=True)
        y = TrainingScheme._factory_transformation_dd[transformation]['inverse'](y).reset_index(names='timestamp')
        y['symbol'] = symbol
        y = y.groupby(by=[pd.Grouper(key='symbol'), pd.Grouper(key='timestamp', freq=self._h)]).sum()
        tmp = y.groupby(by=[pd.Grouper(level='symbol'), pd.Grouper(level='timestamp', freq=agg)])
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
        y.reset_index(level='symbol', inplace=True)
        y.index = pd.to_datetime(y.index, utc=True)
        y = y.assign(model=self._model_type)
        y = y.rename(columns={symbol: 'y', 0: 'y_hat'})
        table_dd = {'r2': r2, 'qlike': qlike, 'mse': mse, 'y': y}
        training_scheme_tables = {'r2': self._training_scheme_r2_dd, 'qlike': self._training_scheme_qlike_dd,
                                  'mse': self._training_scheme_mse_dd, 'y': self._training_scheme_y_dd}
        if self.__class__.__name__ == 'CAM':
            table_dd.update({'feature_importance': self.feature_importance})
            training_scheme_tables.update({'feature_importance': self.feature_importance_symbol})
        transformation_dd = {'log': 'log', None: 'level'}
        for table_name, table in table_dd.items():
            if table_name != 'feature_importance':
                table = table.assign(symbol=symbol, model=self._model_type, L=self._L,
                                     training_scheme=self.__class__.__name__, regression=regression_type,
                                     transformation=transformation_dd[transformation])
            table['h'] = self._h
            training_scheme_tables[table_name][symbol] = table

    def add_metrics_per_symbol(self, symbol: str, df: pd.DataFrame, agg: str, transformation: str,
                               regression_type: str = 'linear', **kwargs) -> None:
        rv_mkt = self.volatility_period(df)
        exog = self.build_exog(symbol=symbol, df=df, regression_type=regression_type, transformation=transformation)
        transformation_dd = {'log': 'log', None: 'level'}
        self.rolling_metrics(
            symbol=symbol, regression_type=regression_type, transformation=transformation, agg=agg, exog=exog, **kwargs
        )
        y = self._training_scheme_y_dd[symbol]
        y.columns = y.columns.str.replace(f'{symbol}_RET_{self._h}', 'y_hat')
        r2 = self._training_scheme_r2_dd[symbol]
        qlike = self._training_scheme_qlike_dd[symbol]
        mse = self._training_scheme_mse_dd[symbol]
        con_dd = {'r2': TrainingScheme._db_connect_r2, 'qlike': TrainingScheme._db_connect_qlike,
                  'mse': TrainingScheme._db_connect_mse, 'y': TrainingScheme._db_connect_y,
                  'pca': TrainingScheme._db_connect_pca}
        table_dd = {'r2': r2, 'qlike': qlike, 'mse': mse, 'y': y}
        count = 1
        for table_name, table in table_dd.items():
            if table_name != 'y':
                table.drop('symbol', inplace=True, axis=1)
                table = table.reset_index(level=0)
            table = pd.concat([table, rv_mkt], axis=1).ffill().dropna()
            table.to_sql(if_exists='append', con=con_dd[table_name], name=f'{table_name}_{self._L}')
            print(f'[Insertion]: '
                  f'Tables for '
                  f'{self.__class__.__name__}_{self._L}_{transformation_dd[transformation]}_{regression_type}_'
                  f'{self._model_type}_{symbol}'
                  f' have been inserted into the database ({count})....')
            count += 1

    def add_metrics(self, regression_type: str, transformation: str, agg: str, df: pd.DataFrame) -> None:
        pass

    @property
    def db_connect_pca(self):
        return self._db_connect_pca


class SAM(TrainingScheme):

    def build_exog(self, symbol: str, df: pd.DataFrame, transformation: str, regression_type: str = 'linear')\
            -> pd.DataFrame:
        exog = TrainingScheme._factory_model_type_dd[self._model_type].builder(F=self._F, df=df, symbol=symbol)
        return exog

    def add_metrics(self, df: pd.DataFrame, agg: str, transformation: str, regression_type: str = 'linear') -> None:
        for symbol in df.columns:
            self.add_metrics_per_symbol(symbol=symbol, transformation=transformation, regression_type=regression_type,
                                        df=df, agg=agg)


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
            ClustAM._kmeans.fit(tmp)
            ClustAM.silhouette.append(silhouette_score(tmp, ClustAM._kmeans.labels_))
        k_best = ClustAM.silhouette.index(max(ClustAM.silhouette)) + 2
        ClustAM._kmeans.n_clusters = k_best

    @staticmethod
    def build_clusters(df: pd.DataFrame) -> None:
        ClustAM.cluster(df)
        tmp = df.transpose().copy()
        ClustAM._kmeans.fit(tmp)
        tmp.loc[:, 'labels'] = ClustAM._kmeans.labels_
        ClustAM._cluster_group = tmp.groupby(by='labels')
        ClustAM._clusters_trained = True

    def cluster_members(self, symbol: str) -> typing.Union[str, typing.List[str]]:
        for cluster, members in ClustAM._cluster_group.groups.items():
            if symbol in members:
                members_ls = list(members)
        return symbol, members_ls

    def build_exog(self, symbol: typing.Union[str, typing.List[str]],
                   df: pd.DataFrame, regression_type: str, transformation: str) -> typing.Union[pd.DataFrame, str]:
        _, member_ls = self.cluster_members(symbol)
        exog = TrainingScheme._factory_model_type_dd[self._model_type].builder(F=self._F, df=df, symbol=member_ls)
        return exog

    def add_metrics(self, regression_type: str, transformation: str, agg: str, df: pd.DataFrame) -> None:
        if ClustAM._cluster_group is None:
            ClustAM.build_clusters(df)
        for symbol in self._universe:
            _, member_ls = self.cluster_members(symbol)
            self.add_metrics_per_symbol(symbol=symbol, df=df[member_ls], agg=agg, regression_type=regression_type,
                                        transformation=transformation, cluster=member_ls)

    @property
    def clusters_trained(self):
        return self._clusters_trained

    @property
    def cluster_group(self):
        return self._cluster_group


class CAM(TrainingScheme):

    def __init__(self, h: str, F: typing.List[str], L: str, Q: str, universe: typing.List[str], model_type: str=None,
                 s: str=None, b: str='5T'):
        super(CAM, self).__init__(h=h, F=F, L=L, Q=Q, universe=universe, model_type=model_type, s=s, b=b)
        self._feature_importance = None
        self._feature_importance_symbol = dict([(symbol, None) for symbol in universe])

    def build_exog(self, df: pd.DataFrame, transformation: str = None, **kwargs) \
            -> pd.DataFrame:
        exog = TrainingScheme._factory_model_type_dd[self._model_type].builder(F=self._F, df=df, symbol=df.columns)
        return exog

    def add_metrics(self, regression_type: str, transformation: str, agg: str, df: pd.DataFrame, ** kwargs) -> None:
        for symbol in self._universe:
            print(f'[Data Process]: Process for {symbol} has started...')
            self.add_metrics_per_symbol(symbol=symbol, df=df, agg=agg, regression_type=regression_type,
                                        transformation=transformation)

    @property
    def feature_importance(self):
        return self._feature_importance

    @property
    def feature_importance_symbol(self):
        return self._feature_importance_symbol

    @feature_importance.setter
    def feature_importance(self, df: pd.DataFrame):
        self._feature_importance = df


