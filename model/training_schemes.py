import concurrent.futures
import pdb
import typing
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error, silhouette_score
from sklearn.cluster import KMeans
from datetime import datetime
import os
from dateutil.relativedelta import relativedelta
from data_centre.helpers import coin_ls
from data_centre.data import Reader
from scipy.stats import t
import sqlite3
from model.feature_engineering_room import FeatureAR, FeatureHAR, FeatureHAREq, FeatureRiskMetricsEstimator
import numpy as np
import optuna
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from model.lab import qlike_score, EarlyStopping
from optuna.samplers import RandomSampler


# class EarlyStopping:
#     """Early stops the training if validation loss doesn't improve after a given patience."""
#     def __init__(self, nn: bool = True, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
#         """
#         Args:
#             patience (int): How long to wait after last time validation loss improved.
#                             Default: 7
#             verbose (bool): If True, prints a message for each validation loss improvement.
#                             Default: False
#             delta (float): Minimum change in the monitored quantity to qualify as an improvement.
#                             Default: 0
#             path (str): Path for the checkpoint to be saved to.
#                             Default: 'checkpoint.pt'
#             trace_func (function): trace print function.
#                             Default: print
#         """
#         self.patience = patience
#         self.verbose = verbose
#         self.counter = 0
#         self.nn = nn
#         self.best_model = None
#         self.best_score = None
#         self.early_stop = False
#         self.val_loss_min = np.Inf
#         self.delta = delta
#         self.path = path
#         self.trace_func = trace_func
#
#     def __call__(self, val_loss, model):
#
#         score = -val_loss
#
#         if self.best_score is None:
#             self.best_score = score
#             if self.nn:
#                 self.save_checkpoint(val_loss, model)
#             else:
#                 self.best_model = model
#         elif score < self.best_score + self.delta:
#             self.counter += 1
#             self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_score = score
#             if self.nn:
#                 self.save_checkpoint(val_loss, model)
#             else:
#                 self.best_model = model
#             self.counter = 0
#
#     def save_checkpoint(self, val_loss, model):
#         '''Saves model when validation loss decrease.'''
#         if self.verbose:
#             self.trace_func(
#                 f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
#         if self.nn:
#             torch.save(model.state_dict(), self.path)
#         else:
#             self.best_model = model
#         self.val_loss_min = val_loss


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
    _factory_model_type_dd = {'ar': FeatureAR(), 'risk_metrics': FeatureRiskMetricsEstimator(), 'har': FeatureHAR(),
                              'har_eq': FeatureHAREq()}

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
        stats_condition = True
        if (self._model_type == 'risk_metrics') & (self.__class__.__name__ == 'SAM') & (regression_type == 'linear'):
            stats_condition = False
        elif regression_type == 'lightgbm':
            stats_condition = False
        return stats_condition

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
                'max_depth': trial.suggest_int('max_depth', 1, 3),
                'lr': trial.suggest_float('lr', .01, .1),
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            }
            tmp_rres = lgb.train(param, train_set=train_loader, valid_sets=[valid_loader],
                                 num_boost_round=10,
                                 callbacks=[lgb.early_stopping(5, first_metric_only=True, verbose=True, min_delta=0.0)])
        elif regression_type == 'lasso':
            param = {
                'objective': 'regression', 'metric': 'mse', 'verbosity': -1,
                'alpha': trial.suggest_float('alpha', .01, .99, log=False)
            }
            tmp_rres = Lasso(alpha=param['alpha'], fit_intercept=training_scheme_name != 'UAM', max_iter=500)
        elif regression_type == 'ridge':
            param = {
                'objective': 'regression', 'metric': 'mse', 'verbosity': -1,
                'alpha': trial.suggest_float('alpha', .01, .99, log=False),
            }
            tmp_rres = Ridge(alpha=param['alpha'], fit_intercept=training_scheme_name != 'UAM')
        elif regression_type == 'elastic':
            param = {
                'objective': 'regression', 'metric': 'mse', 'verbosity': -1,
                'alpha': trial.suggest_float('alpha', .01, .99, log=False),
                'l1_ratio': trial.suggest_float('l1_ratio', .01, .99, log=False),
            }
            tmp_rres = ElasticNet(alpha=param['alpha'], l1_ratio=param['l1_ratio'],
                                  fit_intercept=training_scheme_name != 'UAM', max_iter=500)
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
        feature_obj = TrainingScheme._factory_model_type_dd[self._model_type]
        train_index = list(set(exog.index[(exog.index.date >= date - L_train) & (exog.index.date < date)]))
        train_index.sort()
        test_index = list(set(exog.index[exog.index.date == date]))
        test_index.sort()
        global X_train
        global y_train
        global X_test
        global y_test
        if (self._model_type == 'risk_metrics') & (self.__class__.__name__ == 'SAM'):
            y_hat = pd.concat([exog.loc[train_index],
                               pd.DataFrame(data=np.nan, index=exog.loc[test_index].index,
                                            columns=exog.columns)])
            y_hat = y_hat.ewm(alpha=feature_obj.factor).mean().loc[L_train:].iloc[:, 0]
            y_hat.name = 0
            y_test = endog.loc[test_index]
        else:
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
                n_trials = 5 if regression_type not in ['xgboost', 'lightgbm'] else 1
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
                    pdb.set_trace()
                    try:
                        X_train = \
                            pd.DataFrame(rres_pca.fit_transform(X_train),
                                         columns=columns_name_dd['pcr' == regression_type][:n_components],
                                         index=X_train.index)
                    except Exception as e:
                        print(e)
                        pdb.set_trace()
                    X_train = pd.concat([X_train, own_train], axis=1)
                    X_test = pd.DataFrame(rres_pca.transform(X_test),
                                          columns=columns_name_dd['pcr' == regression_type][:n_components],
                                          index=X_test.index)

                    X_test = pd.concat([X_test, own_test], axis=1)
                rres.fit(X_train, y_train)
            y_hat = pd.Series(data=rres.predict(X_test), index=y_test.index)
            # """ Coefficients """
            # if regression_type != 'lightgbm':
            #     daily_coefficient = np.concatenate((np.array([rres.intercept_]), rres.coef_))
            #     coefficient.loc[exog.index[test_index[0]], :] = daily_coefficient
            #     coefficient.loc[exog.index[test_index[0]], :].fillna(0, inplace=True)
            #     columns_ls = list(set(X_train.columns).intersection(set(coefficient.columns)))
            #     columns_ls.sort()
            #     X_train = X_train.assign(const=1)
            #     X_train = X_train[['const'] + columns_ls]
            #     XtX = np.dot(X_train.transpose().values, X_train.values)
            #     std_err = np.zeros(coefficient.shape[1])
            #     std_err[:X_train.shape[1]] = np.diag(XtX) ** (.5) / np.sqrt(X_train.shape[0])
            #     tstats.loc[exog.index[test_index[0]], :] = \
            #         coefficient.loc[exog.index[test_index[0]], :].div(std_err)
            #     pvalues.loc[exog.index[test_index[0]], :] = 2 * t.pdf(tstats.loc[exog.index[test_index[0]], :],
            #                                                           df=L_train - 1)
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
        stats_condition = self.stats_condition(regression_type=regression_type)
        if self.__class__.__name__ != 'UAM':
            if self._model_type == 'har_eq':
                exog.loc[:, ~exog.columns.str.contains('_SESSION')] = \
                    self._factory_transformation_dd[transformation]['transformation'](
                        exog.loc[:, ~exog.columns.str.contains('_SESSION')])
            else:
                exog = self._factory_transformation_dd[transformation]['transformation'](exog)
        endog = exog.pop(symbol)
        y = list()
        dates = list(np.unique(exog.index.date))
        L_train = relativedelta(minutes=TrainingScheme.L_shift_dd[self._L] * 5)
        start = dates[0] + L_train
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.rolling_metrics_per_date,
                                       exog=exog, endog=endog, date=date, regression_type=regression_type,
                                       transformation=transformation, **kwargs)
                       for date in dates[dates.index(start):]]
            for future in concurrent.futures.as_completed(futures):
                y.append(future.result().sort_index())
            y = pd.concat(y).dropna()
            if self.__class__.__name__ != 'UAM':
                y = TrainingScheme._factory_transformation_dd[transformation]['inverse'](y).reset_index(
                    names='timestamp')
                y['symbol'] = symbol
            else:
                y[['RV', 0]] = TrainingScheme._factory_transformation_dd[transformation]['inverse'](y[['RV', 0]])
                y.reset_index(inplace=True)
            y = y.groupby(by=[pd.Grouper(key='symbol'), pd.Grouper(key='timestamp', freq=self._h)]).sum()
            tmp = y.groupby(by=[pd.Grouper(level='symbol'), pd.Grouper(level='timestamp', freq=agg)])
            mse = tmp.apply(lambda x: mean_squared_error(x.iloc[:, 0], x.iloc[:, -1]))
            qlike = tmp.apply(qlike_score)
            r2 = tmp.apply(lambda x: r2_score(x.iloc[:, 0], x.iloc[:, -1]))
            if self.__class__.__name__ == 'UAM':
                mse = mse.groupby(by=pd.Grouper(level='timestamp')).sum()
                r2 = r2.groupby(by=pd.Grouper(level='timestamp')).mean()
                qlike = qlike.groupby(by=pd.Grouper(level='timestamp')).mean()
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
            transformation_dd = {'log': 'log', None: 'level'}
            for table_name, table in table_dd.items():
                if table_name != 'y':
                    table['symbol'] = symbol
                    table['model'] = self._model_type
                table['L'] = self._L
                table['training_scheme'] = self.__class__.__name__
                table['transformation'] = transformation_dd[transformation]
                table['regression'] = regression_type
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
        # stats_condition = self.stats_condition(regression_type=regression_type)
        # if stats_condition:
        #     tstats = self._training_scheme_tstats_dd[symbol]
        #     pvalues = self._training_scheme_pvalues_dd[symbol]
        #     coefficient = self._training_scheme_coefficient_dd[symbol]
        #     con_dd.update({'tstats': TrainingScheme._db_connect_tstats,
        #                    'pvalues': TrainingScheme._db_connect_pvalues,
        #                    'coefficient': TrainingScheme._db_connect_coefficient})
        #     table_dd.update({'tstats': tstats, 'pvalues': pvalues, 'coefficient': coefficient})
        count = 1
        for table_name, table in table_dd.items():
            table.drop('symbol', inplace=True, axis=1)
            if table_name != 'y':
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
            self.add_metrics_per_symbol(symbol=symbol, transformation=transformation,
                                        regression_type=regression_type, df=df, agg=agg)


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
        exog = TrainingScheme._factory_model_type_dd[self._model_type].builder(F=self._F, df=df, symbol=symbol)
        return exog

    # def add_metrics(self, regression_type: str, transformation: str, agg: str, df: pd.DataFrame) -> None:
    #     # for _, symbol in enumerate(self._universe):
    #     #     if ClustAM._cluster_group is None:
    #     #         ClustAM.build_clusters(df)
    #     #     exog = self.build_exog(df=df, symbol=symbol, regression_type=regression_type, transformation=transformation)
    #     #     self.add_metrics_per_symbol(symbol=symbol, df=exog, agg=agg, regression_type=regression_type,
    #     #                                 transformation=transformation)
    #
    #     #exog = self.build_exog(df=df, symbol=symbol, regression_type=regression_type, transformation=transformation)
    #     if ClustAM._cluster_group is None:
    #         ClustAM.build_clusters(df)
    #     with concurrent.futures.ThreadPoolExecutor() as executor:
    #         # futures = [executor.submit(self.build_exog, df=df, symbol=symbol, regression_type=regression_type,
    #         #                            transformation=transformation) for symbol in self._universe]
    #         futures = [executor.submit(self.cluser_members, symbol='DOGEUSDT') for symbol in self._universe]
    #         for future in concurrent.futures.as_completed(futures):
    #             # exog, symbol = future.result()
    #             member_ls = future.result()
    #             pdb.set_trace()
    #             # self.add_metrics_per_symbol_copy(symbol=symbol, df=exog, agg=agg, regression_type=regression_type,
    #             #                                  transformation=transformation)
    #             self.add_metrics_per_symbol_copy(symbol=member_ls, df=df, agg=agg, regression_type=regression_type,
    #                                              transformation=transformation)

    def add_metrics(self, regression_type: str, transformation: str, agg: str, df: pd.DataFrame) -> None:
        if ClustAM._cluster_group is None:
            ClustAM.build_clusters(df)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(lambda x: self.cluster_members(x), 'DOGEUSDT') for symbol in self._universe]
            for future in concurrent.futures.as_completed(futures):
                symbol, member_ls = list(future.result())
                self.add_metrics_per_symbol(symbol=symbol, df=df[member_ls], agg=agg, regression_type=regression_type,
                                            transformation=transformation, cluster=member_ls)

    @property
    def clusters_trained(self):
        return self._clusters_trained


class CAM(TrainingScheme):

    def build_exog(self, df: pd.DataFrame, transformation: str = None, **kwargs) \
            -> pd.DataFrame:
        exog = TrainingScheme._factory_model_type_dd[self._model_type].builder(F=self._F, df=df, symbol=df.columns)
        return exog

    # def add_metrics(self, regression_type: str, transformation: str, agg: str, df: pd.DataFrame, ** kwargs) -> None:
    #     #for symbol in self._universe:
    #     with concurrent.futures.ThreadPoolExecutor() as executor:
    #         futures = [executor.submit(self.add_metrics_per_symbol, symbol=symbol, df=df, agg=agg,
    #                                    regression_type=regression_type, transformation=transformation)
    #                    for symbol in self._universe]

    def add_metrics(self, regression_type: str, transformation: str, agg: str, df: pd.DataFrame, ** kwargs) -> None:
        for symbol in self._universe:
            print(f'[DATA PROCESS]: Process for {symbol} has started...')
            self.add_metrics_per_symbol(symbol=symbol, df=df, agg=agg, regression_type=regression_type,
                                        transformation=transformation)

        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     futures = [executor.submit(self.add_metrics_per_symbol_copy, symbol=symbol, df=df, agg=agg,
        #                                regression_type=regression_type, transformation=transformation)
        #                for symbol in self._universe]
        #     for future in concurrent.futures.as_completed(futures):



if __name__ == '__main__':

    reader_obj = Reader()
    df = reader_obj.rv_read().iloc[:, :20]
    universe = df.columns.tolist()
    training_scheme = 'SAM'
    model_type = 'har'
    h = '30T'
    F = ['30T', '1H', '6H', '12H']
    L = '6M'
    lookback_ls = ['1D', '1W', '1M', '6M']
    lookback_ls = \
        lookback_ls[lookback_ls.index(L):1] if lookback_ls.index(L) == 0 else lookback_ls[0:lookback_ls.index(L) + 1]
    F = F + lookback_ls if model_type not in ['ar', 'risk_metrics'] else [F[0]]
    training_scheme_factory_dd = {'SAM': SAM}
    transformation = 'log'
    training_scheme_obj = training_scheme_factory_dd[training_scheme](h=h, F=F, L=L, Q='1D', model_type=model_type,
                                                                      universe=universe)
    training_scheme_obj.add_metrics(agg='1W', transformation=transformation, regression_type='har_eq', df=df)


