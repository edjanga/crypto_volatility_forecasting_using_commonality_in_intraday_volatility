import concurrent.futures
import pdb
import typing
import sqlite3
import os
import pandas as pd
from model.training_schemes import ClustAM, CAM, TrainingScheme
from data_centre.data import Reader
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
from dateutil.relativedelta import relativedelta
import numpy as np
import optuna
import lightgbm as lgb
from optuna.samplers import RandomSampler
from datetime import datetime
import time


def objective(trial):
    regression_type = trial.study.study_name.split('_')[3]
    if regression_type == 'lightgbm':
        train_loader = lgb.Dataset(X_train, label=y_train)
        valid_loader = lgb.Dataset(X_valid, label=y_valid)
        param = {
            'objective': 'regression', 'metric': 'mse', 'verbosity': -1,
            'max_depth': trial.suggest_int('max_depth', 1, 3), 'lr': trial.suggest_float('lr', .01, .1),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        }
        try:
            tmp_rres = lgb.train(param, train_set=train_loader, valid_sets=[valid_loader],
                                 num_boost_round=10,
                                 callbacks=[lgb.early_stopping(5, first_metric_only=True, verbose=True, min_delta=0.0)])
        except lgb.basic.LightGBMError as e:
            print(e)
            pdb.set_trace()
    elif regression_type == 'lasso':
        param = {'objective': 'regression', 'metric': 'mse', 'verbosity': -1,
                 'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True)
                 }
        tmp_rres = Lasso(alpha=param['alpha'])
    if regression_type != 'lightgbm':
        try:
            tmp_rres.fit(X_train, y_train)
        except UnboundLocalError:
            pdb.set_trace()
    loss = mean_squared_error(y_valid, tmp_rres.predict(X_valid))
    trial.set_user_attr(key='best_estimator', value=tmp_rres)
    return loss


def callback(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key='best_estimator', value=trial.user_attrs['best_estimator'])


class TrainingSchemeAnalysis:

    """
        Class providing extra analysis for some training schemes.
        Available now:
            - 1st principal component for ClustAM for clusters for which size > 1 and CAM
            - Feature importance for decision tree based models.
    """
    _data_centre_dir = \
        os.path.abspath(__file__).replace('/model/extra_regression_analysis.py', '/data_centre/databases')

    _ClustAM_obj = ClustAM
    _CAM_obj = CAM
    _reader_obj = Reader()
    _pca_obj = PCA(n_components=1)
    _linear_reg_obj = LinearRegression
    _lasso_obj = Lasso()
    _db_feature_importance = sqlite3.connect(
        database=os.path.abspath(f'{_data_centre_dir}/feature_importance.db'), check_same_thread=False
    )

    def __init__(self, L: str=None, training_scheme: str=None):
        self._L = L
        self._training_scheme = training_scheme
        self._rv = TrainingSchemeAnalysis._reader_obj.rv_read()

    @property
    def L(self):
        return self._L

    @L.setter
    def L(self, L):
        self._L = L

    @property
    def training_scheme(self):
        return self._training_scheme

    @training_scheme.setter
    def training_scheme(self, training_scheme):
        self._training_scheme = training_scheme

    @property
    def rv(self):
        return self._rv

    def coefficient_analysis_per_date(self, date: datetime, df: pd.DataFrame, regression_type: str,
                                      coefficient: pd.DataFrame, const: bool=False):
        train_index = list(set(df.index[(df.index.date >= date - L_train) & (df.index.date < date)]))
        train_index.sort()
        test_index = list(set(df.index[df.index.date == date]))
        test_index.sort()
        if regression_type != 'linear':
            valid_index = train_index[-len(train_index) // 5:]
            train_index = train_index[:-len(train_index) // 5]
            global X_train
            global y_train
            global X_valid
            global y_valid
            X_train = df.loc[train_index, :].copy()
            X_train = X_train.assign(FIRST_COMP=TrainingSchemeAnalysis._pca_obj.fit_transform(X_train))
            y_train = X_train.pop('FIRST_COMP')
            X_valid = df.loc[valid_index, :].copy()
            X_valid = X_valid.assign(FIRST_COMP=TrainingSchemeAnalysis._pca_obj.transform(X_valid))
            y_valid = X_valid.pop('FIRST_COMP')
        else:
            X_train = df.loc[train_index, :].copy()
            X_train = X_train.assign(FIRST_COMP=TrainingSchemeAnalysis._pca_obj.fit_transform(X_train))
            y_train = X_train.pop('FIRST_COMP')
        global X_test
        global y_test
        X_test = df.loc[test_index, :].copy()
        X_test = X_test.assign(FIRST_COMP=TrainingSchemeAnalysis._pca_obj.transform(X_test))
        y_test = X_test.pop('FIRST_COMP')
        if regression_type != 'linear':
            study = optuna.create_study(
                direction='minimize', sampler=RandomSampler(123),
                study_name=
                f'{"_".join((self._training_scheme, self._L, "log", regression_type, date.strftime("%Y-%m-%d")))}'
            )
            n_trials = 5 if regression_type not in ['xgboost', 'lightgbm'] else 1
            study.optimize(objective, n_trials=n_trials, callbacks=[callback], n_jobs=-1)
            rres = study.user_attrs['best_estimator']
        else:
            rres = TrainingSchemeAnalysis._linear_reg_obj(fit_intercept=False)
        rres.fit(X_train, y_train)
        if const:
            coefficient.loc[date, 'const'], coefficient.loc[date, df.columns] = rres.intercept_, rres.coef_
        coefficient.loc[date, df.columns] = rres.coef_

    # @staticmethod
    # def df_per_day(df: pd.DataFrame, date: datetime):
    #     return date, df.loc[(df.index.date >= date - L_train) & (df.index.date <= date)]

    def coefficient_analysis(self, regression_type: str ='linear', const: bool=False):
        df = self._rv.copy()
        if self._training_scheme == 'ClustAM':
            self._ClustAM_obj.build_clusters(df=df)
        dates = list(np.unique(df.index.date))
        global L_train
        L_train = relativedelta(minutes=TrainingScheme.L_shift_dd[self._L] * 5)
        start = dates[0] + L_train
        coefficient = pd.DataFrame(index=dates, columns=df.columns.tolist(), data=np.nan)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            if self._training_scheme == 'ClustAM':
                for cluster, member in self._ClustAM_obj._cluster_group.groups.items():
                    if len(member) > 1:
                        futures = [executor.submit(TrainingScheme.df_per_day, df=df[list(member)], date=date)
                                   for date in dates[dates.index(start):]]
            else:
                futures = [executor.submit(TrainingScheme.df_per_day, df=df, date=date) for date in
                           dates[dates.index(start):]]
            for future in concurrent.futures.as_completed(futures):
                date, date_df = future.result()
                self.coefficient_analysis_per_date(regression_type=regression_type, date=date, coefficient=coefficient,
                                                   df=date_df, const=const)
            coefficient.dropna(inplace=True, axis=1, how='all')
            coefficient.dropna(inplace=True)
            coefficient = pd.melt(coefficient, var_name='variable', value_name='values', ignore_index=False)
            coefficient = coefficient.assign(L=self._L, training_scheme=self._training_scheme)
            coefficient.to_sql(name=f'coefficient_1st_principal_component', con=TrainingScheme._db_connect_pca,
                               if_exists='append')
            print(f'[DATA INSERTION]: coefficient_{self._training_scheme}_{self._L}_1st_principal_component '
                  f'has been inserted.')

    def feature_imp_per_date(self, model_type: str, date: datetime, symbol: str, feature_imp_per_symbol: pd.DataFrame,
                             df: pd.DataFrame, transformation: str = 'log'):
        train_index = list(set(df.index[(df.index.date >= date - L_train) & (df.index.date < date)]))
        train_index.sort()
        test_index = list(set(df.index[df.index.date == date]))
        test_index.sort()
        valid_index = train_index[-len(train_index) // 5:]
        train_index = train_index[:-len(train_index) // 5]
        X = df.loc[train_index + valid_index + test_index, :].copy()
        y = X.pop(symbol)
        global X_train
        global X_valid
        global X_test
        global y_train
        global y_valid
        global y_test
        X_train, X_valid = X.loc[train_index, :], X.loc[valid_index, :]
        y_train, y_valid = y.loc[train_index], y.loc[valid_index]
        X_test, y_test = X.loc[test_index, :], y.loc[test_index]
        study = optuna.create_study(direction='minimize', sampler=RandomSampler(123),
                                    study_name='_'.join((
                                        self._training_scheme, self._L, transformation, 'lightgbm', model_type, symbol,
                                        date.strftime('%Y-%m-%d'))
                                    ))
        study.optimize(objective, n_trials=1, callbacks=[callback], n_jobs=-1)
        rres = study.user_attrs['best_estimator']
        feature_imp_per_symbol.loc[date, rres.feature_name()] = rres.feature_importance()

    def feature_imp_symbol(self, h: str, F: typing.List[str], model_type: str, universe: typing.List[str], symbol: str,
                           training_scheme_obj: TrainingScheme, Q='1D', transformation: str = 'log') -> None:
        feature_imp = pd.DataFrame()
        global L_train
        L_train = relativedelta(minutes=TrainingScheme.L_shift_dd[training_scheme_obj.L] * 5)
        if self._training_scheme == 'ClustAM':
            _, member_ls = training_scheme_obj.cluster_members(symbol)
            df = training_scheme_obj.build_exog(symbol=member_ls, df=self._rv[member_ls].copy(),
                                                regression_type='lightgbm', transformation=transformation)
        else:
            df = training_scheme_obj.build_exog(df=self._rv.copy(), transformation=transformation,
                                                regression_type='lightgbm')
        if 'dates' not in locals():
            dates = list(np.unique(df.index.date))
            start = dates[0] + L_train
        feature_imp_per_symbol = pd.DataFrame(index=dates, columns=df.columns.tolist(), data=np.nan)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(TrainingScheme.df_per_day.df_per_day, date=date, df=df) for date
                       in dates[dates.index(start):]]
            for future in concurrent.futures.as_completed(futures):
                date, date_df = future.result()
                self.feature_imp_per_date(model_type=model_type, date=date, symbol=symbol,
                                          feature_imp_per_symbol=feature_imp_per_symbol, df=date_df,
                                          transformation=transformation)
            feature_imp_per_symbol.dropna(how='all', inplace=True)
            feature_imp_per_symbol = \
                pd.melt(feature_imp_per_symbol, var_name='feature', value_name='importance',
                        ignore_index=False)
            feature_imp_per_symbol = \
                feature_imp_per_symbol.assign(L=self._L, model_type=model_type, transformation=transformation,
                                              training_scheme=self._training_scheme)
            feature_imp_dd[symbol] = feature_imp_per_symbol

    def feature_imp(self, h: str, F: typing.List[str], model_type: str, universe: typing.List[str],
                    Q='1D', transformation: str = 'log') -> None:
        training_scheme_obj = \
            TrainingSchemeAnalysis._ClustAM_obj if self._training_scheme == 'ClustAM' else \
                TrainingSchemeAnalysis._CAM_obj
        training_scheme_obj = training_scheme_obj(h=h, F=F, model_type=model_type, universe=universe, Q=Q, L=self._L)
        if self._training_scheme == 'ClustAM':
            if not training_scheme_obj.clusters_trained:
                training_scheme_obj.build_clusters(df=self._rv.copy())
        global feature_imp_dd
        feature_imp_dd = {symbol: pd.DataFrame() for symbol in universe}
        for symbol in universe:
            self.feature_imp_symbol(h=h, F=F, model_type=model_type, universe=universe, symbol=symbol, Q=Q,
                                    transformation=transformation, training_scheme_obj=training_scheme_obj)
        feature_imp = pd.concat(feature_imp_dd.values())
        feature_imp.dropna(inplace=True)
        feature_imp = feature_imp.groupby(
            by=[pd.Grouper(key='feature')]).agg(
            {'importance': 'mean', 'training_scheme': 'last', 'model_type': 'last',
             'transformation': 'last'}
        )
        feature_imp.to_sql(name=self._training_scheme, con=self._db_feature_importance, if_exists='append')
        print(
            f'[DATA INSERTION]: Feature importance for {self._training_scheme}_{self._L}_{model_type}'
            f'_{transformation} has been inserted.'
        )
    @staticmethod
    def lags_and_model_type(F: typing.List[str], model_type: str):
        return model_type, [F[0]] if model_type in ['ar', 'risk_metrics'] else F

    @property
    def reader_obj(self):
        return self._reader_obj