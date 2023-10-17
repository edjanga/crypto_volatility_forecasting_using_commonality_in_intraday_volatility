import concurrent.futures
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
from data_centre.data import Reader
import sqlite3
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
import itertools
from figures.maker_copy import PlotResults

_data_centre_dir = os.path.abspath(__file__).replace('/model/lab.py', '/data_centre/databases')

"""Functions used to facilitate computation within classes."""
qlike_score = lambda x: ((x.iloc[:, 0].div(x.iloc[:, -1]))-np.log(x.iloc[:, 0].div(x.iloc[:, -1]))-1).mean()


def flatten(ls: list()) -> list:
    flat_ls = []
    for sub_ls in ls:
        flat_ls += sub_ls
    return flat_ls


class DMTest:

    _db_connect_y = sqlite3.connect(database=os.path.abspath(f'{_data_centre_dir}/y.db'), check_same_thread=False)

    def __init__(self):
        pass
    @staticmethod
    def table() -> pd.DataFrame: #L: str, training_scheme: str
        """
            Method computing a matrix with DM statistics for every (i,j) as entries
        """
        regression_per_training_scheme_dd = {'SAM': ['linear', 'lasso', 'elastic', 'pcr', 'lightgbm'],
                                             #'ClustAM': ['lasso', 'pcr', 'elastic', 'lightgbm'],
                                             'CAM': ['lasso', 'pcr', 'elastic', 'lightgbm']
                                            }
        regression = lambda x, scheme: (scheme, x[scheme])
        for L in ([['1W'], ['1M'], ['6M']]):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(regression, x=regression_per_training_scheme_dd, scheme=scheme) for
                           scheme in list(regression_per_training_scheme_dd.keys())]
                for future in concurrent.futures.as_completed(futures):
                    scheme, regression_ls = future.result()
                    for scheme, l, regression in itertools.product([scheme], L, regression_ls):
                        if regression != 'pcr':
                            # query = f'SELECT "timestamp", ("y"/"y_hat") * ( "y"-"y_hat") AS "e", "symbol" '\
                            #         f'FROM (SELECT "timestamp", "y", "y_hat", "symbol", "model", "L", "training_scheme",' \
                            #         f'"regression" FROM y_{L} WHERE "training_scheme" = \"{scheme}\" AND ' \
                            #         f'"regression" = \"{regression}\" AND "model" = "linear");'
                            continue
                        else:
                            models = \
                                ''.join(('(', (','.join([f'\"{model}\"' for model in ['ar', 'har', 'har_eq']])), ')'))
                            query = f'SELECT "timestamp", ("y"-"y_hat")*("y"-"y_hat") AS e, "symbol", "L",' \
                                    f'"training_scheme", "model", "regression" FROM ' \
                                    f'(SELECT "timestamp", "y", "y_hat", "symbol", "model", "L", "training_scheme",' \
                                    f'"regression" FROM y_{l} WHERE "training_scheme" = \"{scheme}\" AND ' \
                                    f'"regression" = \"{regression}\" AND "model" IN {models});'
                        try:
                            tmp = pd.read_sql(query, con=DMTest._db_connect_y, chunksize=10_000)
                        except Exception:
                            continue
                        tmp = pd.concat(list(tmp)).set_index('timestamp')
                        tmp.index = pd.to_datetime(tmp.index, utc=True)
                        tmp.sort_index(inplace=True)
                        tmp = tmp.assign(tag=['_'.join((training_scheme, L, regression, model)) for training_scheme, L,
                        regression, model in zip(tmp.training_scheme, tmp.L, tmp.regression, tmp.model)])
                        tmp.drop(['training_scheme', 'L', 'regression', 'model'], axis=1, inplace=True)
                        tmp = tmp.groupby(by=[pd.Grouper(key='symbol')])
                        if 'dm_table' not in locals():
                            dm_table = tmp.apply(lambda x: pd.pivot(x[['e', 'tag']], columns='tag', values='e'))
                        else:
                            dm_table = \
                                pd.concat(
                                    [dm_table, tmp.apply(lambda x: pd.pivot(x[['e', 'tag']],
                                                                            columns='tag', values='e'))], axis=1
                                )
            combination_tags = list(itertools.combinations(dm_table.columns.tolist(), r=2))
            models_dd = dict()
            return_pair = lambda x: x
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(return_pair, x=pair) for pair in combination_tags]
                for future in concurrent.futures.as_completed(futures):
                    pair = future.result()
                    tmp_dm = dm_table[list(pair)].apply(lambda x: x.diff(), axis=1).dropna(axis=1)
                    tmp_dm = tmp_dm.groupby(by=pd.Grouper(level='timestamp')).mean()
                    mean, std = tmp_dm.mean().values[0], tmp_dm.std().values[0]
                    dm = mean/std
                    models_dd[pair] = [dm]
            pdb.set_trace()
        dm_table = pd.melt(dm_table, var_name='tag', value_name='values', ignore_index=False)
        dm_table = dm_table.groupby(by=pd.Grouper(level='symbol'))
        pdb.set_trace()
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
    _commonality_df = None

    def __init__(self, L: str = None, transformation: str = 'log'):
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

    @L.setter
    def L(self, L: str):
        self._L = L

    def raw_commonality(self) -> None:
        commonality = list()
        for L in ['1W', '1M', '6M']:
            self._L = L
            print(f'[Computation]: Process for {self._L} lookback window has started...')
            tmp_commonality_df = pd.DataFrame(data=np.nan, index=list(Commonality._rv.resample('D').groups.keys()),
                                              columns=Commonality._rv.columns)
            dates = list(Commonality._rv.resample('1D').groups.keys())
            if 'D' not in self._L:
                days = int(self._L[:-1])*30 if 'M' in self._L else int(self._L[:-1])*7
            else:
                days = 1
            L_train = relativedelta(days=days)
            start = dates[0]+relativedelta(days=days)
            ls = list(itertools.product(dates[dates.index(start):], Commonality._rv.columns.tolist()))
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = \
                    [executor.submit(lambda x, y: (x.date(), y), x=pair[0], y=pair[1]) for pair in ls]
                for future in concurrent.futures.as_completed(futures):
                    date, symbol = future.result()
                    print(f'[Commonality]: Computation for {symbol} - {date.strftime("%Y-%m-%d")} has started...')
                    y = \
                        Commonality._rv.loc[
                            (Commonality._rv.index.date >= date-L_train) &
                            (Commonality._rv.index.date < date+relativedelta(days=1)), :]
                    y = y.assign(RV_MKT=y.mean(axis=1))
                    X = y.pop('RV_MKT')
                    X_train, y_train, X_test = X.loc[(X.index.date < date)], y.loc[(y.index.date < date), :],\
                        X.loc[(X.index.date == date)]
                    model_pred = \
                        lambda x_train, y_train, x_test: LinearRegression().fit(x_train, y_train).predict(x_test)
                    x_train, y_train, x_test = \
                        X_train.values.reshape(-1, 1), y_train[[symbol]].values.reshape(-1, 1),\
                            X_test.values.reshape(-1, 1)
                    y_hat = model_pred(x_train=x_train, y_train=y_train, x_test=x_test)
                    self._rv_fitted.loc[self._rv_fitted.index.date == date, symbol] = y_hat.reshape(-1)
                    tmp = pd.concat([Commonality._rv.loc[Commonality._rv.index.date == date, symbol],
                                     self._rv_fitted.loc[self._rv_fitted.index.date == date, symbol]], axis=1)
                    tmp = tmp.groupby(by=pd.Grouper(level=0, freq='D'))
                    r2 = tmp.apply(
                        lambda x: 1 - (1 - r2_score(x.iloc[:, 0], x.iloc[:, -1])) * (x.shape[0] - 1) / (x.shape[0] - 2)
                    )
                    tmp_commonality_df.loc[date, symbol] = r2.values[0]
                    print(f'[Commonality]: {symbol} has been computed - {date.strftime("%Y-%m-%d")}...')
            tmp_commonality_df.index = pd.to_datetime(tmp_commonality_df.index, utc=True)
            tmp_commonality_df = pd.melt(tmp_commonality_df, ignore_index=False, var_name='symbol', value_name='values')
            tmp_commonality_df = tmp_commonality_df.assign(L=L)
            commonality.append(tmp_commonality_df)
        commonality = pd.concat(commonality).dropna()
        commonality.to_sql(f'raw_commonality', con=Commonality._commonality_connect_db, if_exists='append')
        print(f'[Insertion]: Raw commonality table {self._L} has been inserted into the database')

    @staticmethod
    def commonality() -> None:
        query = f'SELECT * FROM raw_commonality;'
        commonality_df = pd.read_sql(query, con=Commonality._commonality_connect_db, index_col='index')
        commonality_df.index = pd.to_datetime(commonality_df.index)
        commonality_df = \
            commonality_df.groupby(
                by=[pd.Grouper(key='L'), pd.Grouper(level='index', freq='1M')]).agg({'values': 'mean'})
        commonality_df.reset_index(inplace=True)
        commonality_df.to_sql('commonality', con=Commonality._commonality_connect_db, if_exists='replace')
        print(f'[Insertion]: Commonality table has been inserted into the database')


if __name__ == '__main__':
    # commonality_obj = Commonality()
    # commonality_obj.commonality()
    plot_results_obj = PlotResults()
    plot_results_obj.commonality(save=False)