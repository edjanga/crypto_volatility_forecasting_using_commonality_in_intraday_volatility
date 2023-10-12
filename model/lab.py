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
    def table(L: str, training_scheme: str) -> pd.DataFrame:
        """
            Method computing a matrix with DM statistics for every (i,j) as entries
        """
        regression_per_training_scheme_dd = {'SAM': ['linear', 'lasso', 'elastic', 'pcr', 'lightgbm'], #'ridge'
                                             'ClustAM': ['lasso', 'pcr', 'elastic', 'lightgbm'], #'ridge'
                                             'CAM': ['lasso', 'pcr', 'elastic', 'lightgbm']} #'ridge'
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

