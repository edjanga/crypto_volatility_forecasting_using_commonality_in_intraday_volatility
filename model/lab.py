import datetime
import pdb
import typing
import pandas as pd
import pytz
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
import os
from pytz import timezone
from dataclasses import dataclass
from datetime import time
from dateutil.relativedelta import relativedelta
import numpy as np
import copy
from scipy.stats.mstats import winsorize
from data_centre.helpers import coin_ls
import concurrent.futures
from data_centre.data import Reader
from hottbox.pdtools import pd_to_tensor
from itertools import product
from scipy.stats import t
import sqlite3
import matplotlib.pyplot as plt
from datetime import datetime


"""Functions used to facilitate computation within classes."""
qlike_score = lambda x: ((x.iloc[:, 0].div(x.iloc[:, -1]))-np.log(x.iloc[:, 0].div(x.iloc[:, -1]))-1).mean()


def group_members_external_df(df: pd.DataFrame, df2: pd.DataFrame, drop_symbol: bool=True) -> pd.DataFrame:
    if drop_symbol:
        return df.droplevel(1).mul(df2)
    else:
        return df.mul(df2)


def har_features_puzzle(df: pd.DataFrame, F: typing.Union[str, typing.List[str]], own=True) -> pd.DataFrame:
    symbol = df.symbol.unique()[0]
    for _, lookback in enumerate(F):
        shift = \
            pd.to_timedelta(lookback)//pd.to_timedelta('5T') if \
                'M' not in lookback else pd.to_timedelta('30D')//pd.to_timedelta('5T')
        tmp = df['rv']
        tmp.name = lookback if own else '_'.join((symbol, lookback))
        df = df.join(tmp.shift(shift), how='left')
    df.drop('rv', axis=1, inplace=True)
    df.dropna(inplace=True)
    df = df.reset_index().set_index(['symbol', 'timestamp'])
    if not own:
        df = df.droplevel(axis=0, level=0)
    return df


@dataclass
class Market:
    """
        Dataclass that describes markets with a timezone object.
    """
    asia_tz = timezone('Asia/Tokyo')
    us_tz = timezone('US/Eastern')
    uk_tz = timezone('UTC')


class FeatureBuilderBase:
    #pd.DateOffset(hours=1))

    """Lookback windows smaller than 1D are dynamic while the rest is static"""
    _lookback_window_dd = dict([('5T', pd.to_timedelta('5T')//pd.to_timedelta('5T')),
                                ('1H', pd.to_timedelta('1H')//pd.to_timedelta('5T')),
                                ('6H', pd.to_timedelta('6H')//pd.to_timedelta('5T')),
                                ('12H', pd.to_timedelta('12H')//pd.to_timedelta('5T')),
                                ('1D', pd.to_timedelta('1D')//pd.to_timedelta('5T')),
                                ('1W', lambda x:x.resample('W').mean()), ('1M', lambda x:x.resample('M').mean())])


    _5min_buckets_lookback_window_dd = \
        {lookback: pd.to_timedelta(lookback) // pd.to_timedelta('5T') for
         lookback in _lookback_window_dd.keys() if lookback != '1M'}

    #Add manually as pd.to_timedelta does not take '1M' as argument
    _5min_buckets_lookback_window_dd['1M'] = pd.to_timedelta('30D') // pd.to_timedelta('5T')

    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    def builder(self, symbol: typing.Union[typing.Tuple[str], str], df: pd.DataFrame, F: typing.List[str]):
        """To be overwritten by each child class"""
        pass


class FeatureCovariance(FeatureBuilderBase):

    def __init__(self):
        super().__init__('covariance_ar')

    def builder(self, F: typing.Union[typing.List[str], str],
                df: pd.DataFrame) -> pd.DataFrame:

        if isinstance(F, str):
            F = [F]
        window = pd.to_timedelta('30D') // pd.to_timedelta('5T') if \
            F[-1] == '1M' else max(4, pd.to_timedelta(F[-1]) // pd.to_timedelta('5T'))
        covariance = \
            df.rolling(window=window).cov().dropna().droplevel(axis=0, level=1).mean(axis=1)
        covariance = covariance.groupby(by=covariance.index).mean()
        covariance.name = 'COV'
        covariance = pd.DataFrame(covariance)
        for _, lookback in enumerate(F):
            if self._5min_buckets_lookback_window_dd[lookback] <= 288:
                offset = self._lookback_window_dd[lookback]
                covariance = covariance.join(covariance.shift(offset), how='left', rsuffix=f'_{lookback}')
        covariance.ffill(inplace=True)
        covariance.dropna(inplace=True)
        return covariance


class FeatureRiskMetricsEstimator(FeatureBuilderBase):

    data_obj = Reader(file='../data_centre/tmp/aggregate2022')

    def __init__(self):
        super().__init__('risk_metrics')

    def builder(self, F: typing.Union[typing.List[str], str],
                df: pd.DataFrame, symbol: typing.Union[str, typing.List[str]]) -> pd.DataFrame:

        if isinstance(F, str):
            F = [F]
        if isinstance(symbol, str):
            symbol = (symbol, symbol)
        list_symbol = list(dict.fromkeys(symbol).keys())
        window = pd.to_timedelta('30D') // pd.to_timedelta('5T') if \
            F[-1] == '1M' else max(4, pd.to_timedelta(F[-1]) // pd.to_timedelta('5T'))
        symbol_returns_df = \
            df[list(set([symbol[-1]]))].copy().rename(columns={symbol[-1]: '_'.join((symbol[-1], 'RET'))})**2
        symbol_rv_df = FeatureRiskMetricsEstimator.data_obj.rv_read(raw=False, symbol=list_symbol)
        symbol_rv_df = \
            symbol_rv_df[list_symbol].rename(columns={sym: '_'.join((sym, 'RV')) for _, sym in enumerate(list_symbol)})
        symbol_rv_df[f'{"_".join((symbol[-1], "RV", F[-1]))}'] = \
        symbol_rv_df[f'{"_".join((symbol[-1], "RV"))}'].shift(pd.to_timedelta(F[-1])//pd.to_timedelta('5T'))
        symbol_df = pd.concat([symbol_rv_df, symbol_returns_df], axis=1).dropna()
        return symbol_df


class FeatureHAR(FeatureBuilderBase):

    def __init__(self):
        super().__init__('har')

    def builder(self, symbol: typing.Union[typing.Tuple[str], str],
                df: pd.DataFrame, F: typing.List[str]) -> pd.DataFrame:
        if isinstance(symbol, str):
            symbol = (symbol, symbol)
        list_symbol = list(dict.fromkeys(symbol).keys())
        symbol_df = df[list_symbol].copy()
        for _, lookback in enumerate(F):
            if self._5min_buckets_lookback_window_dd[lookback] <= 288:
                offset = self._lookback_window_dd[lookback]
                symbol_df = symbol_df.join(symbol_df[[symbol[-1]]].shift(offset), how='left',
                                           rsuffix=f'_{lookback}')
            else:
                symbol_df = symbol_df.join(symbol_df[[symbol[-1]]].apply(self._lookback_window_dd[lookback]).shift(1),
                                           how='left', rsuffix=f'_{lookback}')
        symbol_df.ffill(inplace=True)
        symbol_df.dropna(inplace=True)
        return symbol_df


class FeatureHARDummy(FeatureBuilderBase):

    def __init__(self):
        super().__init__('har_dummy_markets')
        self._markets = Market()

    @property
    def markets(self):
        return self._markets

    def builder(self, symbol: typing.Union[typing.Tuple[str], str], df: pd.DataFrame,
                F: typing.List[str]) -> pd.DataFrame:
        if isinstance(symbol, str):
            symbol = (symbol, symbol)
        list_symbol = list(dict.fromkeys(symbol).keys())
        symbol_df = df[list_symbol].copy()
        for _, lookback in enumerate(F):
            if self._5min_buckets_lookback_window_dd[lookback] <= 288:
                offset = self._lookback_window_dd[lookback]
                symbol_df = symbol_df.join(symbol_df[[symbol[-1]]].shift(offset), how='left',
                                           rsuffix=f'_{lookback}')
            else:
                symbol_df = symbol_df.join(symbol_df[[symbol[-1]]].apply(self._lookback_window_dd[lookback]).shift(1),
                                           how='left', rsuffix=f'_{lookback}')
        symbol_df.ffill(inplace=True)
        symbol_df.dropna(inplace=True)
        asia_idx = symbol_df.index.tz_convert(self._markets.asia_tz)
        asia_uk_idx = \
            pd.Series(index=asia_idx,
                      data=False).between_time(time(hour=9, minute=0),
                                               time(hour=15, minute=0)).tz_convert(self._markets.uk_tz)
        us_idx = symbol_df.index.tz_convert(self._markets.us_tz)
        us_uk_idx = \
            pd.Series(index=us_idx,
                      data=False).between_time(time(hour=9, minute=30),
                                               time(hour=16, minute=0)).tz_convert(self._markets.uk_tz)
        eu_idx = \
            pd.Series(index=symbol_df.index,
                      data=False).between_time(time(hour=8, minute=0),
                                               time(hour=16, minute=30)).tz_convert(self._markets.uk_tz)
        symbol_df = symbol_df.assign(asia_session=False, us_session=False, europe_session=False)
        symbol_df.loc[asia_uk_idx.index, 'asia_session'] = True
        symbol_df.loc[us_uk_idx.index, 'us_session'] = True
        symbol_df.loc[eu_idx.index, 'europe_session'] = True
        symbol_df[['asia_session', 'us_session', 'europe_session']] = \
            symbol_df[['asia_session', 'us_session', 'europe_session']].astype(int)
        symbol_df.dropna(inplace=True)
        symbol_df = symbol_df.loc[~symbol_df.index.duplicated(), :]
        return symbol_df


class FeatureHARCDR(FeatureBuilderBase):

    def __init__(self):
        super().__init__('har_cdr')
        self._markets = Market()

    def builder(self, symbol: typing.Union[typing.Tuple[str], str], df: pd.DataFrame,
              df2: pd.DataFrame, F: typing.List[str]) -> pd.DataFrame:
        if isinstance(symbol, str):
            symbol = (symbol, symbol)
        list_symbol = list(dict.fromkeys(symbol).keys())
        symbol_rv_df = df[list_symbol].copy()
        symbol_cdr_df = df2[[list_symbol[-1]]].copy()
        for _, lookback in enumerate(F):
            if self._5min_buckets_lookback_window_dd[lookback] <= 288:
                offset = self._lookback_window_dd[lookback]
                symbol_rv_df = symbol_rv_df.join(symbol_rv_df[[symbol[-1]]].shift(offset), how='left',
                                                 rsuffix=f'_{lookback}')
            else:
                symbol_rv_df = \
                    symbol_rv_df.join(symbol_rv_df[[symbol[-1]]].apply(self._lookback_window_dd[lookback]).shift(1),
                                      how='left', rsuffix=f'_{lookback}')
        symbol_rv_df.ffill(inplace=True)
        symbol_rv_df.dropna(inplace=True)
        symbol_rv_df = symbol_rv_df.join(symbol_cdr_df, how='left', rsuffix='_CDR')
        return symbol_rv_df


class FeatureHARCSR(FeatureBuilderBase):

    def __init__(self):
        super().__init__('har_csr')
        self._markets = Market()

    def builder(self, symbol: typing.Union[typing.Tuple[str], str], df: pd.DataFrame,
              df2: pd.DataFrame, F: typing.List[str]) -> pd.DataFrame:
        if isinstance(symbol, str):
            symbol = (symbol, symbol)
        list_symbol = list(dict.fromkeys(symbol).keys())
        symbol_rv_df = df[list_symbol].copy()
        symbol_csr_df = df2[[list_symbol[-1]]].copy()
        for _, lookback in enumerate(F):
            if self._5min_buckets_lookback_window_dd[lookback] <= 288:
                offset = self._lookback_window_dd[lookback]
                symbol_rv_df = symbol_rv_df.join(symbol_rv_df[[symbol[-1]]].shift(offset), how='left',
                                                 rsuffix=f'_{lookback}')
                symbol_csr_df = symbol_csr_df.join(symbol_csr_df[[symbol[-1]]].shift(offset), how='left',
                                                   rsuffix=f'_{lookback}')
            else:
                symbol_rv_df = \
                    symbol_rv_df.join(symbol_rv_df[[symbol[-1]]].apply(self._lookback_window_dd[lookback]).shift(1),
                                      how='left', rsuffix=f'_{lookback}')
                symbol_csr_df = \
                    symbol_csr_df.join(symbol_csr_df[[symbol[-1]]].apply(self._lookback_window_dd[lookback]).shift(1),
                                       how='left', rsuffix=f'_{lookback}')
        symbol_rv_df.ffill(inplace=True)
        symbol_rv_df.dropna(inplace=True)
        symbol_csr_df.ffill(inplace=True)
        symbol_csr_df.dropna(inplace=True)
        symbol_rv_df = symbol_rv_df.join(symbol_csr_df, how='left', rsuffix='_CSR')
        return symbol_rv_df


class FeatureHARUniversal(FeatureBuilderBase):

    def __init__(self):
        super().__init__('har_universal')

    def builder(self, df: pd.DataFrame, F: typing.List[str]) -> pd.DataFrame:
        rv_universal_df = pd.melt(df.reset_index(), id_vars='timestamp', value_name='rv', var_name='symbol')

        def build_per_symbol(symbol_rv_df: pd.DataFrame, F: typing.List[str]=F) -> pd.DataFrame:
            symbol_rv_df = pd.pivot(symbol_rv_df, index='timestamp', values='rv', columns='symbol')
            symbol_rv_df.columns.name = None
            symbol = symbol_rv_df.columns[0]
            for _, lookback in enumerate(F):
                if self._5min_buckets_lookback_window_dd[lookback] <= 288:
                    offset = self._lookback_window_dd[lookback]
                    symbol_rv_df = symbol_rv_df.join(symbol_rv_df[[symbol]].shift(offset), how='left',
                                                     rsuffix=f'_{lookback}')
                else:
                    symbol_rv_df = \
                        symbol_rv_df.join(symbol_rv_df[[symbol]].apply(self._lookback_window_dd[lookback]).shift(1),
                                          how='left', rsuffix=f'_{lookback}')
            symbol_rv_df.ffill(inplace=True)
            symbol_rv_df.dropna(inplace=True)
            symbol_rv_df = symbol_rv_df.assign(symbol=symbol)
            symbol_rv_df.columns = symbol_rv_df.columns.str.replace(symbol, 'RV')
            return symbol_rv_df
        group_symbol = rv_universal_df.groupby(by='symbol')
        rv_universal_df = group_symbol.apply(build_per_symbol).droplevel(0)
        label_encoder_obj = LabelEncoder()
        rv_universal_df.symbol = label_encoder_obj.fit_transform(rv_universal_df.symbol)
        rv_universal_df = rv_universal_df.reset_index().set_index(['timestamp', 'symbol'])
        return rv_universal_df


class FeatureHARUniversalPuzzle(FeatureBuilderBase):

    def __init__(self):
        super().__init__('har_universal_puzzle')

    def builder(self, df: pd.DataFrame, F: typing.Union[str, typing.List[str]]) -> pd.DataFrame:
        y = df.unstack()
        y.name = 'Target'
        global X
        X = \
            pd.melt(df, ignore_index=False,
                    value_name='rv', var_name='symbol').groupby(by='symbol', group_keys=True)
        own = pd.concat([har_features_puzzle(X.get_group(group), F) for group in X.groups])
        row = pd.concat([har_features_puzzle(X.get_group(group), F, own=False) for group in X.groups], axis=1)
        X = own.reset_index().set_index('timestamp').join(row, how='left').reset_index()
        X = X.set_index(['symbol', 'timestamp']).sort_index(level=0)
        rv_universal_crossed_df = pd.concat([y, X], axis=1).dropna()
        return rv_universal_crossed_df


class ModelBuilder:

    _factory_model_type_dd = {'covariance_ar': FeatureCovariance(),
                              'risk_metrics': FeatureRiskMetricsEstimator(),
                              'har': FeatureHAR(),
                              'har_dummy_markets': FeatureHARDummy(),
                              'har_cdr': FeatureHARCDR(),
                              'har_csr': FeatureHARCSR(),
                              'har_universal': FeatureHARUniversal()}
    _factory_regression_dd = {'linear': LinearRegression()}
    factor = 100_000
    _factory_transformation_dd = {'log': {'transformation': np.log, 'inverse': np.exp},
                                  None: {'transformation': lambda x: x, 'inverse': lambda x: x},
                                  'scalar': {'transformation': lambda x: x*ModelBuilder.factor,
                                             'inverse': lambda x: x/ModelBuilder.factor}}
    models = [None, 'risk_metrics']#'covariance_ar', 'har', 'har_dummy_markets', 'har_csr', 'har_cdr', 'har_universal']
    models_rolling_metrics_dd = {model: dict([('qlike', {}), ('r2', {}), ('mse', {}),
                                              ('tstats', {}), ('pvalues', {}), ('coefficient', {})])
                                 for _, model in enumerate(models) if model}
    models_forecast_dd = {model: list() for _, model in enumerate(models) if model}
    _coins = coin_ls
    _coins = [''.join((coin, 'usdt')).upper() for _, coin in enumerate(_coins)]
    _pairs = list(product(_coins, repeat=2))
    _pairs = [(syms[0], syms[-1]) for _, syms in enumerate(_pairs)]
    L_shift_dd = {'5T': pd.to_timedelta('5T') // pd.to_timedelta('5T'),
                  '1D': pd.to_timedelta('1D') // pd.to_timedelta('5T'),
                  '1W': pd.to_timedelta('1W') // pd.to_timedelta('5T'),
                  '1M': pd.to_timedelta('30D') // pd.to_timedelta('5T')}
    start_dd = {'5T': relativedelta(minutes=5),
                '1D': relativedelta(days=1), '1W': relativedelta(weeks=1), '1M': relativedelta(months=1)}
    db_connect_coefficient = sqlite3.connect(database=os.path.abspath('../data_centre/databases/coefficients.db'))
    db_connect_mse = sqlite3.connect(database=os.path.abspath('../data_centre/databases/mse.db'))
    db_connect_qlike = sqlite3.connect(database=os.path.abspath('../data_centre/databases/qlike.db'))
    db_connect_r2 = sqlite3.connect(database=os.path.abspath('../data_centre/databases/r2.db'))
    db_connect_y = sqlite3.connect(database=os.path.abspath('../data_centre/databases/y.db'))
    db_connect_correlation = sqlite3.connect(database=os.path.abspath('../data_centre/databases/correlation.db'))

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
        for L in ['1D', '1W', '1M']:
            window = pd.to_timedelta('30D') // pd.to_timedelta('5T') if \
                L == '1M' else pd.to_timedelta(L) // pd.to_timedelta('5T')
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

    def rolling_metrics(self, symbol: typing.Union[typing.Tuple[str], str], regression_type: str='linear',
                        transformation=None, **kwargs) -> typing.Tuple[pd.Series]:
        if self._model_type not in ModelBuilder.models:
            raise ValueError('Model type not available')
        else:
            if isinstance(symbol, str):
                symbol = (symbol, symbol)
            model_obj = ModelBuilder._factory_model_type_dd[self._model_type]
            if self._model_type in ['har', 'har_dummy_markets', 'risk_metrics']:
                data = model_obj.builder(symbol=symbol, df=kwargs['df'], F=self._F)
            elif self._model_type in ['har_csr', 'har_cdr']:
                data = model_obj.builder(symbol=symbol, df=kwargs['df'], F=self._F, df2=kwargs['df2'])
            elif self._model_type in ['har_universal', 'covariance_ar']:
                data = model_obj.builder(df=kwargs['df'], F=self._F)
        pdb.set_trace()
        data = \
            pd.DataFrame(
                data=np.vectorize(ModelBuilder._factory_transformation_dd[transformation]['transformation'])
                (data.values), index=data.index, columns=data.columns)
        endog = data.pop(symbol[0]) if self._model_type != 'har_universal' else data.pop('RV')
        endog.replace(0, np.nan, inplace=True)
        endog.ffill(inplace=True)
        exog = data
        if self._model_type != 'har_universal':
            exog_rv = data.filter(regex=f'{symbol[-1]}')
            exog_rest = data.loc[:, ~data.columns.isin(exog_rv.columns.tolist())]
            exog_rv.replace(0, np.nan, inplace=True)
            exog_rv.ffill(inplace=True)
            exog_rv.fillna(axis=1, method='ffill', inplace=True)
            exog = pd.concat([exog_rv, exog_rest], axis=1)
        regression = ModelBuilder._factory_regression_dd[regression_type]
        idx_ls = set(exog.index) if self._model_type != 'har_universal' else set(exog.index.get_level_values(0))
        idx_ls = list(idx_ls)
        idx_ls.sort()
        columns_name = \
            ['const'] + exog.columns.tolist() if self._model_type != \
            'har_universal' else ['const'] + ['_'.join(('RV', F)) for _, F in enumerate(self._F)]
        coefficient = pd.DataFrame(data=np.nan, index=idx_ls, columns=columns_name)
        tstats = pd.DataFrame(data=np.nan, index=idx_ls, columns=columns_name)
        pvalues = pd.DataFrame(data=np.nan, index=idx_ls, columns=columns_name)
        if self._model_type != 'har_universal':
            coefficient_update = exog.index
        else:
            exog_tmp = \
                exog.loc[exog.index.get_level_values(0).date <=
                         (exog.index.get_level_values(0).date[-1]-ModelBuilder.start_dd[self._L]), :]
            exog_tmp = \
                exog_tmp.groupby(by=exog_tmp.index.get_level_values(1),
                group_keys=True).apply(lambda x: x.droplevel(1).shift(ModelBuilder.L_shift_dd[self._L]))
            coefficient_update = exog_tmp.index.get_level_values(1).unique()
        coefficient_update = coefficient_update[::pd.to_timedelta(self._Q)//pd.to_timedelta(self._b)]
        y = list()
        left_date = \
            (pd.to_timedelta(self._L)//pd.to_timedelta('1D')) if self._L != '1M' \
                else (pd.to_timedelta('30D')//pd.to_timedelta('1D'))
        for date in coefficient_update[left_date:-1]:
            start = pd.to_datetime(date) - ModelBuilder.start_dd[self._L]
            start = pd.to_datetime(start, utc=True)
            if self._model_type != 'har_universal':
                X_train, y_train = exog.loc[(exog.index >= start) & (exog.index < pd.to_datetime(date, utc=True))],\
                endog.loc[(endog.index >= start) & (endog.index < pd.to_datetime(date, utc=True))]
            else:
                X_train, y_train = exog.loc[(exog.index.get_level_values(0) >= start) &
                                            (exog.index.get_level_values(0) < pd.to_datetime(date, utc=True))],\
                    endog.loc[(endog.index.get_level_values(0) >= start) &
                              (endog.index.get_level_values(0) < pd.to_datetime(date, utc=True))]
            X_train.dropna(inplace=True)
            y_train.dropna(inplace=True)
            X_train.replace(np.inf, np.nan, inplace=True)
            X_train.replace(-np.inf, np.nan, inplace=True)
            y_train.replace(np.inf, np.nan, inplace=True)
            y_train.replace(-np.inf, np.nan, inplace=True)
            X_train.ffill(inplace=True)
            y_train.ffill(inplace=True)
            try:
                rres = regression.fit(X_train, y_train)
            except ValueError:
                pdb.set_trace()
            coefficient.loc[date, :] = np.concatenate((np.array([rres.intercept_]), rres.coef_))
            """Test set"""
            test_date = (pd.to_datetime(date) + ModelBuilder.start_dd['1D']).date()
            X_test, y_test = exog.loc[test_date.strftime('%Y-%m-%d'), :], endog.loc[test_date.strftime('%Y-%m-%d')]
            X_test = X_test.assign(const=1)
            X_test.replace(np.inf, np.nan, inplace=True)
            X_test.replace(-np.inf, np.nan, inplace=True)
            y_test.replace(np.inf, np.nan, inplace=True)
            y_test.replace(-np.inf, np.nan, inplace=True)
            X_test.ffill(inplace=True)
            y_test.ffill(inplace=True)
            y_hat = \
                pd.DataFrame(data=np.multiply(X_test.values,
                                              coefficient.loc[date, X_test.columns.tolist()].values),
                             columns=columns_name, index=X_test.index)
            y_hat = y_hat.sum(axis=1)
            y_hat = \
                pd.Series(data=
                          y_hat.apply(lambda x: ModelBuilder._factory_transformation_dd[transformation]['inverse'](x)),
                          index=y_hat.index)
            y_test = \
                pd.Series(data=
                          y_test.apply(lambda x: ModelBuilder._factory_transformation_dd[transformation]['inverse'](x)),
                          index=y_test.index)
            y_test[y_test<0] = 0
            y.append(pd.concat([y_test, y_hat], axis=1))
            """Tstats"""
            tstats.loc[date, :] = \
                coefficient.loc[date, :].div((np.diag(np.matmul(X_test.values.transpose(),
                                                                X_test.values))/np.sqrt(X_test.shape[0])))
            """Pvalues"""
            pvalues.loc[date, :] = 2*(1-t.cdf(tstats.loc[date, :].values, df=X_train.shape[0]-coefficient.shape[1]-1))
        y = pd.concat(y)
        ModelBuilder.models_forecast_dd[self._model_type].append(y)
        tmp = y.groupby(by=pd.Grouper(level=0, freq='1D')) if \
            self._model_type != 'har_universal' else \
            y.groupby(by=[pd.Grouper(level=-1), pd.Grouper(level=0, freq='1D')])
        mse = tmp.apply(lambda x: mean_squared_error(x.iloc[:, 0], x.iloc[:, -1]))
        mse = pd.Series(mse, name=symbol)
        qlike = tmp.apply(qlike_score)
        qlike = pd.Series(qlike, name=symbol)
        r2 = tmp.apply(lambda x: r2_score(x.iloc[:, 0], x.iloc[:, -1]))
        r2 = pd.Series(r2, name=symbol)
        coefficient.ffill(inplace=True)
        coefficient.dropna(inplace=True)
        tstats.ffill(inplace=True)
        tstats.dropna(inplace=True)
        pvalues.ffill(inplace=True)
        pvalues.dropna(inplace=True)
        coefficient = coefficient.loc[coefficient.index[::pd.to_timedelta(self._h)//pd.to_timedelta(self._b)], :]
        coefficient.columns = \
            coefficient.columns.str.replace('_'.join((symbol[-1], '')), '') \
                if self._model_type != 'har_universal' else coefficient.columns.str.replace('_'.join(('RV', '')), '')
        tstats.columns = tstats.columns.str.replace('_'.join((symbol[-1], '')), '') \
            if self._model_type != 'har_universal' else tstats.columns.str.replace('_'.join(('RV', '')), '')
        pvalues.columns = pvalues.columns.str.replace('_'.join((symbol[-1], '')), '') \
            if self._model_type != 'har_universal' else pvalues.columns.str.replace('_'.join(('RV', '')), '')
        return mse, qlike, r2, coefficient, tstats, pvalues

    @staticmethod
    def reinitialise_models_forecast_dd():
        ModelBuilder.models_forecast_dd = {model: list() for _, model in enumerate(ModelBuilder.models) if model}

    def add_metrics(self, regression_type: str='linear', transformation=None, cross: bool=False, **kwargs) -> None:
        if self._model_type != 'har_universal':
            if self._model_type in ['covariance_ar']:
                pass#coin_ls = [coin.replace('USDT', '') for _, coin in enumerate(self._coins)]
            else:
                coin_ls = set(self._coins).intersection(set(kwargs['df'].columns))
                coin_ls = list(coin_ls)
            if cross:
                coin_ls = list(product(coin_ls, repeat=2))
            r2_dd = dict([(pair, pd.DataFrame()) for _, pair in enumerate(coin_ls)])
            mse_dd = dict([(pair, pd.DataFrame()) for _, pair in enumerate(coin_ls)])
            qlike_dd = dict([(pair, pd.DataFrame()) for _, pair in enumerate(coin_ls)])
            coefficient_dd = dict([(pair, pd.DataFrame()) for _, pair in enumerate(coin_ls)])
            tstats_dd = dict([(pair, pd.DataFrame()) for _, pair in enumerate(coin_ls)])
            pvalues_dd = dict([(pair, pd.DataFrame()) for _, pair in enumerate(coin_ls)])
        else:
            r2_dd = dict()
            mse_dd = dict()
            qlike_dd = dict()
            coefficient_dd = dict()
            tstats_dd = dict()
            pvalues_dd = dict()

        def add_metrics_per_symbol(symbol: typing.Union[typing.Tuple[str], str],
                                   regression_type: str='linear', transformation=None, **kwargs) -> None:
            mse_s, qlike_s, r2_s, coefficient_df, tstats_df, pvalues_df = \
                self.rolling_metrics(symbol=symbol, regression_type=regression_type,
                                     transformation=transformation, **kwargs)
            if isinstance(symbol, tuple):
                for series_s in [mse_s, qlike_s, r2_s]:
                    series_s.name = '_'.join(series_s.name)
                symbol = '_'.join(symbol)
            mse_dd[symbol] = mse_s
            qlike_dd[symbol] = qlike_s
            r2_dd[symbol] = r2_s
            coefficient_dd[symbol] = coefficient_df
            tstats_dd[symbol] = tstats_df
            pvalues_dd[symbol] = pvalues_df

        if self._model_type != 'har_universal':
            cross_dd = {True: self._pairs, False: coin_ls}
            with concurrent.futures.ThreadPoolExecutor() as executor:
                add_metrics_per_symbol_results_dd = \
                    {symbol: executor.submit(add_metrics_per_symbol(regression_type=regression_type, **kwargs,
                                                                    transformation=transformation, symbol=symbol))
                     for symbol in cross_dd[cross]}
        else:
            mse, qlike, r2, coefficient, tstats, pvalues = \
                self.rolling_metrics(symbol=None, regression_type=regression_type, transformation=transformation,
                                     **kwargs)
            mse_dd['mse'] = mse
            qlike_dd['qlike'] = qlike
            r2_dd['r2'] = r2
            coefficient_dd['coefficient'] = coefficient
            tstats_dd['tstats'] = tstats
            pvalues_dd['pvalues'] = pvalues
        ModelBuilder.models_forecast_dd[self._model_type] = \
            [df.rename(columns={df.columns[0]: 'y', df.columns[1]: 'y_hat'}) for df in
             ModelBuilder.models_forecast_dd[self._model_type]]
        y = pd.concat(ModelBuilder.models_forecast_dd[self._model_type])
        y = y.assign(model=self._model_type)
        pdb.set_trace()
        ModelBuilder.models_forecast_dd[self._model_type] = y
        ModelBuilder.remove_redundant_key(mse_dd)
        ModelBuilder.remove_redundant_key(qlike_dd)
        ModelBuilder.remove_redundant_key(r2_dd)
        ModelBuilder.remove_redundant_key(tstats_dd)
        ModelBuilder.remove_redundant_key(pvalues_dd)
        ModelBuilder.remove_redundant_key(coefficient_dd)
        r2 = pd.DataFrame(r2_dd).mean(axis=1)
        r2.name = 'values'
        r2 = pd.DataFrame(r2)
        mse = pd.DataFrame(mse_dd).mean(axis=1)
        mse.name = 'values'
        mse = pd.DataFrame(mse)
        qlike = pd.DataFrame(qlike_dd).mean(axis=1)
        qlike.name = 'values'
        qlike = pd.DataFrame(qlike)
        r2 = \
            r2 if self._model_type != 'har_universal' else \
                r2.groupby(by=[pd.Grouper(level=-1),
                               pd.Grouper(level=0)], group_keys=True).mean().groupby(by=pd.Grouper(level=0)).mean()
        r2 = r2.assign(model=self._model_type)
        mse = mse if self._model_type != 'har_universal' else \
            mse.groupby(by=[pd.Grouper(level=-1), pd.Grouper(level=0)],
                        group_keys=True).mean().groupby(by=pd.Grouper(level=0)).mean()
        mse = mse.assign(model=self._model_type)
        qlike = qlike if self._model_type != 'har_universal' else qlike.groupby(by=[pd.Grouper(level=-1),
                pd.Grouper(level=0)], group_keys=True).mean().groupby(by=pd.Grouper(level=0)).mean()
        qlike = qlike.assign(model=self._model_type)
        ModelBuilder.models_rolling_metrics_dd[self._model_type]['mse'] = mse
        ModelBuilder.models_rolling_metrics_dd[self._model_type]['qlike'] = qlike
        ModelBuilder.models_rolling_metrics_dd[self._model_type]['r2'] = r2
        if self._model_type == 'har_universal':
            ModelBuilder.models_rolling_metrics_dd[self._model_type]['tstats'] = tstats
            ModelBuilder.models_rolling_metrics_dd[self._model_type]['pvalues'] = pvalues
            ModelBuilder.models_rolling_metrics_dd[self._model_type]['coefficient'] = coefficient
        else:
            ModelBuilder.models_rolling_metrics_dd[self._model_type]['tstats'] = \
                pd.DataFrame({symbol: tstats.mean() for symbol, tstats in tstats_dd.items()})
            ModelBuilder.models_rolling_metrics_dd[self._model_type]['pvalues'] = \
                pd.DataFrame({symbol: pvalues.mean() for symbol, pvalues in pvalues_dd.items()})
            ModelBuilder.models_rolling_metrics_dd[self._model_type]['coefficient'] = \
                pd.DataFrame({symbol: coeff.mean() for symbol, coeff in coefficient_dd.items()})


    @staticmethod
    def remove_redundant_key(dd: dict) -> None:
        dd_copy = copy.copy(dd)
        for key, df in dd_copy.items():
            if df.empty:
                dd.pop(key)


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


if __name__ == '__main__':
    data_obj = Reader(file='../data_centre/tmp/aggregate2022')
    returns = data_obj.returns_read(raw=False)
    rv = data_obj.rv_read()
    cdr = data_obj.cdr_read()
    csr = data_obj.csr_read()
    F = ['5T']
    L = '5T'
    model_builder_obj = ModelBuilder(F=F, h='30T', L='5T', Q='1D')
    agg = '1D'
    cross_name_dd = {True: 'cross', False: 'not_crossed'}
    transformation = None
    for _ in range(0, 1):#L in ['1D', '1W', '1M']:
        model_builder_obj.reinitialise_models_forecast_dd()
        for cross in [False]:
            print(f'[Computation]: Compute all tables for {(L, F, cross)}...')
            """
            Generate all tables for L, F and not cross|cross (name of table: L_(not)_cross
            """
            for _, model_type in enumerate(model_builder_obj.models):
                if model_type:
                    model_builder_obj.model_type = model_type
                    print(model_builder_obj.model_type)
                    if model_type in ['har', 'har_dummy_markets', 'har_universal']:
                        if cross & (model_type == 'har_universal'):
                            pass
                        else:
                            model_builder_obj.add_metrics(df=rv, cross=cross, agg=agg,
                                                          transformation=transformation)
                    elif model_type == 'har_cdr':
                        model_builder_obj.add_metrics(df=rv, df2=cdr, cross=cross, agg=agg,
                                                      transformation=transformation)
                    elif model_type == 'har_csr':
                        model_builder_obj.add_metrics(df=rv, df2=csr, cross=cross, agg=agg,
                                                      transformation=transformation)
                    elif model_type in ['covariance_ar', 'risk_metrics']:
                        model_builder_obj.add_metrics(df=data_obj.returns_read(raw=False),
                                                      cross=cross, agg=agg, transformation=transformation)
            mse = [model_builder_obj.models_rolling_metrics_dd[model]['mse'] for model in
                   model_builder_obj.models_rolling_metrics_dd.keys() if model
                   and isinstance(model_builder_obj.models_rolling_metrics_dd[model]['mse'], pd.DataFrame)]
            mse = pd.concat(mse)
            qlike = [model_builder_obj.models_rolling_metrics_dd[model]['qlike'] for model in
                     model_builder_obj.models_rolling_metrics_dd.keys() if model
                     and isinstance(model_builder_obj.models_rolling_metrics_dd[model]['qlike'], pd.DataFrame)]
            qlike = pd.concat(qlike)
            r2 = [model_builder_obj.models_rolling_metrics_dd[model]['r2'] for model in
                  model_builder_obj.models_rolling_metrics_dd.keys() if model
                  and isinstance(model_builder_obj.models_rolling_metrics_dd[model]['r2'], pd.DataFrame)]
            r2 = pd.concat(r2)
            model_axis_dd = {model: False if model == 'har_universal' else True
                             for _, model in enumerate(model_builder_obj.models)}
            coefficient = \
                [pd.DataFrame(
                    model_builder_obj.models_rolling_metrics_dd[model]['coefficient'].mean(axis=model_axis_dd[model]),
                    columns=[model])
                 for model in model_builder_obj.models_rolling_metrics_dd.keys() if model
                 and isinstance(model_builder_obj.models_rolling_metrics_dd[model]['coefficient'], pd.DataFrame)]
            coefficient = pd.concat(coefficient, axis=1)
            coefficient = coefficient.loc[~coefficient.index.str.contains('USDT'), :]
            model_specific_features = list(set(coefficient.index).difference((set(['const']+F))))
            model_specific_features.sort()
            coefficient = coefficient.T[['const']+F+model_specific_features].T
            coefficient.index.name = 'params'
            coefficient = pd.melt(coefficient.reset_index(), value_name='value', var_name='model', id_vars='params')
            coefficient.dropna(inplace=True)
            # model_builder_obj.models_forecast_dd['har_universal'] = \
            #     model_builder_obj.models_forecast_dd['har_universal'].droplevel(axis=0, level=1)
            y = pd.concat(model_builder_obj.models_forecast_dd)
            y = y.droplevel(0)
            pdb.set_trace()
            """
            Table insertion
            """
            r2.to_sql(con=model_builder_obj.db_connect_r2, name=f'{"_".join(("r2", L, cross_name_dd[cross]))}',
                      if_exists='replace')
            mse.to_sql(con=model_builder_obj.db_connect_mse, name=f'{"_".join(("mse", L, cross_name_dd[cross]))}',
                      if_exists='replace')
            qlike.to_sql(con=model_builder_obj.db_connect_qlike, name=f'{"_".join(("qlike", L, cross_name_dd[cross]))}',
                      if_exists='replace')
            coefficient.to_sql(con=model_builder_obj.db_connect_coefficient,
                               name=f'{"_".join(("coefficient",L, cross_name_dd[cross]))}', if_exists='replace')
            y.to_sql(con=model_builder_obj.db_connect_y,
                     name=f'{"_".join(("y",L, cross_name_dd[cross]))}', if_exists='replace')
            print(f'[Insertion]: All tables for {(L, F, cross)} have been inserted into the database.')
    """
    Close databases
    """
    model_builder_obj.db_connect_r2.close()
    model_builder_obj.db_connect_mse.close()
    model_builder_obj.db_connect_qlike.close()
    model_builder_obj.db_connect_coefficient.close()
    model_builder_obj.db_connect_y.close()
    # reader_obj = Reader(file=os.path.abspath('../data_centre/tmp/aggregate2022'))
    # returns = reader_obj.returns_read(raw=False)
    # correlation_matrix = returns.rolling(window=4).corr().dropna()
    # correlation = correlation_matrix.droplevel(axis=0, level=1).mean(axis=1)
    # correlation = correlation.groupby(by=correlation.index).mean()
    # correlation.replace(np.inf, np.nan, inplace=True)
    # correlation.replace(-np.inf, np.nan, inplace=True)
    # correlation.ffill(inplace=True)
    # correlation.name = 'CORR'
    # correlation = pd.DataFrame(correlation)
    # correlation['CORR_5M'] = correlation.shift()
    # correlation.dropna(inplace=True)
    pdb.set_trace()

