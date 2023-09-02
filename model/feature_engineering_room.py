import os.path
import pdb
import typing
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from pytz import timezone
from dataclasses import dataclass
from datetime import time
import numpy as np
from data_centre.data import Reader


"""Functions and variables used to facilitate computation within classes."""
# L_shift_dd = dict([('5T', pd.to_timedelta('5T')//pd.to_timedelta('5T')),
#                    ('30T', pd.to_timedelta('30T') // pd.to_timedelta('5T')),
#                    ('1H', pd.to_timedelta('1H')//pd.to_timedelta('5T')),
#                    ('6H', pd.to_timedelta('6H')//pd.to_timedelta('5T')),
#                    ('12H', pd.to_timedelta('12H')//pd.to_timedelta('5T')),
#                    ('1D', pd.to_timedelta('1D')//pd.to_timedelta('5T')),
#                    ('1W', lambda x:x.resample('7D').mean()), ('SAM', lambda x:x.resample('30D').mean())])


# def universal(df: pd.DataFrame, F: typing.Union[str, typing.List[str]], own=True) -> pd.DataFrame:
#     symbol = df.symbol.unique()[0]
#     for _, lookback in enumerate(F):
#         tmp = df['RV']
#         tmp.name = '_'.join(('RV', lookback)) if own else '_'.join((symbol, lookback))
#         if pd.to_timedelta(lookback)//pd.to_timedelta('5T') <= 288:
#             offset = pd.to_timedelta(lookback)//pd.to_timedelta('5T')
#             df = df.join(tmp.shift(offset), how='left', rsuffix=f'_{lookback}')
#         else:
#             df = df.join(L_shift_dd[lookback](tmp).shift(1), how='left', rsuffix=f'_{lookback}')
#     df.ffill(inplace=True)
#     df.drop('RV', axis=1, inplace=True)
#     df.dropna(inplace=True)
#     df = df.reset_index().set_index(['symbol', 'timestamp'])
#     if not own:
#         df = df.droplevel(axis=0, level=0)
#     return df


# def rv_1w_correction(df: pd.DataFrame, L: str='1W') -> pd.DataFrame:
#     feature_name = df.filter(regex=f'_{L}').columns[0]
#     df.loc[:, feature_name] = df.loc[:, feature_name].value_counts().sort_values(ascending=False).index[0]
#     return df


@dataclass
class Market:
    """
        Dataclass that describes markets with a timezone object.
    """
    asia_tz = timezone('Asia/Tokyo')
    us_tz = timezone('US/Eastern')
    uk_tz = timezone('UTC')


class FeatureBuilderBase:

    """Lookback windows smaller than 1D are dynamic while the rest is static"""
    _lookback_window_dd = dict([('5T', pd.to_timedelta('5T')//pd.to_timedelta('5T')),
                                ('30T', pd.to_timedelta('30T') // pd.to_timedelta('5T')),
                                ('1H', pd.to_timedelta('1H')//pd.to_timedelta('5T')),
                                ('6H', pd.to_timedelta('6H')//pd.to_timedelta('5T')),
                                ('12H', pd.to_timedelta('12H')//pd.to_timedelta('5T')),
                                ('1D', pd.to_timedelta('1D')//pd.to_timedelta('5T')),
                                ('1W', '7D'),
                                ('1M', '30D')])


    _5min_buckets_lookback_window_dd = \
        {lookback: pd.to_timedelta(lookback) // pd.to_timedelta('5T') for
         lookback in _lookback_window_dd.keys() if lookback != '1M'}

    #Add manually as pd.to_timedelta does not take '1M' as argument
    _5min_buckets_lookback_window_dd['1M'] = pd.to_timedelta('30D') // pd.to_timedelta('5T')

    def __init__(self, name, indiv: bool=True):
        self._name = name
        self._indv = indiv

    @property
    def name(self):
        return self._name

    def builder(self, symbol: typing.Union[typing.Tuple[str], str], df: pd.DataFrame, F: typing.List[str],
                training_scheme: str):
        """To be overwritten by each child class"""
        pass


class FeatureAR(FeatureBuilderBase):

    def __init__(self, indiv: bool=True):
        super().__init__('ar', indiv)

    def builder(self, df: pd.DataFrame, symbol: typing.Union[typing.Tuple[str], str],
                F: typing.List[str] = None) -> pd.DataFrame:
        if isinstance(symbol, str):
            symbol = (symbol, symbol)
        list_symbol = list(dict.fromkeys(symbol).keys())
        try:
            symbol_df = df[list_symbol].copy()
        except KeyError:
            pdb.set_trace()
        if len(symbol) > 1:
            tmp = symbol_df.copy()
            for f in F:
                tmp = tmp.rename(columns={f'{sym}': f'{sym}_{f}' for sym in tmp.columns})
                tmp = tmp.shift(FeatureAR._lookback_window_dd[f])
                symbol_df = symbol_df.join(tmp, how='left')
        return symbol_df.dropna()


class FeatureRiskMetricsEstimator(FeatureBuilderBase):

    data_obj = Reader(directory=os.path.abspath('./data_centre/tmp'))
    factor = .94 #lambda in formula

    def __init__(self):
        super().__init__('risk_metrics')

    def builder(self, F: typing.Union[typing.List[str], str],
                df: pd.DataFrame, symbol: typing.Union[str, typing.List[str]]) -> pd.DataFrame:
        if isinstance(F, str):
            F = [F]
        if isinstance(symbol, str):
            symbol = (symbol, symbol)
        list_symbol = list(dict.fromkeys(symbol).keys())
        symbol_df = FeatureRiskMetricsEstimator.data_obj.returns_read(raw=False, symbol=list_symbol)
        symbol_df = symbol_df.rename(columns={sym: '_'.join((sym, 'RET', F[0])) for sym in symbol_df.columns})
        symbol_df = symbol_df.sub(symbol_df.mean())**2
        symbol_df = symbol_df.shift(FeatureRiskMetricsEstimator._lookback_window_dd[F[0]]).dropna()
        symbol_df = symbol_df.join(FeatureRiskMetricsEstimator.data_obj.rv_read(symbol=list_symbol),
                                   how='left')
        symbol_df.columns.name = None
        return symbol_df


class FeatureHAR(FeatureBuilderBase):

    def __init__(self):
        super().__init__('har')

    def builder(self, symbol: typing.Union[typing.Tuple[str], str], df: pd.DataFrame,
                F: typing.List[str]) -> pd.DataFrame:
        if isinstance(symbol, str):
            symbol = (symbol, symbol)
        list_symbol = list(dict.fromkeys(symbol).keys())
        symbol_df = df[list_symbol].copy()
        for _, lookback in enumerate(F):
            offset = self._lookback_window_dd[lookback]
            if self._5min_buckets_lookback_window_dd[lookback] <= 288:
                symbol_df = symbol_df.join(symbol_df[list_symbol].rolling(offset+1, closed='both').mean().shift(),
                                           how='left', rsuffix=f'_{lookback}')
            else:
                symbol_df = \
                    symbol_df.join(symbol_df[list_symbol].resample(offset).mean(),
                                   how='left', rsuffix=f'_{lookback}')
        symbol_df.ffill(inplace=True)
        symbol_df.dropna(inplace=True)
        return symbol_df


class FeatureHARMkt(FeatureBuilderBase):

    def __init__(self):
        super().__init__('har_mkt')
        self._markets = Market()

    @property
    def markets(self):
        return self._markets

    @staticmethod
    def binary_to_odds(df: pd.DataFrame) -> pd.DataFrame:
        def binary_to_odds_per_series(series: pd.Series) -> pd.Series:
            count = series.value_counts()
            tmp = pd.Series(data=count.values, index=count.index[::-1], name=count.name)
            odds = count.divide(tmp).to_dict()
            series = pd.Series(data=np.where(series == 1,
                                             odds[1], odds[0]), index=series.index, name=series.name)
            return series
        df = df.apply(lambda x: binary_to_odds_per_series(x))
        return df

    def builder(self, symbol: typing.Union[typing.Tuple[str], str], df: pd.DataFrame,
                F: typing.List[str], odds=False)\
            -> pd.DataFrame:
        """
            Odds instead of binary to allow for log transformation.
        """
        if isinstance(symbol, str):
            symbol = (symbol, symbol)
        list_symbol = list(dict.fromkeys(symbol).keys())
        symbol_df = df[list_symbol].copy()
        for _, lookback in enumerate(F):
            offset = self._lookback_window_dd[lookback]
            if self._5min_buckets_lookback_window_dd[lookback] <= 288:
                symbol_df = symbol_df.join(symbol_df[list_symbol].rolling(offset + 1, closed='both').mean().shift(),
                                           how='left', rsuffix=f'_{lookback}')
            else:
                symbol_df = \
                    symbol_df.join(symbol_df[list_symbol].resample(offset).mean(),
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
        if odds:
            symbol_df[['asia_session', 'us_session', 'europe_session']] = \
                symbol_df[['asia_session', 'us_session', 'europe_session']].groupby(
                by=symbol_df.index.date).apply(lambda x: FeatureHARMkt.binary_to_odds(x))
        return symbol_df


# class FeatureHARJ(FeatureBuilderBase):
#
#     def __init__(self):
#         super().__init__('har_j')
#         self._markets = Market()
#
#     def builder(self, symbol: typing.Union[typing.Tuple[str], str], df: pd.DataFrame,
#               df2: pd.DataFrame, F: typing.List[str]) -> pd.DataFrame:
#         if isinstance(symbol, str):
#             symbol = (symbol, symbol)
#         list_symbol = list(dict.fromkeys(symbol).keys())
#         symbol_rv_df = df[list_symbol].copy()
#         symbol_cdr_df = df2[[list_symbol[-1]]].copy()
#         for _, lookback in enumerate(F):
#             if self._5min_buckets_lookback_window_dd[lookback] <= 288:
#                 offset = self._lookback_window_dd[lookback]
#                 symbol_rv_df = symbol_rv_df.join(symbol_rv_df[[symbol[-1]]].shift(offset), how='left',
#                                                  rsuffix=f'_{lookback}')
#                 symbol_cdr_df = symbol_cdr_df.join(symbol_cdr_df[[symbol[-1]]].shift(offset), how='left',
#                                                  rsuffix=f'_{lookback}')
#             else:
#                 symbol_rv_df = \
#                     symbol_rv_df.join(symbol_rv_df[[symbol[-1]]].apply(self._lookback_window_dd[lookback]).shift(1),
#                                       how='left', rsuffix=f'_{lookback}')
#                 symbol_cdr_df = \
#                     symbol_cdr_df.join(symbol_cdr_df[[symbol[-1]]].apply(self._lookback_window_dd[lookback]).shift(1),
#                                       how='left', rsuffix=f'_{lookback}')
#         symbol_rv_df.ffill(inplace=True)
#         symbol_rv_df.dropna(inplace=True)
#         symbol_rv_df = symbol_rv_df.join(symbol_cdr_df, how='left', rsuffix='_CDR')
#         return symbol_rv_df
#
#     @staticmethod
#     def jump(df: pd.DataFrame) -> pd.DataFrame:
#         """
#             Definition of jump according to Rahimikia & Poon
#         """


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
            offset = self._lookback_window_dd[lookback]
            if self._5min_buckets_lookback_window_dd[lookback] <= 288:
                symbol_rv_df = symbol_rv_df.join(symbol_rv_df[list_symbol].shift(offset), how='left',
                                                 rsuffix=f'_{lookback}')
                symbol_cdr_df = symbol_cdr_df.join(symbol_cdr_df[list_symbol].shift(offset), how='left',
                                                 rsuffix=f'_{lookback}')
            else:
                tmp = symbol_rv_df[list_symbol].resample('1D').mean()
                symbol_rv_df = \
                    symbol_rv_df.join(tmp.rolling(self._lookback_window_dd[lookback]).mean(),
                                      how='left', rsuffix=f'_{lookback}')
                tmp = symbol_cdr_df[list_symbol].resample('1D').mean()
                symbol_cdr_df = \
                    symbol_cdr_df.join(tmp.rolling(self._lookback_window_dd[lookback]).mean(),
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
        for _, lookback in enumerate([F[-1]]):
            if self._5min_buckets_lookback_window_dd[lookback] <= 288:
                offset = self._lookback_window_dd[lookback]
                symbol_rv_df = symbol_rv_df.join(symbol_rv_df[list_symbol].shift(offset), how='left',
                                                 rsuffix=f'_{lookback}')
                symbol_csr_df = symbol_csr_df.join(symbol_csr_df[list_symbol].shift(offset), how='left',
                                                   rsuffix=f'_{lookback}')
            else:
                tmp = symbol_rv_df[list_symbol].resample('1D').mean()
                symbol_rv_df = \
                    symbol_rv_df.join(tmp.rolling(self._lookback_window_dd[lookback]).mean(),
                                      how='left', rsuffix=f'_{lookback}')
                tmp = symbol_csr_df[list_symbol].resample('1D').mean()
                symbol_csr_df = \
                    symbol_csr_df.join(tmp.rolling(self._lookback_window_dd[lookback]).mean(),
                                       how='left', rsuffix=f'_{lookback}')
        symbol_rv_df.ffill(inplace=True)
        symbol_rv_df.dropna(inplace=True)
        symbol_csr_df.ffill(inplace=True)
        symbol_csr_df.dropna(inplace=True)
        symbol_rv_df = symbol_rv_df.join(symbol_csr_df, how='left', rsuffix='_CSR')
        return symbol_rv_df


class FeatureUniversal(FeatureBuilderBase):

    models_dd = {'risk_metrics': FeatureRiskMetricsEstimator(), 'ar': FeatureAR(),
                 'har': FeatureHAR(), 'har_mkt': FeatureHARMkt()}

    def __init__(self):
        super().__init__('har_universal')

    def builder(self, df: pd.DataFrame, model: str, F: typing.List[str],
                drop: bool = False) -> pd.DataFrame:
        model_obj = FeatureUniversal.models_dd[model]
        tmp = df.copy().replace(0, np.nan).ffill()
        own = pd.DataFrame(tmp.unstack(), columns=['RV']).reset_index().set_index('timestamp')
        own = pd.concat([own, pd.get_dummies(own.level_0)], axis=1)
        own = own.rename(columns={col: '_'.join(('is', col)) for col in own.columns[2:]})
        if drop:
            own.drop(own.columns[np.random.randint(1, own.shape[1])], axis=1, inplace=True)
        universe = pd.concat([model_obj.builder(symbol=tmp.columns, df=tmp, F=F)] * tmp.shape[1])
        universe = pd.concat([own.reset_index(), universe.reset_index()], axis=1)
        universe.dropna(inplace=True)
        universe = universe.loc[:, ~universe.columns.duplicated('first')]
        universe = universe.set_index('timestamp')
        universe.drop('level_0', axis=1, inplace=True)
        return universe
