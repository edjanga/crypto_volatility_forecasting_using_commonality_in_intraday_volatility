import pdb
import typing
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pytz import timezone
from dataclasses import dataclass
from datetime import time
import numpy as np
from data_centre.data import Reader


"""Functions and variables used to facilitate computation within classes."""
L_shift_dd = dict([('5T', pd.to_timedelta('5T')//pd.to_timedelta('5T')),
                   ('30T', pd.to_timedelta('30T') // pd.to_timedelta('5T')),
                   ('1H', pd.to_timedelta('1H')//pd.to_timedelta('5T')),
                   ('6H', pd.to_timedelta('6H')//pd.to_timedelta('5T')),
                   ('12H', pd.to_timedelta('12H')//pd.to_timedelta('5T')),
                   ('1D', pd.to_timedelta('1D')//pd.to_timedelta('5T')),
                   ('1W', lambda x:x.resample('1W').mean()), ('1M', lambda x:x.resample('30D').mean())])


def har_features_puzzle(df: pd.DataFrame, F: typing.Union[str, typing.List[str]], own=True) -> pd.DataFrame:
    symbol = df.symbol.unique()[0]
    for _, lookback in enumerate(F):
        tmp = df['RV']
        tmp.name = '_'.join(('RV', lookback)) if own else '_'.join((symbol, lookback))
        if pd.to_timedelta(lookback)//pd.to_timedelta('5T') <= 288:
            offset = pd.to_timedelta(lookback)//pd.to_timedelta('5T')
            df = df.join(tmp.shift(offset), how='left', rsuffix=f'_{lookback}')
        else:
            df = df.join(L_shift_dd[lookback](tmp).shift(1), how='left', rsuffix=f'_{lookback}')
    df.ffill(inplace=True)
    df.drop('RV', axis=1, inplace=True)
    df.dropna(inplace=True)
    df = df.reset_index().set_index(['symbol', 'timestamp'])
    if not own:
        df = df.droplevel(axis=0, level=0)
    return df


def rv_1w_correction(df: pd.DataFrame, L: str='1W') -> pd.DataFrame:
    feature_name = df.filter(regex=f'_{L}').columns[0]
    df.loc[:, feature_name] = df.loc[:, feature_name].value_counts().sort_values(ascending=False).index[0]
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

    """Lookback windows smaller than 1D are dynamic while the rest is static"""
    _lookback_window_dd = dict([('5T', pd.to_timedelta('5T')//pd.to_timedelta('5T')),
                                ('30T', pd.to_timedelta('30T') // pd.to_timedelta('5T')),
                                ('1H', pd.to_timedelta('1H')//pd.to_timedelta('5T')),
                                ('6H', pd.to_timedelta('6H')//pd.to_timedelta('5T')),
                                ('12H', pd.to_timedelta('12H')//pd.to_timedelta('5T')),
                                ('1D', pd.to_timedelta('1D')//pd.to_timedelta('5T')),
                                ('1W', lambda x:x.resample('1W').mean()), ('1M', lambda x:x.resample('30D').mean())])


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


class FeatureAR(FeatureBuilderBase):

    def __init__(self):
        super().__init__('ar')

    def builder(self, df: pd.DataFrame, symbol: str, F: typing.List[str] = None) -> pd.DataFrame:
        if isinstance(symbol, str):
            symbol = (symbol, symbol)
        list_symbol = list(dict.fromkeys(symbol).keys())
        symbol_df = df[list_symbol].copy()
        symbol_df[f'{list_symbol[-1]}_30T'] = symbol_df[f'{list_symbol[-1]}'].shift(FeatureAR._lookback_window_dd[F[0]])
        return symbol_df.dropna()


class FeatureRiskMetricsEstimator(FeatureBuilderBase):

    data_obj = Reader(file='./data_centre/tmp/aggregate2022')
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
        symbol_returns_df = df[list_symbol].copy().rename(columns={symbol[-1]: '_'.join((symbol[-1], 'RET', F[0]))})**2
        symbol_returns_df = symbol_returns_df.shift(FeatureRiskMetricsEstimator._lookback_window_dd[F[0]])
        symbol_returns_df = symbol_returns_df.sub(symbol_returns_df.mean()[0])
        symbol_rv_df = FeatureRiskMetricsEstimator.data_obj.rv_read(raw=False, symbol=list_symbol)
        symbol_rv_df = symbol_rv_df[list_symbol].copy()
        symbol_rv_df[f'{"_".join((symbol[-1], F[0]))}'] = \
            symbol_rv_df[[symbol[-1]]].shift(FeatureRiskMetricsEstimator._lookback_window_dd[F[0]])
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

    @staticmethod
    def binary_to_odds(df: pd.DataFrame) -> pd.DataFrame:
        def binary_to_odds_per_series(series: pd.Series) -> pd.Series:
            count = series.value_counts()
            tmp = pd.Series(data=count.values, index=count.index[::-1], name=count.name)
            odds = count.divide(tmp).to_dict()
            series = pd.Series(data=np.where(series==1, odds[1], odds[0]), index=series.index, name=series.name)
            return series
        df = df.apply(lambda x: binary_to_odds_per_series(x))
        return df

    def builder(self, symbol: typing.Union[typing.Tuple[str], str], df: pd.DataFrame, F: typing.List[str], odds=True)\
            -> pd.DataFrame:
        """
            Odds instead of binary to allow for log transformation.
        """
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
        if odds:
            symbol_df[['asia_session', 'us_session', 'europe_session']] = \
                symbol_df[['asia_session', 'us_session', 'europe_session']].groupby(
                by=symbol_df.index.date).apply(lambda x: FeatureHARDummy.binary_to_odds(x))
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
        for _, lookback in enumerate([F[-1]]):
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

    label_encoder_obj = LabelEncoder()
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
        rv_universal_df.symbol = FeatureHARUniversal.label_encoder_obj.fit_transform(rv_universal_df.symbol)
        rv_universal_df = rv_universal_df.reset_index().set_index(['timestamp', 'symbol'])
        return rv_universal_df


class FeatureHARUniversalPuzzle(FeatureBuilderBase):

    label_encoder_obj = LabelEncoder()

    def __init__(self):
        super().__init__('har_universal_puzzle')

    def builder(self, df: pd.DataFrame, F: typing.Union[str, typing.List[str]]) -> pd.DataFrame:
        y = df.unstack()
        y.name = 'RV'
        global X
        X = \
            pd.melt(df, ignore_index=False,
                    value_name='RV', var_name='symbol').groupby(by='symbol', group_keys=True)
        own = pd.concat([har_features_puzzle(X.get_group(group), F) for group in X.groups])
        row = pd.concat([har_features_puzzle(X.get_group(group), F, own=False) for group in X.groups], axis=1)
        X = own.reset_index().set_index('timestamp').join(row, how='left').reset_index()
        X = X.set_index(['symbol', 'timestamp']).sort_index(level=0)
        rv_universal_crossed_df = pd.concat([y, X], axis=1).dropna()
        rv_universal_crossed_df = rv_universal_crossed_df.swaplevel(i='timestamp', j='symbol')
        rv_universal_crossed_df.index = \
            rv_universal_crossed_df.index.set_levels(
                FeatureHARUniversalPuzzle.label_encoder_obj.fit_transform(
                    rv_universal_crossed_df.index.levels[1]), level=1)
        return rv_universal_crossed_df
