import os.path
import pdb
import typing
import pandas as pd
from pytz import timezone
from dataclasses import dataclass
from datetime import time
import numpy as np
from data_centre.data import Reader


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
                                ('1W', '7D'), ('1M', '30D'), ('6M', '180D')])

    _5min_buckets_lookback_window_dd = {lookback: pd.to_timedelta(lookback) // pd.to_timedelta('5T') for
                                        lookback in _lookback_window_dd.keys() if lookback not in ['1M', '6M']}
    #Add manually as pd.to_timedelta does not take '1M' as argument
    _5min_buckets_lookback_window_dd['1M'] = pd.to_timedelta('30D') // pd.to_timedelta('5T')
    _5min_buckets_lookback_window_dd['6M'] = pd.to_timedelta('180D') // pd.to_timedelta('5T')

    def __init__(self, name):
        self._name = name

    def builder(self, symbol: typing.Union[typing.Tuple[str], str], df: pd.DataFrame, F: typing.List[str],
                training_scheme: str):
        """To be overwritten by each child class"""
        pass


class FeatureAR(FeatureBuilderBase):

    def __init__(self):
        super().__init__('ar')

    def builder(self, df: pd.DataFrame, symbol: typing.Union[typing.Tuple[str], str],
                F: typing.List[str] = None) -> pd.DataFrame:
        if isinstance(symbol, str):
            symbol = (symbol, symbol)
        list_symbol = list(dict.fromkeys(symbol).keys())
        symbol_df = df[list_symbol].copy()
        tmp = symbol_df.copy()
        tmp = tmp.shift(FeatureAR._lookback_window_dd[F[0]])
        symbol_df = symbol_df.join(tmp, how='left', rsuffix=f'_{F[0]}')
        return symbol_df.dropna()


class FeatureRiskMetricsEstimator(FeatureBuilderBase):

    factor = .94 #lambda in formula

    def __init__(self):
        super().__init__('risk_metrics')
        FeatureRiskMetricsEstimator.data_obj = Reader()

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
                    symbol_df.join(symbol_df[list_symbol].resample(offset).mean().shift(),
                                   how='left', rsuffix=f'_{lookback}')
        symbol_df.ffill(inplace=True)
        symbol_df.dropna(inplace=True)
        return symbol_df


class FeatureHAREq(FeatureBuilderBase):

    def __init__(self):
        super().__init__('har_eq')
        self._markets = Market()

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
                    symbol_df.join(symbol_df[list_symbol].resample(offset).mean().shift(),
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
        symbol_df = symbol_df.assign(ASIAN_SESSION=False, US_SESSION=False, EUROPE_SESSION=False)
        symbol_df.loc[asia_uk_idx.index, 'ASIAN_SESSION'] = True
        symbol_df.loc[us_uk_idx.index, 'US_SESSION'] = True
        symbol_df.loc[eu_idx.index, 'EUROPE_SESSION'] = True
        symbol_df[['ASIAN_SESSION', 'US_SESSION', 'EUROPE_SESSION']] = \
            symbol_df[['ASIAN_SESSION', 'US_SESSION', 'EUROPE_SESSION']].astype(int)
        symbol_df.dropna(inplace=True)
        symbol_df = symbol_df.loc[~symbol_df.index.duplicated(), :]
        if odds:
            tmp_odds = symbol_df[['ASIAN_SESSION', 'US_SESSION', 'EUROPE_SESSION']].groupby(
                by=symbol_df.index.date).apply(lambda x: FeatureHAREq.binary_to_odds(x))
            tmp_odds = tmp_odds.droplevel(0, 0)
            symbol_df[['ASIAN_SESSION', 'US_SESSION', 'EUROPE_SESSION']] = \
                tmp_odds[['ASIAN_SESSION', 'US_SESSION', 'EUROPE_SESSION']]
        return symbol_df


class FeatureUniversal(FeatureBuilderBase):

    models_dd = {'ar': FeatureAR(),
                 'har': FeatureHAR(), 'har_eq': FeatureHAREq(),
                 'risk_metrics': FeatureRiskMetricsEstimator()}

    def __init__(self):
        super().__init__('har_universal')

    def builder(self, df: pd.DataFrame, model: str, F: typing.List[str],
                drop: bool = False, one_hot_encoding: bool = False, full: bool = True) -> pd.DataFrame:
        model_obj = FeatureUniversal.models_dd[model]
        tmp = df.copy().replace(0, np.nan).ffill()
        own = pd.DataFrame(tmp.unstack(), columns=['RV']).reset_index().set_index('timestamp')
        if full:
            universe = model_obj.builder(symbol=tmp.columns, df=tmp, F=F)
            universe = own.groupby(by='level_0').apply(lambda x, universe: x.join(universe, how='inner'), universe)
            universe = universe.droplevel(axis=0, level=0)
        else:
            universe = {symbol: model_obj.builder(symbol=symbol, df=tmp, F=F) for symbol in df.columns}
            for symbol, df_symbol in universe.items():
                df_symbol.columns = df_symbol.columns.str.replace(symbol, 'RV')
                df_symbol['level_0'] = symbol
            universe = pd.concat(universe.values())
        universe = universe.rename(columns={'level_0': 'symbol'})
        return universe
