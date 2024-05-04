import datetime
from dateutil.relativedelta import relativedelta
import pdb
import typing
import numpy as np
import pandas as pd
from pytz import timezone
from dataclasses import dataclass
from datetime import time
from data_centre.data import Reader
from scipy.optimize import minimize


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

    def builder(self, symbol: typing.Union[typing.Tuple[str], str], df: pd.DataFrame, training_scheme: str,
                **kwargs):
        """To be overwritten by each child class"""
        pass


class FeatureRiskMetrics(FeatureBuilderBase):

    _alpha = .94

    def __init__(self):
        super().__init__('risk_metrics')

    @staticmethod
    def predict(y: pd.Series, date: datetime) -> pd.Series:
        yt_hat = y.copy()
        yt_hat[yt_hat.index.date == date] = np.nan
        yt_hat[yt_hat.index.date == date] = \
            ((yt_hat-yt_hat.mean())**2).ewm(alpha=FeatureRiskMetrics._alpha, min_periods=1, adjust=False).mean()[-288:]
        return yt_hat[yt_hat.index.date == date]


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

    _markets = Market()

    def __init__(self):
        super().__init__('har_eq')

    @staticmethod
    def eq_mkt_builder(**kwargs) -> pd.DataFrame:
        if kwargs is None:
            return pd.DataFrame()
        elif kwargs.get('trading_session') == 1:
            eq_mkt_df = pd.DataFrame(index=kwargs['df'].index)
            asia_idx = eq_mkt_df.index.tz_convert(FeatureHAREq._markets.asia_tz)
            asia_uk_idx = \
                pd.Series(index=asia_idx,
                          data=False).between_time(time(hour=9, minute=0),
                                                   time(hour=15, minute=0)).tz_convert(FeatureHAREq._markets.uk_tz)
            us_idx = eq_mkt_df.index.tz_convert(FeatureHAREq._markets.us_tz)
            us_uk_idx = \
                pd.Series(index=us_idx,
                          data=False).between_time(time(hour=9, minute=30),
                                                   time(hour=16, minute=0)).tz_convert(FeatureHAREq._markets.uk_tz)
            eu_idx = \
                pd.Series(index=eq_mkt_df.index,
                          data=False).between_time(time(hour=8, minute=0),
                                                   time(hour=16, minute=30)).tz_convert(FeatureHAREq._markets.uk_tz)
            eq_mkt_df = eq_mkt_df.assign(ASIAN_SESSION=False, US_SESSION=False, EUROPE_SESSION=False)
            eq_mkt_df.loc[asia_uk_idx.index, 'ASIAN_SESSION'] = True
            eq_mkt_df.loc[us_uk_idx.index, 'US_SESSION'] = True
            eq_mkt_df.loc[eu_idx.index, 'EUROPE_SESSION'] = True
            eq_mkt_df[['ASIAN_SESSION', 'US_SESSION', 'EUROPE_SESSION']] = \
                eq_mkt_df[['ASIAN_SESSION', 'US_SESSION', 'EUROPE_SESSION']].astype(int)
        else:
            top_of_book = {True: '_0', False: ''}
            eq_mkt_df = kwargs['vixm'].filter(regex=f'VIXM{top_of_book[kwargs["top_book"]]}')
            for _, lookback in enumerate(kwargs['F']):
                offset = FeatureHAREq._lookback_window_dd[lookback]
                if FeatureHAREq._5min_buckets_lookback_window_dd[lookback] <= 288:
                    eq_mkt_df = \
                        eq_mkt_df.join(eq_mkt_df.filter(regex=f'VIXM_[0-9]$').rolling(offset + 1,
                                                                                      closed='both').mean().shift(),
                                       how='left', rsuffix=f'_{lookback}')
                else:
                    eq_mkt_df = eq_mkt_df.join(eq_mkt_df.filter(regex=f'VIXM_[0-9$]$').resample(offset).mean().shift(),
                                               how='left', rsuffix=f'_{lookback}')
        return eq_mkt_df.interpolate(method='linear', limit_direction='forward').dropna()

    def builder(self, df: pd.DataFrame, symbol: typing.Union[typing.Tuple[str], str], F: typing.List[str],
                **kwargs) -> pd.DataFrame:
        if isinstance(symbol, str):
            symbol = (symbol, symbol)
        list_symbol = list(dict.fromkeys(symbol).keys())
        symbol_df = df[list_symbol].copy()
        for _, lookback in enumerate(F):
            offset = FeatureHAREq._lookback_window_dd[lookback]
            if FeatureHAREq._5min_buckets_lookback_window_dd[lookback] <= 288:
                symbol_df = symbol_df.join(symbol_df[list_symbol].rolling(offset + 1, closed='both').mean().shift(),
                                           how='left', rsuffix=f'_{lookback}')
            else:
                symbol_df = symbol_df.join(symbol_df[list_symbol].resample(offset).mean().shift(),
                                           how='left', rsuffix=f'_{lookback}')
        symbol_df = symbol_df.ffill()
        symbol_df = symbol_df.dropna()
        if 'vixm' in kwargs.keys():
            if kwargs.get('trading_session') == 1:
                kwargs.update({'df': symbol_df})
            else:
                kwargs.update({'F': F})
            eq_mkt_df = FeatureHAREq.eq_mkt_builder(**kwargs)
            try:
                symbol_df = symbol_df.join(eq_mkt_df, how='left')
            except ValueError:
                pass
            symbol_df = symbol_df.fillna(symbol_df.ewm(span=12, min_periods=1).mean())
        return symbol_df.dropna()
