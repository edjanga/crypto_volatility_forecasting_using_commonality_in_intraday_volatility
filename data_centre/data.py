import pdb

import pytz
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os
import numpy as np
import typing


class Reader:

    _min_max_scaler = MinMaxScaler()

    def __init__(self):
        _data_centre_dir = os.path.abspath(__file__).replace('/data.py', '/tmp')
        self._directory = _data_centre_dir
        self._df = \
            pd.concat([pd.read_parquet(f'{self._directory}/aggregate{str(year)}') for year in [2021, 2022, 2023]])
        self._df.ffill(inplace=True)
        self._df.dropna(axis=1, inplace=True)
        self._volume_df = \
            pd.concat([pd.read_parquet(f'{self._directory}/aggregate{str(year)}_volume') for
                       year in [2021, 2022, 2023]])[self._df.columns]
        self._df.index = pd.to_datetime(self._df.index, utc=pytz.UTC)
        self._volume_df.index = pd.to_datetime(self._volume_df.index, utc=pytz.UTC)
        most_liquid_pairs = self._volume_df.mul(self._df).sum().sort_values(ascending=False)[:20].index
        self._df = self._df[most_liquid_pairs]
        self._volume_df = self._volume_df[most_liquid_pairs]

    def prices_read(self, symbol: typing.Union[str, typing.List[str]] = None) -> pd.DataFrame:
        prices = self._df
        if symbol:
            if isinstance(symbol, str):
                symbol = [symbol]
            prices = prices[symbol]
        return prices

    def volumes_read(self) -> pd.DataFrame:
        volumes = pd.concat([pd.read_parquet(f'{self._directory}/aggregate{str(year)}_volume')
                             for year in [2021, 2022, 2023]])
        volumes = volumes[self._df.columns]
        volumes.index = pd.to_datetime(volumes.index, utc=pytz.UTC)
        return volumes

    def returns_read(self, cutoff_low: float = .01, cutoff_high: float = .01, raw: bool=False,
                     resampled: bool=True, symbol: typing.Union[typing.List[str], str]=None) -> pd.DataFrame:
        returns_df = np.log(self._df.div(self._df.shift()))
        if symbol:
            if isinstance(symbol, str):
                symbol = [symbol]
            returns_df = returns_df[symbol]
        returns_df.dropna(inplace=True)
        """Winsorise returns"""
        if not raw:
            returns_df = returns_df.apply(lambda x: winsorize(x, (cutoff_low, cutoff_high)))
        if resampled:
            returns_df = returns_df.resample('5T').sum()
        returns_df.fillna(returns_df.expanding().mean(), inplace=True)
        return returns_df

    def rv_read(self, cutoff_low: float = .01, cutoff_high: float = .01, raw: bool = False,
                symbol: typing.Union[typing.List[str], str] = None, variance: bool = True)\
            -> pd.DataFrame:
        rv_df = \
            self.returns_read(cutoff_low=cutoff_low, cutoff_high=cutoff_high,
                              raw=raw, resampled=False, symbol=symbol)**2
        rv_df.ffill(inplace=True)
        rv_df = rv_df.resample('5T').sum() if variance else rv_df.resample('5T').sum()**.5
        rv_df.replace(0, np.nan, inplace=True)
        rv_df.ffill(inplace=True)
        return rv_df

    def cdr_read(self, cutoff_low: float = .0001, cutoff_high: float = .0001) -> pd.DataFrame:
        cdr_df = pd.read_parquet(os.path.abspath(self._file),
                                 columns=['timestamp', 'volBuyQty', 'volSellQty', 'symbol'])
        cdr_df = cdr_df.set_index((['timestamp', 'symbol']))
        cdr_df = cdr_df.assign(volume=cdr_df.sum(axis=1))
        cdr_df.drop(['volBuyQty', 'volSellQty'], axis=1, inplace=True)
        cdr_df = pd.pivot(cdr_df.reset_index(), index='timestamp', columns='symbol', values='volume')
        cdr_df.index = pd.to_datetime(cdr_df.index)
        cdr_df = cdr_df.apply(lambda x: winsorize(x, (cutoff_low, cutoff_high)))
        cdr_df = cdr_df.resample('5T').last() / cdr_df.resample('5T').sum()
        return cdr_df

    def csr_read(self, feature_range: typing.Tuple[float] = None,
                 cutoff_low: float = .0001, cutoff_high: float = .0001) -> pd.DataFrame:
        if feature_range:
            Reader._min_max_scaler.feature_range = feature_range
        csr_df = pd.read_parquet(os.path.abspath(self._file),
                                 columns=['timestamp', 'bidPx', 'askPx', 'symbol'])
        csr_df = csr_df.set_index((['timestamp', 'symbol']))
        csr_df = csr_df.assign(px=csr_df.mean(axis=1))
        csr_df.drop(['bidPx', 'askPx'], axis=1, inplace=True)
        csr_df = pd.pivot(csr_df.reset_index(), index='timestamp', columns='symbol', values='px')
        csr_df.index = pd.to_datetime(csr_df.index)
        csr_df = csr_df.apply(lambda x: winsorize(x, (cutoff_low, cutoff_high)))
        csr_df = csr_df.resample('5T').last()
        csr_df = pd.DataFrame(data=Reader._min_max_scaler.fit_transform(X=csr_df),
                              index=csr_df.index, columns=csr_df.columns)
        csr_df.replace(0, np.nan, inplace=True)
        csr_df.ffill(inplace=True)
        return csr_df

    def correlation_read(self, cutoff_low: float = .01, cutoff_high: float = .01, raw: bool=False) -> pd.DataFrame:
        returns = self.returns_read(cutoff_low=cutoff_low, cutoff_high=cutoff_high, raw=raw)
        correlation_matrix = returns.rolling(window=4).corr().dropna()
        correlation = correlation_matrix.droplevel(axis=0, level=1).mean(axis=1)
        correlation = correlation.groupby(by=correlation.index).mean()
        correlation.replace(np.inf, np.nan, inplace=True)
        correlation.replace(-np.inf, np.nan, inplace=True)
        correlation.ffill(inplace=True)
        correlation.name = 'CORR'
        correlation = pd.DataFrame(correlation)
        return correlation
