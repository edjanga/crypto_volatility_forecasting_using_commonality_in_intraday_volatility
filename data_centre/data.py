import pdb
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os
import numpy as np
import typing


class Reader:

    _min_max_scaler = MinMaxScaler()

    def __init__(self, file: typing.Union[typing.List[str], str] = os.path.abspath('./data_centre/tmp/aggregate2022')):
        self._directory = file

    def returns_read(self, cutoff_low: float = .01, cutoff_high: float = .01, raw: bool=False,
                     resampled: bool=True, symbol: typing.Union[typing.List[str], str]=None) -> pd.DataFrame:
        returns_df = pd.read_parquet(os.path.abspath(self._directory), columns=['timestamp', 'pret_1m', 'symbol'])
        returns_df = pd.pivot(index='timestamp', columns='symbol', values='pret_1m', data=returns_df)
        if symbol:
            if isinstance(symbol, str):
                symbol = [symbol]
            returns_df = returns_df[symbol]
        else:
            returns_df.drop('BUSDUSDT', axis=1, inplace=True)
        returns_df.index = pd.to_datetime(returns_df.index)
        returns_df.dropna(inplace=True)
        """Winsorise returns"""
        if not raw:
            returns_df = returns_df.apply(lambda x: winsorize(x, (cutoff_low, cutoff_high)))
        if resampled:
            returns_df = returns_df.resample('5T').sum()
        returns_df.replace(0, np.nan, inplace=True)
        returns_df.ffill(inplace=True)
        return returns_df

    def rv_read(self, cutoff_low: float = .01, cutoff_high: float = .01, raw: bool = False,
                symbol: typing.Union[typing.List[str], str] = None, variance: bool = False)\
            -> pd.DataFrame:
        rv_df = \
            self.returns_read(cutoff_low=cutoff_low, cutoff_high=cutoff_high,
                              raw=raw, resampled=False, symbol=symbol)**2
        rv_df = rv_df.resample('5T').sum() if variance else rv_df.resample('5T').sum()**.5
        return rv_df

    def cdr_read(self, cutoff_low: float = .05, cutoff_high: float = .05) -> pd.DataFrame:
        cdr_df = pd.read_parquet(os.path.abspath(self._directory),
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
                 cutoff_low: float = .01, cutoff_high: float = .01) -> pd.DataFrame:
        if feature_range:
            Reader._min_max_scaler.feature_range = feature_range
        csr_df = pd.read_parquet(os.path.abspath(self._directory),
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
