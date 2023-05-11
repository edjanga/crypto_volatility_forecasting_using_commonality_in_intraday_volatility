from scipy.stats.mstats import winsorize
from sklearn.preprocessing import Normalizer
import pandas as pd
import os
import numpy as np
import typing


class Reader:

    _normalizer = Normalizer()

    def __init__(self, file: typing.Union[typing.List[str], str] = os.path.abspath('./data_centre/tmp/aggregate2022')):
        self._directory = file

    def returns_read(self, cutoff_low: float = .01, cutoff_high: float = .01) -> pd.DataFrame:
        returns_df = pd.read_parquet(os.path.abspath(self._directory), columns=['timestamp', 'pret_1m', 'symbol'])
        returns_df = pd.pivot(index='timestamp', columns='symbol', values='pret_1m', data=returns_df)
        returns_df.index = pd.to_datetime(returns_df.index)
        returns_df.dropna(inplace=True)
        returns_df.drop('BUSDUSDT', axis=1, inplace=True)
        """Winsorise returns"""
        returns_df = returns_df.apply(lambda x: winsorize(x, (cutoff_low, cutoff_high)))
        returns_df = returns_df.resample('5T').sum()
        returns_df.replace(0, np.nan, inplace=True)
        returns_df.ffill(inplace=True)
        return returns_df

    def rv_read(self, cutoff_low: float = .01, cutoff_high: float = .01) -> pd.DataFrame:
        rv_df = pd.read_parquet(os.path.abspath(self._directory), columns=['timestamp', 'pret_1m', 'symbol'])
        rv_df = pd.pivot(index='timestamp', columns='symbol', values='pret_1m', data=rv_df)
        rv_df.index = pd.to_datetime(rv_df.index)
        rv_df.dropna(inplace=True)
        rv_df.drop('BUSDUSDT', axis=1, inplace=True)
        """Winsorise returns"""
        rv_df = rv_df.apply(lambda x: winsorize(x, (cutoff_low, cutoff_high)))**2
        rv_df = rv_df.resample('5T').sum()**.5
        rv_df.replace(0, np.nan, inplace=True)
        rv_df.ffill(inplace=True)
        # if save_figure:
        #     rv_df = pd.melt(rv_df, var_name='symbol', value_name='rv')
        #     fig = \
        #         px.box(rv_df, x='symbol', y='rv', color='symbol',
        #            title=f'RV - Boxplot (cutoff: {cutoff_low}, {cutoff_high})')
        #     fig.write_image(os.path.abspath(f'./rv_boxplot.png'))
        #     fig.show()
        return rv_df

    def cdr_read(self) -> pd.DataFrame:
        cdr_df = pd.read_parquet(os.path.abspath(self._directory),
                                 columns=['timestamp', 'volBuyQty', 'volSellQty', 'symbol'])
        cdr_df = cdr_df.set_index((['timestamp', 'symbol']))
        cdr_df = cdr_df.assign(volume=cdr_df.sum(axis=1))
        cdr_df.drop(['volBuyQty', 'volSellQty'], axis=1, inplace=True)
        cdr_df = pd.pivot(cdr_df.reset_index(), index='timestamp', columns='symbol', values='volume')
        cdr_df.index = pd.to_datetime(cdr_df.index)
        cdr_df = cdr_df.resample('5T').last() / cdr_df.resample('5T').sum()
        return cdr_df

    def csr_read(self) -> pd.DataFrame:
        csr_df = pd.read_parquet(os.path.abspath(self._directory),
                                 columns=['timestamp', 'bidPx', 'askPx', 'symbol'])
        csr_df = csr_df.set_index((['timestamp', 'symbol']))
        csr_df = csr_df.assign(px=csr_df.mean(axis=1))
        csr_df.drop(['bidPx', 'askPx'], axis=1, inplace=True)
        csr_df = pd.pivot(csr_df.reset_index(), index='timestamp', columns='symbol', values='px')
        csr_df.index = pd.to_datetime(csr_df.index)
        csr_df = csr_df.resample('5T').last()
        csr_df = pd.DataFrame(data=Reader._normalizer.fit_transform(X=csr_df),
                              index=csr_df.index, columns=csr_df.columns)
        return csr_df
