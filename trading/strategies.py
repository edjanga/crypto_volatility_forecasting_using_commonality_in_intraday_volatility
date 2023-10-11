import concurrent.futures
import os.path
import pdb
import typing
import pandas as pd
import numpy as np
import pytz

from data_centre.data import Reader
import sqlite3
from scipy.stats import skew, kurtosis
from statsmodels.stats.stattools import robust_kurtosis, robust_skewness
import matplotlib.pyplot as plt
from itertools import product
import plotly.express as px


class Trader:

    """
        Backtesting class
    """
    db_connect_y = sqlite3.connect(os.path.abspath('../data_centre/databases/y.db'), check_same_thread=False)
    reader_obj = Reader()
    PnL_dd = dict()
    returns_dd = dict()

    def __init__(self, top_performers: typing.List[str], bottom_performers: typing.List[str], hp: int, h: str = '30T'):
        self._h = h
        self._hp = int(hp*(pd.to_timedelta('1H')//pd.to_timedelta(self._h)))
        self._prices = Trader.reader_obj.prices_read().resample(self._h).last()
        self._returns = (self._prices-self._prices.shift(self._hp)).div(self._prices.shift(self._hp)).dropna()
        self._volumes = Trader.reader_obj.volumes_read().resample(self._h).last()
        tmp_volxprice = self._prices.mul(self._volumes)
        tmp_vol_sum = self._volumes
        self._benchmark = tmp_volxprice.div(tmp_vol_sum)
        self._top_performers = top_performers
        self._bottom_performers = bottom_performers

    def PnL(self) -> pd.DataFrame:
        def PnL_per_item(h: str, training_scheme: str, L: str, transformation: str,
                         regression_type: str, model: str) -> None:
            query = \
                f'SELECT \"y_hat\", \"symbol\", \"index\", \"model\", \"training_scheme\", \"L\",' \
                f'\"transformation\", \"regression\" FROM y_{L} WHERE \"training_scheme\" = \"{training_scheme}\"' \
                f' AND \"transformation\" = \"{transformation}\" AND \"regression\" = \"{regression_type}\"' \
                f' AND \"model\" = \"{model}\";'
            y_hat = pd.read_sql(query, con=Trader.db_connect_y)[['index', 'y_hat', 'symbol']]
            y_hat = y_hat.set_index('index')
            y_hat.index = pd.to_datetime(y_hat.index, utc=pytz.UTC)
            y_hat = \
                pd.pivot_table(y_hat.reset_index(), values='y_hat',
                               columns='symbol', index='index').resample(h).sum()**(.5)
            signals = y_hat.rank(axis=1, ascending=False, pct=True)
            n_pair = signals.shape[1]//5
            deciles = signals.iloc[0, :].sort_values()
            top, bottom = deciles.iloc[:n_pair].values, deciles.iloc[-n_pair:].values
            long_signals, short_signals = signals.isin(bottom).astype(float), signals.isin(top).astype(float)
            signals = long_signals.add(short_signals).shift().dropna()
            signals = signals.add(-1*signals.shift(self._hp)).shift(1).fillna(0)
            Trader.returns_dd[(f'{training_scheme}_{L}_{regression_type}_{model}')] = \
                self._returns.mul(signals).mean(axis=1).fillna(0)
            cumPnL = pd.Series(Trader.returns_dd[(f'{training_scheme}_{L}_{regression_type}_{model}')].cumsum(),
                               name=f'{training_scheme}_{L}_{regression_type}_{model}').fillna(0)
            Trader.PnL_dd[f'{training_scheme}_{L}_{regression_type}_{model}'] = cumPnL
        with concurrent.futures.ThreadPoolExecutor() as executor:
            PnL_per_item_results_dd = {option: executor.submit(PnL_per_item, h=self._h,
                                                               training_scheme=option[0],
                                                               L=option[1], transformation='log',
                                                               regression_type=option[2], model=option[3])
                                       for option in self._top_performers.values.tolist()+
                                       self._bottom_performers.values.tolist()}


if __name__ == '__main__':
    reader_obj = Reader()
    returns = reader_obj.returns_read()
    """Selection of top 5 performing models using QLIKE"""
    performance = pd.read_csv('../performance.csv')
    performance = performance.loc[~performance.regression.isin(['ridge', 'risk_metrics']), :]
    performance = pd.pivot(data=performance,
                           columns=['metric'],
                           values='values', index=['training_scheme', 'L', 'regression', 'model'])[['qlike']]
    n_performers = 5
    """Top 5 and bottom 5 performers"""
    top_performers = performance.sort_values(by='qlike').iloc[:n_performers].index
    bottom_performers = performance.sort_values(by='qlike').iloc[-n_performers:].index
    trader_obj = Trader(top_performers=top_performers, bottom_performers=bottom_performers, hp=6)
    trader_obj.PnL()
    returns = pd.concat(trader_obj.returns_dd, axis=1)
    stats = returns.agg([lambda x: x.mean()*(pd.to_timedelta('365D')//pd.to_timedelta(trader_obj._h)),
                         lambda x: x.std()*((pd.to_timedelta('365D')//pd.to_timedelta(trader_obj._h))*.5)])
    stats.index = ['mean', 'std']
    stats = stats.transpose()
    stats = \
        stats.assign(skew=
                     skew(returns)*((pd.to_timedelta('365D')//pd.to_timedelta(trader_obj._hp*trader_obj._h))**.5),
                     kurtosis=
                     kurtosis(returns)*((pd.to_timedelta('365D')//pd.to_timedelta(trader_obj._hp*trader_obj._h))**.5),
                     sharpe=stats['mean'].div(stats['std']))
    stats = stats.round(2).transpose()
    cumPnL = pd.concat(trader_obj.PnL_dd, axis=1)
    vwap = \
        trader_obj._benchmark.sub(trader_obj._benchmark.shift(trader_obj._hp)).div(trader_obj._benchmark.shift(
            trader_obj._hp)).fillna(0).mean(axis=1).cumsum()
    vwap.name = 'VWAP'
    stats = stats.assign(VWAP=[vwap.mean(),
                               vwap.std(),
                               skew(vwap),
                               kurtosis(vwap),
                               vwap.mean()/vwap.std()])
    stats = stats.transpose()
    cumPnL = cumPnL.assign(VWAP=vwap)
    PnL_per_day = cumPnL.diff().fillna(0).mean()
    stats['PnL/day'] = 10_000*PnL_per_day
    print(stats.sort_values(by='sharpe', ascending=False).round(2).to_latex())
    cumPnL = pd.melt(cumPnL, ignore_index=False, var_name='model/strategy', value_name='values')
    fig = px.line(data_frame=cumPnL, y='values', color='model/strategy', title='Trading performance: Cumulative PnL')
    fig.add_hline(y=0, line_width=1, line_dash='dash', line_color='black')
    fig.update_xaxes(tickangle=45, title='Date')
    fig.update_yaxes(title='Cumulative PnL')
    fig.write_image(os.path.abspath(f'../plots/cum_PnL.pdf'))


