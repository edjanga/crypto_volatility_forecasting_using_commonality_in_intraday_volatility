import concurrent.futures
import os.path
import pdb
import typing
import pandas as pd
import numpy as np
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
    reader_obj = Reader(file='../data_centre/tmp/aggregate2022')
    PnL_dd = dict()
    returns_dd = dict()

    def __init__(self, top_performers: typing.List[str], bottom_performers: typing.List[str], hp: int, h: str = '30T'):
        self._h = h
        self._hp = hp
        self._returns = Trader.reader_obj.returns_read(raw=True).resample(h).sum()
        tmp_volxprice = Trader.reader_obj.prices_read().mul(Trader.reader_obj.volumes_read()).resample(self._h).sum()
        tmp_vol_sum = Trader.reader_obj.volumes_read().resample(self._h).sum()
        self._benchmark = tmp_volxprice.div(tmp_vol_sum)
        self._top_performers = top_performers
        self._bottom_performers = bottom_performers

    def PnL(self) -> pd.DataFrame:
        def PnL_per_item(h: str, returns: pd.DataFrame, training_scheme: str, L: str, transformation: str,
                         regression_type: str, model: str) -> None:
            query = \
                f'SELECT \"y_hat\", \"symbol\", \"timestamp\", \"model\", \"training_scheme\", \"L\",' \
                f'\"transformation\", \"regression\" FROM y_{L} WHERE \"training_scheme\" = \"{training_scheme}\"' \
                f' AND \"transformation\" = \"{transformation}\" AND \"regression\" = \"{regression_type}\"' \
                f' AND \"model\" = \"{model}\";'
            y_hat = pd.read_sql(query, con=Trader.db_connect_y)[['timestamp', 'y_hat', 'symbol']]
            y_hat = y_hat.set_index('timestamp')
            y_hat.index = pd.to_datetime(y_hat.index)
            y_hat = \
                pd.pivot_table(y_hat.reset_index(), values='y_hat',
                               columns='symbol', index='timestamp').resample(h).sum()**(.5)
            signals = y_hat.rank(axis=1, ascending=False, pct=True)
            n_pair = signals.shape[1]//10
            deciles = signals.iloc[0, :].sort_values()
            top, bottom = deciles.iloc[:n_pair].values, deciles.iloc[-n_pair:].values
            long_signals, short_signals = signals.isin(bottom).astype(float), signals.isin(top).astype(float)
            signals = long_signals.add(short_signals).shift().dropna()
            signals = signals.add(-1*signals.shift(self._hp))
            Trader.returns_dd[(f'{training_scheme}_{L}_{regression_type}_{model}')] = \
                signals.mul(returns).mean(axis=1).mul(self._hp).fillna(0)
            cumPnL = pd.Series(Trader.returns_dd[(f'{training_scheme}_{L}_{regression_type}_{model}')].cumsum(),
                               name=f'{training_scheme}_{L}_{regression_type}_{model}').fillna(0)
            Trader.PnL_dd[f'{training_scheme}_{L}_{regression_type}_{model}'] = cumPnL
        with concurrent.futures.ThreadPoolExecutor() as executor:
            PnL_per_item_results_dd = {option: executor.submit(PnL_per_item, h=self._h,
                                                               returns=self._returns, training_scheme=option[0],
                                                               L=option[1], transformation='log',
                                                               regression_type=option[2], model=option[3])
                                       for option in self._top_performers.values.tolist()+
                                       self._bottom_performers.values.tolist()}
        # option = self._top_performers[0]
        # PnL_per_item(h=self._h, returns=self._returns, training_scheme=option[0], L=option[1], transformation='log',
        #              regression_type=option[2], model=option[3])


if __name__ == '__main__':
    reader_obj = Reader(file=os.path.abspath('../data_centre/tmp/aggregate2022'))
    returns = reader_obj.returns_read()
    """Selection of top 5 performing models using QLIKE"""
    performance = pd.read_csv('../performance.csv')
    performance = pd.pivot(data=performance,
                           columns=['metric'],
                           values='values', index=['training_scheme', 'L', 'regression', 'model'])[['qlike']]
    """Top 5 and bottom 5 performers"""
    top_performers = performance.sort_values(by='qlike').iloc[:5].index
    bottom_performers = performance.sort_values(by='qlike').iloc[-5:].index
    trader_obj = Trader(top_performers=top_performers, bottom_performers=bottom_performers, hp=12)
    trader_obj.PnL()
    returns = pd.concat(trader_obj.returns_dd, axis=1)
    stats = returns.agg([lambda x: x.mean()*(pd.to_timedelta('1Y')//pd.to_timedelta(trader_obj._h)),
                         lambda x: x.std()*((pd.to_timedelta('1Y')//pd.to_timedelta(trader_obj._h))*.5)])
    stats.index = ['mean', 'std']
    stats = stats.transpose()
    stats = stats.assign(skew=
                         skew(returns)*((pd.to_timedelta('1Y')//pd.to_timedelta(trader_obj._hp*trader_obj._h))**.5),
                         kurtosis=
                         kurtosis(returns)*((pd.to_timedelta('1Y')//pd.to_timedelta(trader_obj._hp*trader_obj._h))**.5),
                         sharpe=stats['mean'].div(stats['std']))
    stats = stats.round(4)
    print(stats.sort_values(by='sharpe', ascending=False).to_latex())
    cumPnL = pd.concat(trader_obj.PnL_dd, axis=1)#.droplevel(axis=1, level=0)
    vwap = np.log(trader_obj._benchmark/trader_obj._benchmark.shift()).mean(axis=1).cumsum().fillna(0)
    vwap.name = 'VWAP'
    cumPnL = cumPnL.assign(VWAP=vwap)
    cumPnL = pd.melt(cumPnL, ignore_index=False, var_name='model/strategy', value_name='values')
    fig = px.line(data_frame=cumPnL, y='values', color='model/strategy', title='Trading performance: Cumulative PnL')
    fig.add_hline(y=0, line_width=1, line_dash='dash', line_color='black')
    fig.update_xaxes(tickangle=45, title='Date')
    fig.update_yaxes(title='Cumulative PnL')
    fig.write_image(os.path.abspath(f'../plots/cum_PnL.pdf'))


