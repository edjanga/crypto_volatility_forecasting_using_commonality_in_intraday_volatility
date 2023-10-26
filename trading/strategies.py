import concurrent.futures
import os
import pdb
import typing
import pandas as pd
import pytz
from data_centre.data import Reader
import sqlite3
import vectorbt as vbt
import numpy as np


class Trader:

    """
        Backtesting class
    """
    _data_centre_dir = os.path.abspath(__file__).replace('/trading/strategies.py', '/data_centre')
    db_connect_y = sqlite3.connect(os.path.abspath(f'{_data_centre_dir}/databases/y.db'), check_same_thread=False)
    reader_obj = Reader()
    PnL_dd = dict()
    returns_dd = dict()

    def __init__(self, top_performers: typing.List[str], bottom_performers: typing.List[str],
                 hp: int = 1, h: str = '30T'):
        self._h = h
        self._hp = hp
        self._prices = Trader.reader_obj.prices_read().resample(self._h).last()
        self._volumes = Trader.reader_obj.volumes_read().resample(self._h).last()
        self._top_performers = top_performers
        self._bottom_performers = bottom_performers

    def PnL(self) -> pd.DataFrame:
        def PnL_per_item(h: str, vol_regime: str, training_scheme: str, transformation: str,
                         regression_type: str, model: str, L: str) -> None:
            query = \
                f'SELECT \"y_hat\", \"symbol\", \"timestamp\", \"model\", \"training_scheme\", \"L\",' \
                f'\"transformation\", \"regression\" FROM y_{L} WHERE \"training_scheme\" = \"{training_scheme}\"' \
                f' AND \"transformation\" = \"{transformation}\" AND \"regression\" = \"{regression_type}\"' \
                f' AND \"model\" = \"{model}\" AND vol_regime = \"{vol_regime}\";'
            y_hat = pd.read_sql(query, con=Trader.db_connect_y)[['timestamp', 'y_hat', 'symbol']]
            y_hat = y_hat.set_index('timestamp')
            y_hat.index = pd.to_datetime(y_hat.index, utc=pytz.UTC)
            y_hat = pd.pivot_table(y_hat.reset_index(), values='y_hat', columns='symbol',
                                   index='timestamp').resample(h).sum()**(.5)
            y_hat.replace(0, np.nan, inplace=True)
            y_hat.ffill(inplace=True)
            signals = y_hat.rank(axis=1, ascending=False, pct=True)
            n_pair = signals.shape[1]//min(10, signals.shape[1])
            deciles = signals.iloc[0, :].sort_values()
            top, bottom = deciles.iloc[:n_pair].values, deciles.iloc[-n_pair:].values
            long_signals, short_signals = signals.isin(bottom).astype(float), signals.isin(top).astype(float)
            self._prices = self._prices[long_signals.columns.tolist()]
            long_signals, short_signals = long_signals[self._prices.columns], short_signals[self._prices.columns]
            pf = vbt.Portfolio.from_signals(self._prices.loc[long_signals.index, :], entries=long_signals,
                                            exits=long_signals.shift(self._hp), init_cash=10000, size=1,
                                            short_entries=short_signals, short_exits=short_signals.shift(self._hp),
                                            freq=pd.to_timedelta('30T'))
            cumPnL = pf.returns().sum(axis=1).add(1).cumprod()
            Trader.PnL_dd[f'{training_scheme}_{L}_{regression_type}_{model}'] = cumPnL
        return_options = lambda x, h: (x, h)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(return_options, x=options, h=self._h)
                       for options in self._top_performers.values.tolist()+self._bottom_performers.values.tolist()]
            for future in concurrent.futures.as_completed(futures):
                option, h = future.result()
                PnL_per_item(h=h, vol_regime=option[0], training_scheme=option[1], L=option[2], transformation='log',
                             regression_type=option[3], model=option[4])
