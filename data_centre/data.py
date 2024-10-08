import concurrent.futures
import glob
import pdb
from scipy.stats.mstats import winsorize
from typing import Tuple
import pytz
import pandas as pd
import os
import numpy as np
import typing
from datetime import datetime
import torch
import sqlite3
from tardis_dev import datasets
from dotenv import load_dotenv
import argparse
from dateutil.relativedelta import relativedelta
import plotly.express as px
load_dotenv()
API_KEY = os.getenv('API_KEY')


def extract_date_start_end(x: str) -> Tuple[str, int, int]:
    return x.split('/')[-1].split('_')[1], int(x.split('/')[-1].split('_')[2]), int(x.split('/')[-1].split('_')[3])


data_centre_dir_data = os.path.abspath(__file__).replace('/data.py', '/tmp')
data_centre_dir_dbs = os.path.abspath(__file__).replace('/data.py', '/databases')


class Reader:

    def __init__(self, num_sym: int = 20):

        self._num_sym = num_sym
        self._directory = data_centre_dir_data
        self._df = \
            pd.concat([pd.read_parquet(f'{self._directory}/aggregate{str(year)}') for year in [2021, 2022, 2023]])
        self._df = self._df.fillna(self._df.ewm(span=12, min_periods=12).mean())
        self._df.dropna(axis=1, inplace=True)
        self._volume_df = \
            pd.concat([pd.read_parquet(f'{self._directory}/aggregate{str(year)}_volume') for
                       year in [2021, 2022, 2023]])[self._df.columns]
        self._df.index = pd.to_datetime(self._df.index, utc=pytz.UTC)
        self._volume_df.index = pd.to_datetime(self._volume_df.index, utc=pytz.UTC)
        most_liquid_pairs = self._volume_df.mul(self._df).sum().sort_values(ascending=False)[:self._num_sym].index
        self._df = self._df[most_liquid_pairs]
        self._volume_df = self._volume_df[most_liquid_pairs]

    def prices_read(self, symbol: typing.Union[str, typing.List[str]] = None) -> pd.DataFrame:
        if symbol:
            if isinstance(symbol, str):
                symbol = [symbol]

            return self._df[symbol]
        else:
            return self._df

    def volumes_read(self) -> pd.DataFrame:
        volumes = pd.concat([pd.read_parquet(f'{self._directory}/aggregate{str(year)}_volume')
                             for year in [2021, 2022, 2023]])
        volumes = volumes[self._df.columns]
        volumes.index = pd.to_datetime(volumes.index, utc=pytz.UTC)
        return volumes

    def returns_read(self, cutoff_low: float = .01, cutoff_high: float = .01, raw: bool = False,
                     resampled: bool = True, symbol: typing.Union[typing.List[str], str] = None) -> pd.DataFrame:
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
        returns_df = returns_df.fillna(returns_df.ewm(span=12, min_periods=1).mean())
        return returns_df

    @staticmethod
    def process_vixm(filename: str) -> typing.Tuple[str, pd.DataFrame]:
        """ index: #row + utc """
        print(f'[Data processing]: {filename} is being processed...')
        df = pd.read_csv(filename, usecols=list(range(0, 40, 2)), header=None)
        date, start, end = extract_date_start_end(filename)
        ref_timestamp = pd.to_datetime(date, utc=True).timestamp()
        start, end = \
            datetime.fromtimestamp(ref_timestamp+start/1_000), datetime.fromtimestamp(ref_timestamp+end/1_000)
        ask, bid = df.iloc[:, ::2].values, df.iloc[:, 1::2].values
        orderbook = np.array([ask, bid]).astype(float)
        orderbook = torch.from_numpy(orderbook)
        prices = torch.mean(orderbook, dim=0).detach().numpy()
        prices = pd.DataFrame(prices, columns=[f'VIXM_{i}' for i in range(0, prices.shape[1])])
        prices.index = pd.date_range(start=start, end=end, tz=pytz.UTC, periods=df.shape[0])
        print(f'[Data processing]: {filename} has been processed.')
        return date, prices

    def rv_read(self, cutoff_low: float = .01, cutoff_high: float = .01, raw: bool = False,
                symbol: typing.Union[typing.List[str], str] = None, variance: bool = True, data: str = 'crypto')\
            -> pd.DataFrame:
        if data == 'crypto':
            rv_df = \
                self.returns_read(cutoff_low=cutoff_low, cutoff_high=cutoff_high, raw=raw, resampled=False,
                                  symbol=symbol)**2
            rv_df.ffill(inplace=True)
            rv_df = rv_df.resample('5T').sum() if variance else rv_df.resample('5T').sum() ** .5
            rv_df.replace(0, np.nan, inplace=True)
            rv_df = rv_df.fillna(rv_df.ewm(span=12, min_periods=1).mean())
        elif data == 'vixm':
            vixm_ls = glob.glob(f'{self._directory}/VIXM/*.csv')
            vixm = dict()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(Reader.process_vixm, filename) for filename in vixm_ls]
                for future in concurrent.futures.as_completed(futures):
                    date, tmp = future.result()
                    vixm[date] = tmp
            vixm_level = pd.concat(vixm).droplevel(axis=0, level=0).sort_index().resample('1T').last()
            vixm_returns = vixm_level.div(vixm_level.shift()).dropna().replace(-np.inf, np.nan).replace(0, np.nan)
            vixm_returns = vixm_returns.fillna(vixm_returns.ewm(span=12, min_periods=1).mean())
            vixm_returns = np.log(vixm_returns.div(vixm_returns.shift()).dropna())
            vixm_returns = vixm_returns.fillna(vixm_returns.ewm(span=12, min_periods=1).mean())
            vixm_returns = vixm_returns.apply(lambda x, low, high: winsorize(x, (low, high)),
                                              low=cutoff_low, high=cutoff_high)
            vixm_rv = (vixm_returns ** 2).resample('5T').sum()
            vixm_rv = vixm_rv.reindex(pd.date_range(start=vixm_rv.index[0].date(),
                                                    end=vixm_rv.index[-1].date() + relativedelta(days=1),
                                                    inclusive='left', freq='5T', tz=pytz.utc))
            vixm_rv = vixm_rv.fillna(vixm_rv.ewm(span=12, min_periods=1).mean()).dropna()
            vixm_r_bar = vixm_returns.resample('5T').mean()
            vixm_r_bar = vixm_r_bar.reindex(vixm_returns.index).ffill().dropna()
            vixm_returns = vixm_returns.loc[vixm_r_bar.index, :]
            (vixm_returns-vixm_r_bar).resample('5T').sum().div(vixm_rv.iloc[1:, :])
            c_hat = ((vixm_returns-vixm_r_bar)**2).resample('5T').sum().div(vixm_rv.iloc[1:, :]).sum(axis=1)
            rv_df = vixm_rv.loc[c_hat.index, :].mul(c_hat, axis=0)
            rv_df = rv_df.replace(0, np.nan)
            rv_df = rv_df.fillna(rv_df.ewm(span=12, min_periods=1).mean())
        return rv_df

    def correlation_read(self, cutoff_low: float = .01, cutoff_high: float = .01, raw: bool = False) -> pd.DataFrame:
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


class OptionChainData:

    def __init__(self):
        pass

    def download_option_chain_data(self, start_date: str, end_date: str, multithreading: int,
                                   universe: typing.Union[str, typing.List[str]] = 'ETH') -> None:
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        dates = list(pd.date_range(start=start_date, end=end_date, inclusive='left', freq='1D'))
        if isinstance(universe, str):
            universe = [universe]
        if multithreading:
            for idx in range(0, len(dates)):
                print(f'[FETCHING]: Option chain data between {dates[idx].strftime("%Y-%m-%d")} and '
                      f'{dates[idx].strftime("%Y-%m-%d")} is being processed...')
                datasets.download(exchange='deribit', data_types=['options_chain'],
                                  from_date=dates[idx].strftime('%Y-%m-%d'),
                                  to_date=dates[idx].strftime('%Y-%m-%d'), symbols=['OPTIONS'],
                                  api_key=API_KEY, format='csv', download_dir='./tmp/datasets')
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(lambda x: x, x=date) for date in dates[idx:idx + 7]]
                for future in concurrent.futures.as_completed(futures):
                    data = list()
                    date = future.result()
                    print(f'[PROCESSING]: Option chain data between {date.strftime("%Y-%m-%d")} is being processed...')
                    tmp = \
                        pd.read_csv(f'./tmp/datasets/deribit_options_chain_{date.strftime("%Y-%m-%d")}_OPTIONS.csv.gz',
                                    compression='gzip', chunksize=100_000)
                    os.remove(path=os.path.abspath(
                        f'./tmp/datasets/deribit_options_chain_{date.strftime("%Y-%m-%d")}_OPTIONS.csv.gz'))
                    for _, chunk in enumerate(tmp):
                        chunk = chunk.loc[chunk.symbol.str.contains('|'.join(universe)), :]
                        chunk = chunk.drop(['local_timestamp', 'underlying_index', 'open_interest', 'exchange',
                                            'last_price', 'ask_iv', 'bid_iv', 'bid_price', 'ask_price'], axis=1)
                        chunk['timestamp'] = [datetime.fromtimestamp(idx / 1_000_000) for idx in chunk['timestamp']]
                        chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], format='ISO8601', utc=True)
                        chunk['mark_iv'] = chunk['mark_iv'].div(100) ** 2
                        chunk['volume'] = (chunk['bid_amount'] + chunk['ask_amount']).mul(chunk['mark_price']).mul(
                            chunk['underlying_price'])
                        chunk = chunk.drop(['bid_amount', 'ask_amount'], axis=1)
                        chunk = chunk.set_index('timestamp')
                        data.append(pd.DataFrame(chunk))
                    data = pd.concat(data)
                    data = self.resample_option_chain_data(data=data)
                    print(f'[END PROCESSING]: Option chain data for {date.strftime("%Y-%m-%d")} has been processed.')
                    data.to_parquet(f'./tmp/datasets/{date.strftime("%Y-%m-%d")}')
        else:
            for date in dates:
                data = list()
                print(f'[FETCHING]: Option chain data between {date.strftime("%Y-%m-%d")} and '
                      f'{date.strftime("%Y-%m-%d")} is being processed...')
                datasets.download(exchange='deribit', data_types=['options_chain'],
                                  from_date=date.strftime('%Y-%m-%d'),
                                  to_date=date.strftime('%Y-%m-%d'), symbols=['OPTIONS'],
                                  api_key=API_KEY, format='csv', download_dir='./tmp/datasets')
                print(f'[PROCESSING]: Option chain data between {date.strftime("%Y-%m-%d")} is being processed...')
                tmp = pd.read_csv(f'./tmp/datasets/deribit_options_chain_{date.strftime("%Y-%m-%d")}_OPTIONS.csv.gz',
                                  compression='gzip', chunksize=100_000)
                os.remove(path=os.path.abspath(
                    f'./tmp/datasets/deribit_options_chain_{date.strftime("%Y-%m-%d")}_OPTIONS.csv.gz'))
                for _, chunk in enumerate(tmp):
                    chunk = chunk.loc[chunk.symbol.str.contains('|'.join(universe)), :]
                    chunk = chunk.drop(['local_timestamp', 'underlying_index', 'open_interest', 'exchange',
                                        'last_price', 'ask_iv', 'bid_iv', 'bid_price', 'ask_price'], axis=1)
                    chunk['timestamp'] = [datetime.fromtimestamp(idx / 1_000_000) for idx in chunk['timestamp']]
                    chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], format='ISO8601', utc=True)
                    chunk['mark_iv'] = chunk['mark_iv'].div(100) ** 2
                    chunk['volume'] = (chunk['bid_amount'] + chunk['ask_amount']).mul(chunk['mark_price']).mul(
                        chunk['underlying_price'])
                    chunk = chunk.drop(['bid_amount', 'ask_amount'], axis=1)
                    chunk = chunk.set_index('timestamp')
                    data.append(pd.DataFrame(chunk))
                data = pd.concat(data)
                data = self.resample_option_chain_data(data=data)
                print(f'[END PROCESSING]: Option chain data for {date.strftime("%Y-%m-%d")} has been processed.')
                data.to_parquet(f'./tmp/datasets/{date.strftime("%Y-%m-%d")}')

    def resample_option_chain_data(self, data: pd.DataFrame, freq: str = '30T') -> pd.DataFrame:
        call = data.query('type == "call"')
        put = data.query('type == "put"')
        data = list()
        for tmp in [call, put]:
            tmp = tmp.assign(option_tag=['-'.join(option_tag[:-1]) for option_tag in tmp['symbol'].str.split('-')])
            tmp = tmp.drop('symbol', axis=1)
            tmp = tmp.set_index(['option_tag'], append=True).swaplevel(i=-1, j=0)
            tmp['mark_price'] = tmp['mark_price'] * tmp['underlying_price']
            tmp['mark_iv'] = tmp['mark_iv'] ** (.5)
            tmp['expiration'] = [datetime.fromtimestamp(idx / 1_000_000) for idx in tmp['expiration']]
            tmp['expiration'] = pd.to_datetime(tmp['expiration'], format='ISO8601', utc=True)
            tmp = tmp.assign(moneyness=tmp['underlying_price'].div(tmp['strike_price']))
            tmp = tmp.groupby(by=[pd.Grouper(level='option_tag'),
                                  pd.Grouper(level='timestamp', freq=freq)]).agg({'type': 'last',
                                                                                  'strike_price': 'last',
                                                                                  'mark_price': 'last',
                                                                                  'mark_iv': 'last',
                                                                                  'underlying_price': 'last',
                                                                                  'delta': 'last',
                                                                                  'gamma': 'last', 'vega': 'last',
                                                                                  'theta': 'last', 'rho': 'last',
                                                                                  'volume': 'sum',
                                                                                  'expiration': 'last'})
            data.append(tmp)
        data = pd.concat(data)
        return data


class DBQuery:

    _list_models = ['ar', 'har', 'har_eq']
    _list_regressions = ['linear', 'lasso', 'ridge', 'elastic', 'lightgbm', 'var', 'lstm']
    _list_training_schemes = ['SAM', 'ClustAM', 'CAM', 'UAM']
    _list_Ls = ['1W', '1M', '6M']
    _list_vol_regimes = ['low', 'normal', 'high']
    _list_metrics = ['r2', 'mse', 'qlike']
    _db_connect_y = \
        sqlite3.connect(database=os.path.abspath(f'{data_centre_dir_dbs}/y.db'), check_same_thread=False)
    _db_connect_dd = {metric: sqlite3.connect(database=os.path.abspath(f'{data_centre_dir_dbs}/{metric}.db'),
                                              check_same_thread=False) for metric in _list_metrics}
    _db_connect_dd.update({'y': _db_connect_y})
    _fct = {False: 'MAX', True: 'MIN'}
    _reader = Reader()
    _vol_regime = _reader.rv_read().resample('D').sum().mean(axis=1)
    _vol_regime.name = 'vol_regime'

    def __init__(self):
        pass

    @staticmethod
    def training_query(L: str, training_scheme: str='SAM'):
        query = f'SELECT "y_hat", "L", "model", "regression", "trading_session", "top_book","timestamp", "symbol" ' \
                f'FROM y_{L} '\
                f'WHERE "training_scheme"=\"{training_scheme}\";'
        return query

    @staticmethod
    def forecast_query(L: str, model: str, regression: str, training_scheme: str, symbol: str = None,
                       trading_session: int = None, top_book: int = None) -> pd.DataFrame:
        query = f'SELECT "y_hat", "L", "vol_regime", "model", "regression", "trading_session", "top_book"'\
                f',"timestamp", "symbol" ' \
                f'FROM y_{L} ' \
                f'WHERE "L"=\"{L}\" AND "regression"=\"{regression}\" AND "training_scheme"=\"{training_scheme}\"'
        if isinstance(symbol, str):
            query = ' AND '.join((query, f'"symbol"=\"{symbol}\"'))
        if isinstance(model, str):
            query = ' AND '.join((query, f'"model"=\"{model}\"'))
        if isinstance(trading_session, int) & (model == 'har_eq'):
            query = ' AND '.join((query, f'"trading_session"={trading_session}'))
        if isinstance(top_book, int) & (model == 'har_eq'):
            query = ' AND '.join((query, f'"top_book"={top_book}'))
        query = ';'.join((query, ''))
        return pd.read_sql(sql=query, con=DBQuery._db_connect_y, index_col='timestamp')

    @staticmethod
    def forecast_performance(L: str, model: str, regression: str, training_scheme: str, symbol: str = None,
                             trading_session: int = None, top_book: int = None, metric: str='qlike') -> pd.DataFrame:
        query = f'SELECT "values", "L", "model", "regression", "trading_session", "top_book"' \
                f',"timestamp","symbol" FROM "{metric}_{L}\" ' \
                f'WHERE "L"=\"{L}\" AND "regression"=\"{regression}\" AND "training_scheme"=\"{training_scheme}\"'
        if isinstance(model, str):
            query = ' AND '.join((query, f'"model"=\"{model}\"'))
        if isinstance(symbol, str):
            query = ' AND '.join((query, f'"symbol"=\"{symbol}\"'))
        pdb.set_trace()
        if ~np.isnan(trading_session):
            query = ' AND '.join((query, f'"trading_session"=\"{trading_session}\"'))
        if ~np.isnan(top_book):
            query = ' AND '.join((query, f'"top_book"=\"{top_book}\"'))
        query = ';'.join((query, ''))
        query = pd.read_sql(sql=query, con=DBQuery._db_connect_dd[metric], index_col='timestamp')
        query.index = pd.to_datetime(query.index)
        return query

    # @staticmethod
    # def best_model_for_all_windows_query(metric: str) -> str:
    #     query = list()
    #     for idx, L in enumerate(DBQuery._list_Ls):
    #         # tmp_query = f'SELECT MIN("values")  AS  "values",  "model", "regression","training_scheme",' \
    #         #             f'"trading_session","top_book","L","h"' \
    #         #             f'FROM (SELECT AVG("values")  AS "values",  "model", "regression","training_scheme",' \
    #         #             f'"trading_session","top_book","L","h" FROM (SELECT "values","metric","model","L",' \
    #         #             f'"training_scheme","regression","trading_session","top_book","h" ' \
    #         #             f'FROM {metric}_{L}) GROUP BY "symbol","regression","model")'
    #         tmp_query = f'SELECT MIN ("values")-1 AS "values", "model", "regression","training_scheme",' \
    #                     f'"trading_session","top_book","L","h"' \
    #                     f'FROM' \
    #                     f'(SELECT AVG("values") AS "values", "model", "regression","training_scheme", ' \
    #                     f'"trading_session","top_book","L","h"' \
    #                     f'FROM ' \
    #                     f'(SELECT "values","model","L","training_scheme","regression","trading_session","top_book","h" ' \
    #                     f'FROM {metric}_{L}) GROUP BY "training_scheme","regression","model","L","h")'
    #         if idx == len(DBQuery._list_Ls)-1:
    #             tmp_query = ';'.join((tmp_query, ''))
    #         query.append(tmp_query)
    #     query = ' UNION ALL '.join(query)
    #     return query

    @staticmethod
    def best_model_for_all_windows_query() -> str:
        query = list()
        for idx, L in enumerate(DBQuery._list_Ls):
            # tmp_query = f'SELECT MIN("values") AS "values","model", "regression", "training_scheme",' \
            #             f'"trading_session","top_book", "L", "h" ' \
            #             f'FROM (SELECT AVG("values") AS "values", "model", "regression","training_scheme",' \
            #             f'"trading_session","top_book", "L", "h" ' \
            #             f'FROM (SELECT ("y"/"y_hat") - LN("y"/"y_hat") - 1 AS "values", "y", "y_hat", "model","L",' \
            #             f'"training_scheme","regression","trading_session","top_book", "h" FROM y_{L})' \
            #             f'GROUP BY "training_scheme","model","regression","trading_session","top_book", "L")'
            # tmp_query = f'SELECT MIN("values") AS "values", "model", "regression", "training_scheme",' \
            #             f'"trading_session", "top_book", "L", "h"' \
            #             f'FROM ' \
            #             f'(SELECT AVG("values") AS "values", "model", "regression", "training_scheme",' \
            #             f'"trading_session", "top_book", "L", "h"' \
            #             f'FROM' \
            #             f'(SELECT strftime("%Y-%m-%d %H:", "timestamp") || CASE WHEN ' \
            #             f'cast(strftime("%M", "timestamp") as integer) < 30 THEN "00" ELSE "30" END as timestamp,' \
            #             f'AVG("values") AS "values", "symbol", "training_scheme", "regression", "model",' \
            #             f'"trading_session", "top_book", "L", "h"' \
            #             f'FROM' \
            #             f'(SELECT EXP("y" / "y_hat") - ("y" / "y_hat") - 1 AS "values", "y", "y_hat", "model", "L",' \
            #             f'"training_scheme", "regression", "trading_session", "top_book", "h", "symbol", "timestamp"' \
            #             f'FROM y_{L})' \
            #             f'GROUP BY "symbol", "training_scheme", "regression", "model", "trading_session", "top_book",' \
            #             f'"timestamp")' \
            #             f'GROUP BY "training_scheme", "regression", "model", "trading_session", "top_book")'
            tmp_query = f'SELECT MIN("values") AS "values", "model", "regression","training_scheme",' \
                        f'"trading_session","top_book", "L", "h"' \
                        f'FROM ' \
                        f'(SELECT AVG("values") AS "values", "model", "regression","training_scheme",' \
                        f'"trading_session","top_book", "L", "h"' \
                        f'FROM ' \
                        f'(SELECT EXP("y"/"y_hat") - ("y"/"y_hat") - 1 AS "values", "model","L", "training_scheme",' \
                        f'"regression","trading_session","top_book", "h", "symbol" ,"timestamp"' \
                        f'FROM ' \
                        f'(SELECT strftime("%Y-%m-%d %H:", "timestamp") || CASE WHEN cast(strftime("%M", "timestamp")' \
                        f' as integer) < 30 THEN "00" ELSE "30" END as timestamp,' \
                        f' SUM("y") AS "y", SUM("y_hat") AS "y_hat","symbol","training_scheme", "regression",' \
                        f'"model","trading_session","top_book", "L", "h"' \
                        f'FROM ' \
                        f'(SELECT "y", "y_hat", "model","L","training_scheme","regression","trading_session",' \
                        f'"top_book", "h", "symbol" ,"timestamp" FROM y_{L}) ' \
                        f'GROUP BY "symbol", "training_scheme", "regression",  "model", "trading_session",' \
                        f'"top_book", "timestamp")) ' \
                        f'GROUP BY "training_scheme", "regression",  "model", "trading_session", "top_book")'
            if idx == len(DBQuery._list_Ls) - 1:
                tmp_query = ';'.join((tmp_query, ''))
            query.append(tmp_query)
        query = ' UNION ALL '.join(query)
        return query

    @staticmethod
    def best_model_for_all_market_regimes(latex: bool = True) -> pd.DataFrame:
        query = list()
        tmp_table = list()
        for idx, L in enumerate(DBQuery._list_Ls):
            tmp_query = f'SELECT "timestamp", AVG("values")  AS "values",  "model", "regression","training_scheme",' \
                        f'"trading_session","top_book","L","h" ' \
                        f'FROM (SELECT "values","metric","model","L", "training_scheme","regression",' \
                        f'"trading_session","top_book","h","timestamp" FROM qlike_{L}) ' \
                        f'GROUP BY  "timestamp", "training_scheme","model", "regression","trading_session","top_book"'
            query.append(tmp_query)
        query = ';'.join((' UNION ALL '.join(query), ''))
        chunks = pd.read_sql(sql=query, con=DBQuery._db_connect_dd['qlike'], chunksize=10_000)
        while True:
            try:
                chunk = next(chunks)
                tmp_table.append(chunk)
            except StopIteration:
                break
        table = pd.concat(tmp_table)
        table['timestamp'] = pd.to_datetime(table['timestamp'], utc=True)
        table = \
            table.assign(tag=table[['training_scheme', 'L', 'model', 'regression', 'trading_session',
                                    'top_book']].apply(tuple, axis=1))
        table.drop(['model', 'regression', 'training_scheme', 'trading_session', 'top_book', 'L', 'h'], axis=1,
                   inplace=True)
        table = table.groupby(by=[pd.Grouper(key='tag'), pd.Grouper(key='timestamp', freq='D')]).mean()
        DBQuery._vol_regime.index.name = table.index.names[-1]
        vol_regime = pd.cut(DBQuery._vol_regime, bins=[0, DBQuery._vol_regime.quantile(.45),
                                                       DBQuery._vol_regime.quantile(.9),
                                                       DBQuery._vol_regime.quantile(1)],
                            labels=['low', 'normal', 'high'])
        table = table.join(vol_regime, how='left')
        table = \
            table.groupby(by=[pd.Grouper(key='vol_regime'),
                              pd.Grouper(level='tag')]).apply(lambda x: x['values'].mean()).reset_index(0)
        table = \
            table.groupby(by=[pd.Grouper(key='vol_regime')]).apply(lambda x: (x[[0]].idxmin(), x[[0]].min()))
        table = pd.DataFrame(table)
        table.reset_index(inplace=True)
        table = table.assign(tag=table[0].apply(lambda x: x[0]), values=table[0].apply(lambda x: x[1]))
        table.drop(0, axis=1, inplace=True)
        table['values'] = table['values'].subtract(1)
        table = table.assign(training_scheme=table['tag'].apply(lambda x: x[0]), L=table['tag'].apply(lambda x: x[1]),
                             model=table['tag'].apply(lambda x: x[2]), regression=table['tag'].apply(lambda x: x[3]),
                             trading_session=table['tag'].apply(lambda x: x[-2]),
                             top_book=table['tag'].apply(lambda x: x[-1]))
        table.drop('tag', axis=1, inplace=True)
        if latex:
            ##########################################################################################################
            ## Formatting
            ##########################################################################################################
            table.fillna(np.nan, inplace=True)
            table.replace(np.nan, '', inplace=True)
            table['trading_session'] = table['trading_session'].where(table['trading_session'] != '1', 'EQ')
            table['model'] = table['model'].str.replace('_eq', '')
            table['model'] = table['model'].str.upper()
            table['L'] = table['L'].str.lower()
            regression_conversion = \
                {'lightgbm': r'\text{LightGBM}', 'lasso': r'\text{LASSO}', 'elastic': r'\text{ELASTIC}',
                 'ridge': r'\text{RIDGE}', 'linear': r'\text{LR}', 'lstm': r'\text{LSTM}', 'var': r'\text{VAR}',
                 'pcr': r'\text{PCR}'}
            for regression, target in regression_conversion.items():
                table['regression'] = table['regression'].str.replace(regression, target)
            plotly_default_colors_int = list(px.colors.DEFAULT_PLOTLY_COLORS)
            plotly_default_colors_int = \
                [list(map(lambda x: int(x),
                          colors.replace('rgb', '').replace('(', '').replace(')', '').replace(' ', '').split(',')))
                 for colors in plotly_default_colors_int]
            table['tag'] = \
                [
                    f'\colorbox{{rgb:red!10,{plotly_default_colors_int[idx][0]};'
                    f'green!10,{plotly_default_colors_int[idx][2]};'
                    f'blue!10,{plotly_default_colors_int[idx][1]}}}{{${row["model"]}_'
                    f'{{{row["trading_session"]}}}^' \
                    f'{{{row["training_scheme"],row["L"],row["regression"],row["top_book"]}}}: '
                    f'{round(row["values"],4)}$}}}}'
                    for idx, row in table[['values', 'training_scheme', 'L', 'model', 'regression', 'trading_session',
                                           'top_book']].iterrows()
                ]
            table = table[['vol_regime', 'tag']].set_index('vol_regime').transpose()[['low', 'normal', 'high']]
            print(table.to_latex())

    @staticmethod
    def query_data(query: str, table: str) -> pd.DataFrame:
        chunk_ls = list()
        chunk = pd.read_sql(sql=query, con=DBQuery._db_connect_dd[table], chunksize=10_000)
        for _, tmp in enumerate(chunk):
            chunk_ls.append(tmp)
        chunk = pd.concat(chunk_ls)
        if 'timestamp' in chunk.columns:
            chunk.index = pd.to_datetime(chunk.index, utc=True)
        return pd.concat(chunk_ls)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script handling various datasets used in this project.')
    parser.add_argument('--multithreading', default=1, type=int, help='Use multithreading to fetch data.')
    parser.add_argument('--start_date', type=str, help='Start date to fetch data from.')
    parser.add_argument('--end_date', type=str, help='End date to fetch data to (exclusive).')
    args = parser.parse_args()
    option_chain_data_obj = OptionChainData()
    option_chain_data_obj.download_option_chain_data(multithreading=args.multithreading, start_date=args.start_date,
                                                     end_date=args.end_date)