import concurrent.futures
import os
import pdb
import typing
import pandas as pd
import pytz
import sqlite3
import numpy as np
import os
from data_centre.data import OptionChainData, DBQuery
from typing import Callable, Union, List, Tuple, Dict
import scipy
from scipy import fft
from datetime import datetime
from tardis_dev import datasets
import plotly.express as px
from sklearn.linear_model import LinearRegression
from dateutil.relativedelta import relativedelta


class CharacteristicFunctions:

    def __init__(self):
        pass

    @staticmethod
    def function(model: str = 'black_scholes', **kwargs):
        if model == 'black_scholes':
            tmp1 = 1j*kwargs['u']*(np.log(kwargs['S'])+kwargs['r']*kwargs['t']-kwargs['sig']**2/2*kwargs['t'])
            tmp2 = -.5*kwargs['sig']**2*kwargs['t']*kwargs['u']**2
            phi = np.exp(tmp1 + tmp2)
        elif model == 'heston':
            if len(set(['kappa', 'theta', 'rho', 'eta']).intersection(set(list(kwargs.keys())))) != 4:
                raise ValueError('Please provide the parameters needed to calibrate Heston model properly.')
            d = \
                ((kwargs['rho']*kwargs['theta']*kwargs['u']*1j-kwargs['kappa'])**2-kwargs['theta']**2*
                 (-1j*kwargs['u']-kwargs['u']**2))**.5
            denom_g = (kwargs['kappa']-kwargs['rho']*kwargs['theta']*kwargs['u']*1j+d)
            num_g = (kwargs['kappa']-kwargs['rho']*kwargs['theta']*kwargs['u']*1j-d)
            g = num_g/denom_g
            tmp1 = 1j * kwargs['u'] * (np.log(kwargs['S']) + (kwargs['r'] - kwargs['q']) * kwargs['t'])
            scaling_factor_tmp2 = kwargs['eta']*kwargs['kappa']*kwargs['theta']**(-2)
            tmp2 = \
                scaling_factor_tmp2*(kwargs['kappa']-kwargs['rho']*kwargs['theta']*kwargs['u']*1j-d)\
                *kwargs['t']-2*np.log((1-g*np.exp(-d*kwargs['t']))/(1-g))
            scaling_factor_tmp3 = kwargs['sig']**2*kwargs['theta']**(-2)
            tmp3 = \
                scaling_factor_tmp3*(kwargs['kappa']-kwargs['rho']*kwargs['theta']*kwargs['u']*1j-d)*\
                (1-np.exp(-d*kwargs['t']) / (1-g*np.exp(-d*kwargs['t'])))
            phi = np.exp(tmp1+tmp2+tmp3)
        return phi


class OptionPricer:

    char_func_obj = CharacteristicFunctions()
    N = 4096
    ALPHA = 1.5
    ETA = .25

    def __init__(self, model: str = None):
        if model not in ['black_scholes', 'heston', None]:
            raise ValueError('Please provide a valid pricing model name.')
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, model: str) -> None:
        self._model = model

    def fast_pricing_vanillas(self, S: float, strike: float, sig: float, t: float,  r: float = 0, q: float = 0,
                              **kwargs) -> float:
        lmbd = 2*np.pi/(OptionPricer.N*OptionPricer.ETA)
        scaled_lmbd = lmbd*OptionPricer.N/2
        k = np.arange(start=-scaled_lmbd, stop=scaled_lmbd, step=lmbd)
        KK = np.exp(k)
        v = np.arange(start=0, stop=OptionPricer.N*OptionPricer.ETA, step=OptionPricer.ETA)
        sw = 3+(-1)**(np.arange(start=1, stop=OptionPricer.N+1, step=1.0))
        sw[0] = 1
        sw /= 3
        if kwargs is None:
            rho = np.exp(-r * t) * OptionPricer.char_func_obj.function(model=self._model,
                                                                       u=v - (OptionPricer.ALPHA + 1) * 1j,
                                                                       S=S, r=r, t=t, sig=sig, q=q)
        else:
            rho = np.exp(-r * t) * OptionPricer.char_func_obj.function(model=self._model,
                                                                       u=v - (OptionPricer.ALPHA + 1) * 1j, S=S, r=r,
                                                                       t=t, sig=sig, q=q, **kwargs)
        rho /= (OptionPricer.ALPHA**2+OptionPricer.ALPHA-v**2+1j*(2*OptionPricer.ALPHA+1)*v)
        A = rho*np.exp(1j*v*scaled_lmbd)*OptionPricer.ETA*sw
        Z = np.real(fft.fft(A))
        call_price = np.exp(-OptionPricer.ALPHA*k)/np.pi*Z
        pricer = scipy.interpolate.CubicSpline(KK, call_price)
        return pricer(strike)

    @staticmethod
    def bs_closed_formula(S: float, strike: float, sig: float, t: float,  r: float = 0, q: float = 0)\
            -> float:
        """To be used as benchmark model"""
        d1 = (np.log(S/strike)+(r-q+sig**2/2)*t)/sig*t**.5
        d2 = d1 - sig*t**.5
        return np.exp(-q*t)*S*scipy.stats.norm.cdf(d1)-np.exp(-r*t)*strike*scipy.stats.norm.cdf(d2)

    def objective_function(self, params: list, mkt_price: Union[np.ndarray, pd.Series, pd.DataFrame],
                           objective: str, *args) -> float:
        model_price = pd.Series(index=mkt_price.index, data=np.nan)
        for idx in range(mkt_price.shape[0]):
            if self._model == 'heston+\mathcal{M}':
                model_price[idx] = self.fast_pricing_vanillas(S=args[0][idx], sig=args[3][idx],
                                                              t=args[2][idx], kappa=params[0], theta=params[2],
                                                              rho=params[3], eta=params[1], strike=args[1][idx])
            elif self._model == 'heston':
                model_price[idx] = self.fast_pricing_vanillas(S=args[0][idx], sig=args[3], t=args[2][idx],
                                                              kappa=params[0], theta=params[2], rho=params[3],
                                                              eta=params[1], strike=args[1][idx])
        if objective == 'rmse':
            objective_value = ((mkt_price - model_price)**2).mean()**.5
        elif objective == 'arpe':
            objective_value = (np.abs(mkt_price - model_price).div(model_price)).mean()
        return objective_value

    def calibration(self, mkt_price: Union[np.ndarray, pd.Series, pd.DataFrame], **kwargs) -> Dict[str, float]:
        np.random.seed(123)
        if 'heston' in self._model:
            """kappa, eta, theta, rho"""
            params = \
                [np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)]
            args = (mkt_price, 'rmse', kwargs['underlying'], kwargs['strike'], kwargs['t'], kwargs['sig'])
        rres = scipy.optimize.minimize(self.objective_function, x0=params, args=args,
                                       # Constraint: Feller condition
                                       constraints={'type': 'ineq', 'fun': lambda x: 2*x[0]*x[1]-x[-2]**2}, tol=1e-4,
                                       options=dict(maxiter=10_000))
        return {param: value for param, value in zip(['kappa', 'eta', 'theta', 'rho'], rres.x)}

    def put_call_parity(self, call: Union[np.ndarray, pd.Series, pd.DataFrame],
                        S: Union[np.ndarray, pd.Series, pd.DataFrame, float],
                        strike: Union[np.ndarray, pd.Series, pd.DataFrame]) -> Union[np.ndarray, pd.Series,
    pd.DataFrame, float]:
        """
            We assume that r = 0. Consequently, future coincides with spot price.
        """
        return strike-S+call


class OptionTrader:

    option_pricer_obj = OptionPricer()
    params_dd = dict()

    def __init__(self, fees: float):
        self._fees = fees
        self._pricing_model = None

    @property
    def fees(self) -> float:
      return self._fees

    @property
    def pricing_model(self) -> str:
        return self._pricing_model

    @fees.setter
    def fees(self, fees: float) -> None:
        self._fees = fees

    @pricing_model.setter
    def pricing_model(self, pricing_model: str) -> None:
        self._pricing_model = pricing_model

    def process_mkt_data(self, data: pd.DataFrame, option_type: str = 'call') -> pd.DataFrame:
        mkt_data = data.loc[data.index.get_level_values(1) == f'{option_type}', :].dropna()
        mkt_data['time2expiration'] = \
            (mkt_data['expiration'] - mkt_data.index.get_level_values(-1)) / pd.to_timedelta('360D')
        return mkt_data

    def add_pnl(self, data: pd.DataFrame) -> pd.DataFrame:
        data['d_iv'] = data.groupby(by=[pd.Grouper(level='option_tag')])[['mark_iv']].diff()
        data['d_underlying'] = data.groupby(by=[pd.Grouper(level='option_tag')])[['underlying_price']].diff()
        data['d_underlying2'] = data['d_underlying'] ** 2
        data['d_t'] = data.index.get_level_values(-1).copy()
        data['d_t'] = data.groupby(by=[pd.Grouper(level='option_tag')], group_keys=False).apply(
            lambda x: x.d_t.diff() / pd.to_timedelta('360D'))
        data['PnL'] = data['theta'] * data['d_t'] + data['delta'] * data['d_underlying'] + \
                          data['vega'] * data['d_iv'] + .5 * data['gamma'] * data['d_underlying2']
        return data

    @staticmethod
    def set_sig(df: pd.DataFrame, sig: float, idx: int, **kwargs) -> pd.DataFrame:
        if idx == 0:
            df['sig'][0] = sig
        else:
            pdb.set_trace()
        return df

    def daily_pricing(self, calls: pd.DataFrame, puts: pd.DataFrame, date: datetime, L: str, **kwargs)\
            -> Tuple[pd.DataFrame]:
        start_date = date - relativedelta(days={'1W': 7, '1M': 30, '6M': 180}[L])
        train_calls = calls.loc[(calls.index.get_level_values(2).date >= start_date) &
                                (calls.index.get_level_values(2).date < date)]
        train_puts = puts.loc[(puts.index.get_level_values(2).date >= start_date) &
                              (puts.index.get_level_values(2).date < date)]
        test_calls = calls.loc[(calls.index.get_level_values(2).date == date)]
        test_puts = puts.loc[(puts.index.get_level_values(2).date == date)]
        train_calls = train_calls.assign(model_price=np.nan)
        train_puts = train_puts.assign(model_price=np.nan)
        test_calls = test_calls.assign(model_price=np.nan)
        test_puts = test_puts.assign(model_price=np.nan)
        OptionTrader.option_pricer_obj.model = self._pricing_model
        if self._pricing_model == 'heston+\mathcal{M}':
            train_calls, train_puts = \
                train_calls.join(kwargs['sig'], on='timestamp'), train_puts.join(kwargs['sig'], on='timestamp')
            test_calls = test_calls.join(kwargs['sig'], on='timestamp')
        elif self._pricing_model in ['black_scholes', 'heston']:
            sig = train_calls.mark_iv.div(100).groupby(by=[pd.Grouper(level='option_tag')]).mean().mean()
            if self._pricing_model == 'heston':
                train_calls['sig'] = np.nan
            else:
                params = None
        if self._pricing_model == 'heston+\mathcal{M}':
            params = option_pricer_obj.calibration(mkt_price=train_calls.mark_price,
                                                   sig=train_calls.sig,
                                                   underlying=train_calls.underlying_price,
                                                   strike=train_calls.strike_price,
                                                   t=train_calls.time2expiration)
        elif self._pricing_model == f'heston':
            params = option_pricer_obj.calibration(mkt_price=train_calls.mark_price, sig=sig,
                                                   underlying=train_calls.underlying_price,
                                                   strike=train_calls.strike_price,
                                                   t=train_calls.time2expiration)
        OptionTrader.params_dd[date] = params
        train_calls = train_calls.sort_index()
        train_puts = train_puts.sort_index()
        test_calls = test_calls.sort_index()
        test_puts = test_puts.sort_index()
        for idx in range(0, train_calls.shape[0]):
            if params is not None:
                if self._pricing_model == 'heston':
                    train_calls = train_calls.groupby(by=[pd.Grouper(level='option_tag')]).apply(
                        lambda x, sig, idx: OptionTrader.set_sig(x, sig, idx, **params), sig=sig, idx=idx, **params)
                    # if idx == 0:
                    #     train_calls['sig'][idx] = sig
                    # else:
                    #     w2 = params[-1]*np.random.normal()
                    #     train_calls['sig'][idx] = train_calls['sig'][idx-1]+params[0]*(
                    #             params[1]-train_calls['sig'][idx-1])*train_calls['d_t']+params[2]*np.sqrt(
                    #         train_calls['sig'][idx-1])*w2
                elif self._pricing_model == 'heston+\mathcal{M}':
                    train_calls['model_price'][idx] = \
                        OptionTrader.option_pricer_obj.fast_pricing_vanillas(S=train_calls.underlying_price[idx],
                                                                             strike=train_calls.strike_price[idx],
                                                                             sig=train_calls.sig[idx],
                                                                             t=train_calls.time2expiration[idx],
                                                                             **params)
            else:
                train_calls['model_price'][idx] = \
                    OptionTrader.option_pricer_obj.fast_pricing_vanillas(S=train_calls.underlying_price[idx],
                                                                         strike=train_calls.strike_price[idx],
                                                                         # Constant variance
                                                                         sig=sig,
                                                                         t=train_calls.time2expiration[idx])
        for idx in range(0, test_calls.shape[0]):
            if params is not None:
                test_calls['model_price'][idx] = \
                    OptionTrader.option_pricer_obj.fast_pricing_vanillas(S=test_calls.underlying_price[idx],
                                                                         strike=test_calls.strike_price[idx],
                                                                         sig=test_calls.sig[idx],
                                                                         t=test_calls.time2expiration[idx],
                                                                         **params)
            else:
                test_calls['model_price'][idx] = \
                    OptionTrader.option_pricer_obj.fast_pricing_vanillas(S=test_calls.underlying_price[idx],
                                                                         strike=test_calls.strike_price[idx],
                                                                         # Constant variance
                                                                         sig=sig,
                                                                         t=test_calls.time2expiration[idx])
        train_calls['model_price'] = \
            train_calls['model_price'].where(train_calls['model_price'] > 0, 0)
        test_calls['model_price'] = \
            test_calls['model_price'].where(test_calls['model_price'] > 0, 0)
        train_puts['model_price'] = \
            OptionTrader.option_pricer_obj.put_call_parity(call=train_calls['model_price'],
                                                           S=train_calls['underlying_price'],
                                                           strike=train_calls['strike_price']).values
        test_puts['model_price'] = \
            OptionTrader.option_pricer_obj.put_call_parity(call=test_calls['model_price'],
                                                           S=test_calls['underlying_price'],
                                                           strike=test_calls['strike_price']).values
        train_calls['model'] = self._pricing_model.title()
        train_puts['model'] = self._pricing_model.title()
        test_calls['model'] = self._pricing_model.title()
        test_puts['model'] = self._pricing_model.title()
        return train_calls, train_puts, test_calls, test_puts
        
    def signals(self, train_mkt_data: pd.DataFrame, test_mkt_data: pd.DataFrame) -> pd.DataFrame:
        lr = LinearRegression()
        lr.fit(X=train_mkt_data[['model_price', 'mark_price']].dropna()['model_price'].values.reshape(-1, 1),
               y=train_mkt_data[['model_price', 'mark_price']].dropna().mark_price.values.reshape(-1, 1)),
        residuals = \
            test_mkt_data['mark_price'] - (lr.intercept_[0] + lr.coef_[0] * test_mkt_data['model_price'])
        test_mkt_data = test_mkt_data.assign(signals=np.nan)
        test_mkt_data.loc[:, 'signals'] = np.where(residuals > 3 * residuals.std(), -1, test_mkt_data.signals)
        test_mkt_data.loc[:, 'signals'] = np.where(residuals < 3 * residuals.std(), 1, test_mkt_data.signals)
        test_mkt_data = test_mkt_data.fillna(0)
        test_mkt_data = pd.pivot_table(test_mkt_data.reset_index(), columns='option_tag', values='signals',
                                       index='timestamp')
        return test_mkt_data

    def backtest_per_strat(self, signals: pd.DataFrame, calls: pd.DataFrame, puts: pd.DataFrame) -> pd.Series:
        calls_PnL = signals.shift().mul(calls).sum(axis=1).cumsum().fillna(0).add(100)
        puts_PnL = signals.shift().mul(puts).sum(axis=1).cumsum().fillna(0).add(100)
        PnL = .5*(calls_PnL + puts_PnL).ffill()
        return PnL


if __name__ == '__main__':
    # option_pricer_obj = OptionPricer(model='heston')
    # option_trader_obj = OptionTrader(fees=.0004)
    option_chain_data_obj = OptionChainData()
    db_obj = DBQuery()
    api_key = 'TD.34Q-uNaLfiYDgGrL.ASWs359NTrG3xtC.hJ6BeUBTavJthip.bv88e75OfyB8uWY.Fj9Bir3R4yU8vBo.V8aa'
    universe = ['ETH']
    data = list()
    dates = list(pd.date_range(start=datetime(2021, 1, 1), end=datetime(2023, 7, 1), inclusive='left'))
    for idx in range(0, 14, len(dates)+1):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(lambda x: x, x=date) for date in dates[idx:idx+14]]
        for future in concurrent.futures.as_completed(futures):
            date = future.result()
            print(f'[FETCHING]: Option chain data for {date.strftime("%Y-%m-%d")} is being processed...')
            datasets.download(exchange='deribit', data_types=['options_chain'], from_date=date.strftime('%Y-%m-%d'),
                              to_date=date.strftime('%Y-%m-%d'), symbols=['OPTIONS'], api_key=api_key,
                              format='csv')
            tmp = pd.read_csv(f'./datasets/deribit_options_chain_{date.strftime("%Y-%m-%d")}_OPTIONS.csv.gz',
                              compression='gzip')
            os.remove(
                path=os.path.abspath(f'./datasets/deribit_options_chain_{date.strftime("%Y-%m-%d")}_OPTIONS.csv.gz'))
            chunk = option_chain_data_obj.resample_option_chain_data(data=tmp)
            print(f'[END FETCHING]: Option chain data for {date.strftime("%Y-%m-%d")} has been processed.')
            pdb.set_trace()
            chunk.to_parquet(date.strftime('%Y-%m-%d'))
    pdb.set_trace()
    data = pd.concat([pd.read_parquet(date.strftime('%Y-%m-%d')) for date in pd.date_range(start='2021-01-01',
                                                                                           end='2023-06-30',
                                                                                           freq='D',
                                                                                           inclusive='both')])
    liquid = data['volume'].quantile(.75)
    # Filtering - Part 2
    data = data.query(f'volume >= {liquid}')
    data.index.name = 'timestamp'
    data = data.set_index(['option_tag', 'type'], append=True).swaplevel(i=1, j=0).swaplevel(i=-1, j=1)
    potential_straddles_tag = data.sort_index(level=0).groupby(by=pd.Grouper(level='option_tag')).size() % 2 == 0
    potential_straddles_tag = potential_straddles_tag[potential_straddles_tag == True].index.tolist()
    data = data.loc[data.index.get_level_values(0).isin(potential_straddles_tag)].sort_index(level=[0, 1])
    data = \
         data.groupby(by=[pd.Grouper(level='option_tag'),
                          pd.Grouper(level='type'), pd.Grouper(level='timestamp', freq='30T')]).last().drop('volume',
                                                                                                            axis=1)
    data.to_csv('option_filtered_data_sample')
    tmp = pd.read_csv('option_filtered_data_sample', chunksize=10_000)
    # data = list()
    # for _, chunk in enumerate(tmp):
    #     chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], format='ISO8601', utc=True)
    #     chunk['expiration'] = pd.to_datetime(chunk['expiration'], format='ISO8601', utc=True)
    #     chunk = chunk.set_index(['option_tag', 'type', 'timestamp'])
    #     data.append(chunk)
    # data = pd.concat(data)
    # data.mark_price = (data.mark_price*data.underlying_price).round(2)
    # data.mark_iv = data.mark_iv.div(100)
    # calls, puts = option_trader_obj.process_mkt_data(data=data),\
    #     option_trader_obj.process_mkt_data(data=data, option_type='put')
    # calls = calls.reset_index(1)
    # puts = puts.reset_index(1)
    # idx_common = list(set(puts.index).intersection(set(calls.index)))
    # calls = calls.loc[idx_common].set_index('type', append=True).swaplevel(-1, 1)
    # puts = puts.loc[idx_common].set_index('type', append=True).swaplevel(-1, 1)
    # calls = option_trader_obj.add_pnl(calls)
    # calls['PnL'] = \
    #     calls['PnL']+(calls['PnL']*option_trader_obj.fees).where(
    #         calls['PnL']*option_trader_obj.fees < 0, -calls['PnL']*option_trader_obj.fees)
    # calls = calls.dropna()
    # puts = option_trader_obj.add_pnl(puts)
    # puts['PnL'] = \
    #     puts['PnL']+(puts['PnL']*option_trader_obj.fees).where(
    #         puts['PnL']*option_trader_obj.fees < 0, -puts['PnL']*option_trader_obj.fees)
    # puts = puts.dropna()
    # qlike = \
    #     db_obj.query_data(db_obj.best_model_for_all_windows_query('qlike'), table='qlike').sort_values(by='L',
    #                                                                                                    ascending=True)
    # suffix_name = \
    #     {'trading_session': {True: 'eq', False: 'eq_vixm'}, 'top_book': {True: 'top_book', False: 'full_book'}}
    # PnL_per_model = dict()
    # stats_per_PnL_model = dict()
    # prices_per_model = dict()
    # qlike = qlike.query('model == "ar"')
    # for idx1, row in qlike.iterrows():
    #     L = row['L']
    #     model, regression, training_scheme, trading_session, top_book, h = row['model'], row['regression'], \
    #         row['training_scheme'], row['trading_session'], row['top_book'], row['h']
    #     y_hat = db_obj.forecast_query(L=L, model=model, regression=regression,
    #                                   trading_session=trading_session, top_book=top_book,
    #                                   training_scheme=training_scheme)[['y_hat', 'symbol']]
    #     y_hat = pd.pivot(data=y_hat, columns='symbol', values='y_hat')[['ETHUSDT']]
    #     y_hat.index = pd.to_datetime(y_hat.index)
    #     y_hat = y_hat.resample(h).sum()
    #     y_hat = y_hat.loc['2022-06-01':'2022-06-29', :].rename(columns={'ETHUSDT': 'sig'}) ** .5
    #     ############################################################################################################
    #     ### Pricing procedure + Signals
    #     ############################################################################################################
    #     dates = list(set(np.unique(y_hat.index.date).tolist()))
    #     dates.sort()
    #     start_idx = dates.index(dates[0]+relativedelta(days={'1W': 7, '1M': 30, '6M': 180}['1W']))
    #     PnL_per_pricing_model = dict()
    #     stats_PnL_per_pricing_model = dict()
    #     for idx, pricing_model in enumerate(['heston', 'black_scholes', f'heston+$\mathcal{{M}}$']):
    #         train_calls, train_puts, test_calls, test_puts, straddles_signals = \
    #             list(), list(), list(), list(), list()
    #         option_trader_obj.pricing_model = pricing_model
    #         with concurrent.futures.ThreadPoolExecutor() as executor:
    #             if option_trader_obj.pricing_model == 'heston':
    #                 futures = [executor.submit(option_trader_obj.daily_pricing, calls=calls,
    #                                            puts=puts, L='1W', date=date,
    #                                            sig=y_hat) for date in dates[start_idx:]]
    #             else:
    #                 futures = \
    #                     [executor.submit(option_trader_obj.daily_pricing, calls=calls, puts=puts,
    #                                      L='1W', date=date) for date in dates[start_idx:]]
    #             for future in concurrent.futures.as_completed(futures):
    #                 tmp_train_calls, tmp_train_puts, tmp_test_calls, tmp_test_puts = future.result()
    #                 signals_calls = \
    #                     option_trader_obj.signals(train_mkt_data=tmp_train_calls, test_mkt_data=tmp_test_calls)
    #                 signals_calls = signals_calls.fillna(0)
    #                 signals_puts = \
    #                     option_trader_obj.signals(train_mkt_data=tmp_train_puts, test_mkt_data=tmp_test_puts)
    #                 signals_puts = signals_puts.fillna(0)
    #                 tmp_straddles_signals = ((signals_calls + signals_puts) / 2).fillna(0)
    #                 train_calls.append(tmp_train_calls)
    #                 test_calls.append(tmp_test_calls)
    #                 train_puts.append(tmp_train_puts)
    #                 test_puts.append(tmp_test_puts)
    #                 straddles_signals.append(tmp_straddles_signals)
    #         train_calls = pd.concat(train_calls).sort_index()
    #         test_calls = pd.concat(test_calls).sort_index()
    #         train_puts = pd.concat(train_puts).sort_index()
    #         test_puts = pd.concat(test_puts).sort_index()
    #         straddles_signals = pd.concat(straddles_signals).sort_index()
    #         straddles_signals = straddles_signals.fillna(0)
    #         test_calls = \
    #             pd.pivot_table(test_calls.reset_index(), columns='option_tag', values='PnL',
    #                            index='timestamp').fillna(0)
    #         test_calls = test_calls.reindex(y_hat.index).fillna(0)
    #         test_puts = \
    #             pd.pivot_table(test_puts.reset_index(), columns='option_tag', values='PnL',
    #                            index='timestamp').fillna(0)
    #         test_puts = test_puts.reindex(y_hat.index).fillna(0)
    #         PnL = ((test_calls+test_puts)*straddles_signals.shift()).sum(axis=1).resample('D').mean()
    #         pricing_model = pricing_model.title().replace('_', ' ')
    #         PnL_per_pricing_model[pricing_model] = PnL
    #         if idx == 0:
    #             prices_per_model[f'{L.lower()}'] = pd.concat([train_calls, train_puts]).reset_index(1)
    #         else:
    #             prices_per_model[f'{L.lower()}'] = \
    #                 pd.concat([prices_per_model[f'{L.lower()}'], pd.concat([train_calls,
    #                                                                         train_puts]).reset_index(1)])
    #     tmp_PnL = pd.concat(PnL_per_pricing_model, axis=1)
    #     stats_per_PnL_model[f'{L.lower()}'] = tmp_PnL.mean().div(tmp_PnL.std())
    #     PnL_per_model[f'{L.lower()}'] = 100+tmp_PnL.cumsum()
    # prices_per_model = pd.concat(prices_per_model).reset_index(level=0).rename(columns={'level_0': '$L_{train}$'})
    # PnL_per_model = pd.concat(PnL_per_model).reset_index(level=0).rename(columns={'level_0': '$L_{train}$'})
    # fig1 = px.line(PnL_per_model, facet_row='$L_{train}$')
    # fig2 = px.scatter(prices_per_model[['mark_price', 'model_price', 'type', '$L_{train}$', 'model']], x='model_price',
    #                   y='mark_price', color='model', trendline='ols', facet_col='type', facet_row='$L_{train}$',
    #                   category_orders={'$L_{train}$': ['1w', '1m', '6m'], 'type': ['put', 'call']})
    # strike_vs_prices = prices_per_model[['mark_price', 'model_price', 'type', '$L_{train}$', 'strike_price', 'model']]
    # tmp_mark_price = strike_vs_prices[['mark_price', 'type', '$L_{train}$', 'strike_price']].copy()
    # tmp_mark_price = tmp_mark_price.assign(model_price=tmp_mark_price.mark_price, model='market')
    # tmp_mark_price = tmp_mark_price.drop('mark_price', axis=1)
    # strike_vs_prices = strike_vs_prices.drop('mark_price', axis=1)
    # strike_vs_prices = pd.concat([strike_vs_prices, tmp_mark_price])
    # fig3 = px.scatter(strike_vs_prices[['model_price', 'type', '$L_{train}$', 'strike_price', 'model']],
    #                   x='strike_price', y='model_price', color='model', facet_col='type', facet_row='$L_{train}$',
    #                   category_orders={'$L_{train}$': ['1w', '1m', '6m'], 'type': ['put', 'call']})
    # stats_per_PnL_model = pd.concat(stats_per_PnL_model).reset_index()
    # stats_per_PnL_model = \
    #     stats_per_PnL_model.rename(columns={'level_0': '$L_{train}$', 'level_1': 'model', 0: 'Sharpe ratio'})
    # fig4 = px.bar(stats_per_PnL_model, x='$L_{train}$', y='Sharpe ratio', color='model', barmode='group')
    # fig2.update_layout(showlegend=True)
    # fig1.show()
    # fig2.show()
    # fig3.show()
    # fig4.show()
    # pdb.set_trace()