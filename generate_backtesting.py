import concurrent.futures
import datetime
import glob
import pdb
import pandas as pd
import numpy as np
from typing import Tuple
from dateutil.relativedelta import relativedelta
import plotly.express as px
from plotly.subplots import make_subplots
from data_centre.data import DBQuery
import os
from data_centre.data import OptionChainData
from model.lab import qlike_score
import argparse
from arch import arch_model
from data_centre.data import Reader
from arch.__future__ import reindexing
reindexing = True

TITLE_FONT_SIZE = 40
LABEL_AXIS_FONT_SIZE = 20
LEGEND_FONT_SIZE = 18


class Trader:

    reader_obj = Reader()

    def __init__(self, fees: float):
        self._fees = fees

    @property
    def fees(self) -> float:
        return self._fees

    @fees.setter
    def fees(self, fees: float) -> None:
        self._fees = fees

    def fit_garch(self, date: datetime.datetime, L_train: dict, L: str, returns: pd.DataFrame,
                  GARCH: pd.DataFrame) -> None:
        start = (date - relativedelta(days=L_train[L])).strftime('%Y-%m-%d')
        print(f'[Start Training]: GARCH model^{L} training on {date} has started.')
        GARCH_model = arch_model(y=returns.loc[start:date.strftime('%Y-%m-%d')].values.reshape(-1),
                                 vol='GARCH', p=1, q=1, mean='Constant', rescale=False)
        GARCH_model = GARCH_model.fit(update_freq=0, disp=False)
        rv_hat = \
            GARCH_model.forecast(horizon=pd.to_timedelta('1D') // pd.to_timedelta(h)).variance.values[-1, :]
        rv_hat = rv_hat.reshape(rv_hat.shape[0], 1)# / GARCH_model.scale ** 2
        GARCH.loc[date.strftime('%Y-%m-%d'):date.strftime('%Y-%m-%d')] = rv_hat
        print(f'[End of Training]: GARCH model^{L} training on {date} has been completed.')


class OptionTrader(Trader):

    def __init__(self, fees: float):
        Trader.__init__(self, fees)

    def process_mkt_data(self, data: pd.DataFrame, option_type: str = 'call') -> pd.DataFrame:
        mkt_data = data.loc[data.index.get_level_values(1) == f'{option_type}', :]
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

    def pivot(self, mkt_data: pd.DataFrame, h: str) -> Tuple[pd.DataFrame]:
        ls = list()
        index = pd.to_datetime(pd.date_range(start='2021-01-01', end='2023-07-01', freq=h, inclusive='left'),
                               utc=True)
        for col in ['strike_price', 'mark_price', 'underlying_price', 'delta',
                    'gamma', 'vega', 'theta', 'rho', 'expiration', 'dt', 'mark_iv']:
            try:
                if col == 'dt':
                    tmp = pd.DataFrame(index=tmp.index, columns=tmp.columns, data=np.nan)
                    tmp.iloc[1:, 0] = (tmp.index[1:] - tmp.index[0:-1]) / pd.to_timedelta('360D')
                    tmp = tmp.ffill(axis=1)
                else:
                    tmp = pd.pivot_table(mkt_data.reset_index(), index='timestamp', columns='option_tag', values=col)
                    tmp = tmp.reindex(index)
                    # tmp = tmp.ffill()
                    if col == 'expiration':
                        tmp = tmp.bfill()
                        tmp = tmp.subtract(tmp.index, axis=0)/pd.to_timedelta('360D')
                        tmp = tmp.where(tmp > 0, np.nan)
            except KeyError:
                continue
            else:
                ls.append(tmp)
        return tuple(ls)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script backtesting option trading strategies.')
    parser.add_argument('--liquidity', default=.80, type=float, help='Threshold of liquid to keep for backtesting.')
    parser.add_argument('--performance', default='returns', type=str,
                        help='Ways to compute P&L: greeks or return based.')
    parser.add_argument('--min_expiration', type=str, help='Minimum expiration to filter option chain data.',
                        default='7D')
    parser.add_argument('--max_expiration', type=str, help='Maximum expiration to filter option chain data.',
                        default='180D')
    parser.add_argument('--min_moneyness', type=float, help='Minimum moneyness to filter option chain data.',
                        default=.95)
    parser.add_argument('--max_moneyness', type=float, help='Maximum moneyness to filter option chain data.',
                        default=1.05)
    parser.add_argument('--h', type=str, help='Time interval.', default='30T')
    parser.add_argument('--save', type=int, help='Save or show figures.', default=0)
    parser.add_argument('--signal_strength', type=float, help='Threshold used to assess how certain signals are.',
                        default=.05)
    args = parser.parse_args()
    index = pd.to_datetime(pd.date_range(start='2021-01-01', end='2023-07-01', freq=args.h, inclusive='left'),
                           utc=True)
    db_obj = DBQuery()
    qlike = \
        db_obj.query_data(db_obj.best_model_for_all_windows_query('qlike'), table='qlike').sort_values(by='L',
                                                                                                       ascending=True)
    suffix_name = \
        {'trading_session': {True: 'eq', False: 'eq_vixm'}, 'top_book': {True: 'top_book', False: 'full_book'}}
    PnL_per_strat = dict()
    trades_per_strat = dict()
    rv_models = dict()
    qlike_models = dict()
    rv_GARCH = dict()
    qlike_GARCH = dict()
    L_train = {'1W': 7, '1M': 30, '6M': 180}
    trader_obj = OptionTrader(fees=.0003)
    option_chain_data_obj = OptionChainData()
    files = [file for file in glob.glob('../data_centre/tmp/datasets/*-*-*')]
    data = pd.concat([pd.read_parquet(option_chain) for option_chain in files])
    if args.performance != 'greeks':
        data = data.loc[:, ~data.columns.isin(['delta', 'gamma', 'vega', 'theta', 'rho'])]
    moneyness = data.underlying_price.div(data.strike_price)
    data = data.assign(moneyness=moneyness.values)
    data = data.query(f"moneyness >= {args.min_moneyness} & moneyness <= {args.max_moneyness}")
    data['volume'] = data.volume * data.mark_price * data.underlying_price
    liquidity = data['volume'].quantile(args.liquidity)
    data = data.query(f'volume >= {args.liquidity}')
    data = data.set_index('type', append=True).swaplevel(i=1, j=-1)
    data = data.assign(
        time2expiration=(data.expiration - data.index.get_level_values(-1)) / pd.to_timedelta('360D'))
    data = data.query(f'time2expiration >= {pd.to_timedelta(args.min_expiration) / pd.to_timedelta("360D")} &'
                      f'time2expiration <= {pd.to_timedelta(args.max_expiration) / pd.to_timedelta("360D")}')
    calls, puts = trader_obj.process_mkt_data(data=data), trader_obj.process_mkt_data(data=data, option_type='put')
    if args.performance == 'greeks':
        calls_strike_price, calls_mark_price, calls_underlying_price, calls_delta, \
            calls_gamma, calls_vega, calls_theta, calls_rho, calls_expiration, calls_dt, calls_mark_iv = \
            trader_obj.pivot(calls, args.h)
        puts_strike_price, puts_mark_price, puts_underlying_price, puts_delta, \
            puts_gamma, puts_vega, puts_theta, puts_rho, puts_expiration, puts_dt, puts_mark_iv = \
            trader_obj.pivot(puts, args.h)
        calls_PnL = calls_theta * calls_dt + calls_delta * calls_underlying_price.diff() + \
                    calls_vega * calls_mark_iv + .5 * calls_gamma * calls_underlying_price.diff() ** 2
        puts_PnL = puts_theta * puts_dt + puts_delta * puts_underlying_price.diff() + puts_vega * puts_mark_iv + \
                   .5 * puts_gamma * puts_underlying_price.diff() ** 2
    elif args.performance == 'returns':
        _, calls_mark_price, _, _, _, _ = trader_obj.pivot(calls, args.h)
        _, puts_mark_price, _, _, _, _ = trader_obj.pivot(puts, args.h)
    straddle_ls = list(set(calls_mark_price.columns).intersection(set(puts_mark_price.columns)))
    PnL = calls_mark_price.filter(regex=f'{"|".join(straddle_ls)}') + \
          puts_mark_price.filter(regex=f'{"|".join(straddle_ls)}')
    straddles = (~calls_mark_price.filter(regex=f'{"|".join(straddle_ls)}').isnull()) + \
                (~puts_mark_price.filter(regex=f'{"|".join(straddle_ls)}').isnull())
    rv = trader_obj.reader_obj.rv_read(symbol='ETHUSDT')
    returns = trader_obj.reader_obj.returns_read(symbol='ETHUSDT').replace(0, np.nan)
    returns = returns.fillna(returns.ewm(span=12, min_periods=1).mean())
    print(args)
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(lambda x: x, x=row) for _, row in qlike.iterrows()]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            L = row['L']
            model, regression, training_scheme, trading_session, top_book, h = row['model'], row['regression'], \
                row['training_scheme'], row['trading_session'], row['top_book'], row['h']
            y_hat = db_obj.forecast_query(L=L, model=model, regression=regression, trading_session=trading_session,
                                          top_book=top_book, training_scheme=training_scheme)[['y_hat', 'symbol']]
            perf = db_obj.forecast_performance(L=L, model=model, regression=regression, training_scheme=training_scheme,
                                               trading_session=trading_session, top_book=top_book)
            y_hat = y_hat.query('symbol == "ETHUSDT"')
            y_hat = pd.pivot(data=y_hat, columns='symbol', values='y_hat')
            y_hat.index = pd.to_datetime(y_hat.index)
            y_hat = y_hat.resample(h).sum()
            y_hat = y_hat.rename(columns={'ETHUSDT': 'sig'})
            rv_models[(r'$\mathcal{M}$', L.lower())] = y_hat
            qlike_models[(r'$\mathcal{M}$', L.lower())] = \
                perf.query('symbol == "ETHUSDT"')[['values']].rename(columns={'values': (r'$\mathcal{M}$', L.lower())})
            qlike_models[(r'$\mathcal{M}$', L.lower())] = \
                qlike_models[(r'$\mathcal{M}$', L.lower())].loc[~qlike_models[(r'$\mathcal{M}$',
                                                                               L.lower())].index.duplicated()]
            dates = list(set(np.unique(y_hat.index.date).tolist()))
            dates.sort()
            ############################################################################################################
            ### Fitting GARCH(1,1)
            ############################################################################################################
            GARCH = pd.DataFrame(index=pd.date_range(start=returns.index[0].date(),
                                                     end=returns.index[-1].date() + relativedelta(days=1),
                                                     freq=h, inclusive='left'), columns=['sig'], data=np.nan)
            GARCH.index = pd.to_datetime(GARCH.index, utc=True)
            with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
                futures = [executor.submit(trader_obj.fit_garch, date=date, returns=returns.loc[y_hat.index],
                                           L_train=L_train, L=L, GARCH=GARCH) for date in dates]
            rv_GARCH[('GARCH', L.lower())] = GARCH.loc[rv_models[(r'$\mathcal{M}$', L.lower())].index]
            qlike_models[('GARCH', L.lower())] = \
                rv.resample(h).sum().join(rv_GARCH[('GARCH', L.lower())],
                                          how='right').resample('1W')[['ETHUSDT', 'sig']].apply(qlike_score)
            qlike_models[('GARCH', L.lower())] = pd.DataFrame(data=qlike_models[('GARCH', L.lower())].values,
                                                              index=qlike_models[('GARCH', L.lower())].index,
                                                              columns=[('GARCH', L.lower())])
    perf_qlike = pd.concat(list(qlike_models.values()), axis=1)
    perf_qlike.columns = pd.MultiIndex.from_tuples(perf_qlike.sort_index(axis=1).columns)
    ########################################################################################################
    ### Signals
    ########################################################################################################
    signals = \
        pd.concat(rv_models, axis=1).droplevel(axis=1, level=2).join(pd.concat(rv_GARCH, axis=1).droplevel(axis=1,
                                                                                                           level=2))**.5
    first_valid_date_per_L = signals.apply(lambda x: x.first_valid_index()).droplevel(axis=0, level=0)
    first_valid_date_per_L = first_valid_date_per_L.apply(lambda x: x - relativedelta(days=1))
    # perf_qlike = perf_qlike.reindex(signals.index).ffill()
    # threshold_qlike = \
    #     perf_qlike.resample('1M').apply(lambda x: x.quantile(args.signal_strength)).reindex(perf_qlike.index).ffill()
    # threshold_qlike = threshold_qlike.reindex(signals.index).ffill()
    # signals_strength = perf_qlike.lt(threshold_qlike)
    # signals = signals.mul(signals_strength.astype(float).reindex(signals.index).ffill())
    signals = signals.diff()
    # Large RV swings
    lower_bound = signals.resample('1D').apply(lambda x: x.quantile(.25)).shift()
    upper_bound = signals.resample('1D').apply(lambda x: x.quantile(.75)).shift()
    iqr = upper_bound.sub(lower_bound)
    lower_bound = lower_bound.sub(iqr.mul(1.5)).reindex(signals.index).ffill()
    upper_bound = upper_bound.add(iqr.mul(1.5)).reindex(signals.index).ffill()
    signals = -signals.lt(lower_bound).astype(float)+signals.gt(upper_bound).astype(float)
    # signals = signals.mul(signals_strength)
    tx_fees = \
        trader_obj.reader_obj.prices_read('ETHUSDT').resample(args.h).last().mul(trader_obj.fees).iloc[:, 0].mul(
            straddles.sum(axis=1))
    tx_fees = signals.apply(lambda x, y: np.abs(x.mul(y)), y=tx_fees).fillna(0).resample('1D').sum()
    PnL_per_strat = signals.apply(lambda x, y: x.mul(y), y=PnL.diff().sum(axis=1)).resample('1D').sum()
    # PnL_per_strat = PnL_per_strat.sub(tx_fees)
    # PnL_per_strat.resample('1D').sum().cumsum().plot()
    # import matplotlib.pyplot as plt
    # plt.show()
    # pdb.set_trace()
    # signals = np.sign(signals.diff()).fillna(0).astype(int)
    # signals = signals.reindex(index).shift()
    # PnL = PnL.sub((np.abs(PnL).mul(straddles.sum(axis=1).mul(trader_obj.fees))))
    # PnL_per_strat = signals.apply(lambda x, y: x*y, y=PnL).resample('1D') \
        # if args.strategy == 'options' else signals.apply(lambda x, y: x*y, y=PnL['ETHUSDT']).resample('1D')
    # PnL_per_strat = PnL_per_strat.sum().replace(0, np.nan)#.fillna(0)
    Tt = np.abs(signals).apply(lambda x, y: x*y.sum(axis=1), y=straddles).resample('1D').sum()
    PPT_per_strat = PnL_per_strat.div(Tt.replace(0, np.nan)).mean()
    # PPT_per_strat = (1e4*PPT_per_strat).round(4)
    ################################################################################################################
    ### Backtesting
    ################################################################################################################
    PnL_per_strat = PnL_per_strat.cumsum()
    # PnL_per_strat = np.log(PnL_per_strat.div(PnL_per_strat.shift())).replace(np.inf, np.nan).ffill()
    sharpeRatio = PnL_per_strat.mean().div(PnL_per_strat.std()).mul(np.sqrt(360)).reset_index().rename(
        columns={'level_0': 'Strategy', 'level_1': r'$L_{train}$', 0: 'Sharpe ratio'})
    sharpeRatio[r'$L_{train}$'] = [f'${L.lower()}$' for L in sharpeRatio[r'$L_{train}$']]
    sharpeRatio['Strategy'] = [model.upper().replace('_', ' ') if model in ['GARCH']
                               else model for model in sharpeRatio['Strategy']]
    print(f'Option universe size: {straddles.shape[1]}.\n')
    print('---------Sharpe ratio table-------------')
    print(sharpeRatio)
    # PnL_per_strat = PnL_per_strat.fillna(0).cumsum()
    # PnL_per_strat = np.log(PnL_per_strat.div(PnL_per_strat.shift())).fillna(0)
    # pdb.set_trace()
    print('---------******************-------------\n')
    print('---------P&L per strategy table-------------')
    print(PnL_per_strat)
    print('---------**********************-------------\n')
    print('---------*********-------------\n')
    print('---------PPT table-------------')
    # PPT_per_strat = PPT_per_strat.apply(lambda x: f'{x} bps')
    PPT_per_strat = PPT_per_strat.reset_index().set_index(['level_0', 'level_1']).transpose()
    PPT_per_strat.columns.names = [None, None]
    PPT_per_strat = PPT_per_strat.loc[:, [('GARCH', '1w'), ('GARCH', '1m'), ('GARCH', '6m'),
                                          (r'$\mathcal{M}$', '1w'), (r'$\mathcal{M}$', '1m'),
                                          (r'$\mathcal{M}$', '6m')]]
    print(PPT_per_strat)
    print('---------*********-------------\n')
    first_valid_date_per_L = first_valid_date_per_L.loc[first_valid_date_per_L.index.duplicated()]
    # PnL_per_strat = PnL_per_strat.ffill()
    PnL_per_strat = PnL_per_strat.unstack().reset_index().groupby(
        'level_1').apply(lambda x, y: x.query(f'level_2 >= "{y[x.level_1.unique()[0]]}"'), y=first_valid_date_per_L)
    PnL_per_strat = PnL_per_strat.droplevel(axis=0, level=0).set_index('level_2')
    PnL_per_strat = PnL_per_strat.reset_index().rename(columns={'level_0': 'Strategy', 'level_1': r'$L_{train}$',
                                                                0: 'PnL', 'level_2': 'timestamp'})
    # numberLongShort = dict()
    # numberTrades = dict()
    # for i in range(0, signals.shape[1]):
    #     numberLongShort[signals.columns[i]] = \
    #         straddles.apply(lambda x, y: x * y, y=signals.iloc[:, i]).fillna(0).apply(
    #             lambda x: x.value_counts()).drop(0.0).sum(axis=1)
    #     numberTrades[signals.columns[i]] = \
    #         straddles.apply(lambda x, y: x * y, y=signals.iloc[:, i]).fillna(0).sum(axis=1)
    # numberLongShort = \
    #     pd.concat(numberLongShort, axis=1).fillna(0).astype(int).loc[:, [('GARCH', '1w'), ('GARCH', '1m'),
    #                                                                      ('GARCH', '6m'), (r'$\mathcal{M}$', '1w'),
    #                                                                      (r'$\mathcal{M}$', '1m'),
    #                                                                      (r'$\mathcal{M}$', '6m')]]
    # numberTrades = pd.concat(numberTrades, axis=1).loc[:, [('GARCH', '1w'), ('GARCH', '1m'),
    #                                                        ('GARCH', '6m'), (r'$\mathcal{M}$', '1w'),
    #                                                        (r'$\mathcal{M}$', '1m'), (r'$\mathcal{M}$', '6m')]]
    # numberTrades = numberTrades.resample('1D').sum().cumsum().unstack().reset_index()
    # numberTrades = \
    #     numberTrades.groupby('level_1').apply(lambda x, y: x.query(f'level_2 >= "{y[x.level_1.unique()[0]]}"'),
    #                                           y=first_valid_date_per_L)
    # numberTrades = numberTrades.droplevel(axis=0, level=0)
    # numberTrades = numberTrades.rename(columns={'level_0': 'Strategy', 'level_1': r'$L_{train}$',
    #                                             'level_2': 'timestamp', 0: 'Number of trades'})
    # print('---------Number of trade table-------------')
    # print(numberLongShort)
    # print('---------*********************-------------\n')
    ###################################################################################################################
    # Figure 1: Sharpe ratio
    ###################################################################################################################
    fig = px.bar(sharpeRatio, x=r'$L_{train}$', y='Sharpe ratio', color='Strategy', barmode='group',
                 title='Annualized Sharpe ratio', text_auto='.2f',
                 category_orders={r'$L_{train}$': [r'$1w$', r'$1m$', r'$6m$'],
                                  'Strategy': ['GARCH', r'$\mathcal{M}$']})
    fig.update_layout(width=1_200, height=700, font=dict(size=LABEL_AXIS_FONT_SIZE),
                      title=dict(font=dict(size=TITLE_FONT_SIZE)),
                      legend=dict(title=None, orientation='h', xanchor='right', yanchor='bottom', y=1, x=1))
    fig.update_annotations(font=dict(size=TITLE_FONT_SIZE), x=1)
    ####################################################################################################################
    ## Figure 2: PnL per strat
    ####################################################################################################################
    fig2 = make_subplots(rows=PnL_per_strat[r'$L_{train}$'].unique().shape[0], cols=1,
                         shared_xaxes=False, shared_yaxes=False, row_titles=[r'$1w$', r'$1m$', r'$6m$'],
                         vertical_spacing=.5/PnL_per_strat[r'$L_{train}$'].unique().shape[0])
    tmp_fig = px.line(PnL_per_strat, x='timestamp', y='PnL', color='Strategy', facet_row=r'$L_{train}$',
                      category_orders={r'$L_{train}$': [r'$1w$', r'$1m$', r'$6m$'],
                                       'Strategy': ['GARCH', r'$\mathcal{M}$']}, facet_row_spacing=.1)
    for i in range(0, len(tmp_fig.data), 3):
        fig2.add_trace(trace=tmp_fig.data[i], col=1, row=1)
        fig2.add_trace(trace=tmp_fig.data[i + 1], col=1, row=2)
        fig2.add_trace(trace=tmp_fig.data[i + 2], col=1, row=3)
    fig2.update_xaxes(title_text='Date', tickangle=45, row=1)
    fig2.update_xaxes(title_text='Date', tickangle=45, row=2)
    fig2.update_xaxes(title_text='Date', tickangle=45, row=3)
    fig2.update_layout(title=dict(text=f'Cumulative P&L curves', font=dict(size=TITLE_FONT_SIZE)),
                       width=1_200, height=1_200, font=dict(size=LABEL_AXIS_FONT_SIZE))
    fig2.update_annotations(font=dict(size=TITLE_FONT_SIZE), x=1)
    ####################################################################################################################
    ## Figure 3: Number of trades per strat
    ####################################################################################################################
    # fig3 = make_subplots(rows=numberTrades[r'$L_{train}$'].unique().shape[0], cols=1,
    #                      shared_xaxes=False, shared_yaxes=False, row_titles=[r'$1w$', r'$1m$', r'$6m$'],
    #                      vertical_spacing=.5 / numberTrades[r'$L_{train}$'].unique().shape[0])
    # tmp_fig = px.line(numberTrades, x='timestamp', y='Number of trades', color='Strategy', facet_row=r'$L_{train}$',
    #                   category_orders={r'$L_{train}$': [r'$1w$', r'$1m$', r'$6m$'],
    #                                    'Strategy': ['GARCH', r'$\mathcal{M}$']}, facet_row_spacing=.1)
    # for i in range(0, len(tmp_fig.data), 3):
    #     fig3.add_trace(trace=tmp_fig.data[i], col=1, row=1)
    #     fig3.add_trace(trace=tmp_fig.data[i + 1], col=1, row=2)
    #     fig3.add_trace(trace=tmp_fig.data[i + 2], col=1, row=3)
    # fig3.update_xaxes(title_text='Date', tickangle=45, row=1)
    # fig3.update_xaxes(title_text='Date', tickangle=45, row=2)
    # fig3.update_xaxes(title_text='Date', tickangle=45, row=3)
    # fig3.update_layout(title=dict(text=f'Number of trades', font=dict(size=40)),
    #                    width=1_200, height=1_200, font=dict(size=20))
    # fig3.update_annotations(font=dict(size=40), x=1)
    ####################################################################################################################
    ## Figure 4: Aggregated QLIKE
    ####################################################################################################################
    perf_qlike = perf_qlike.unstack().reset_index().groupby(
        'level_1').apply(lambda x, y: x.query(f'timestamp >= "{y[x.level_1.unique()[0]]}"'), y=first_valid_date_per_L)
    perf_qlike = perf_qlike.droplevel(axis=0, level=0).set_index('level_1')
    perf_qlike = perf_qlike.reset_index().rename(columns={'level_0': 'Strategy', 'level_1': r'$L_{train}$', 0: 'QLIKE'})
    fig4 = make_subplots(rows=perf_qlike[r'$L_{train}$'].unique().shape[0], cols=1,
                         shared_xaxes=False, shared_yaxes=False, row_titles=[r'$1w$', r'$1m$', r'$6m$'],
                         vertical_spacing=.5 / PnL_per_strat[r'$L_{train}$'].unique().shape[0])
    tmp_fig = px.line(perf_qlike, x='timestamp', y='QLIKE', color='Strategy', facet_row=r'$L_{train}$',
                      category_orders={r'$L_{train}$': [r'1w', r'1m', r'6m'],
                                       'Strategy': ['GARCH', r'$\mathcal{M}$']}, facet_row_spacing=.1)
    for i in range(0, len(tmp_fig.data), 3):
        fig4.add_trace(trace=tmp_fig.data[i], col=1, row=1)
        fig4.add_trace(trace=tmp_fig.data[i + 1], col=1, row=2)
        fig4.add_trace(trace=tmp_fig.data[i + 2], col=1, row=3)
    fig4.update_xaxes(title_text='Date', tickangle=45, row=1)
    fig4.update_xaxes(title_text='Date', tickangle=45, row=2)
    fig4.update_xaxes(title_text='Date', tickangle=45, row=3)
    fig4.update_layout(title=dict(text=f'Aggregated QLIKE', font=dict(size=TITLE_FONT_SIZE)),
                       width=1_200, height=1_200, font=dict(size=LABEL_AXIS_FONT_SIZE))
    fig4.update_annotations(font=dict(size=TITLE_FONT_SIZE), x=1)
    if args.save == 1:
        fig.write_image(os.path.abspath(f'../figures/sharpe_ratio_{args.performance}.pdf'), engine='kaleido')
        print(f'[Figures]: Sharpe ratio has been saved.')
        fig2.write_image(os.path.abspath(f'../figures/pnl_{args.performance}.pdf'), engine='kaleido')
        print(f'[Figures]: PnL has been saved.')
        # fig3.write_image(os.path.abspath(f'../figures/number_trades_{args.performance}.pdf'), engine='kaleido')
        # print(f'[Figures]: Number of trades has been saved.')
        fig4.write_image(os.path.abspath(f'../figures/qlike_perf_{args.performance}.pdf'), engine='kaleido')
        print(f'[Figures]: Aggregate QLIKE has been saved.')
    else:
        fig.show()
        fig2.show()
        # fig3.show()
        fig4.show()
    print('---------PPT table to LaTex-------------')
    print(PPT_per_strat.to_latex())
    print('---------************************-----\n')
    # print('---------Number of trade table to LaTex-------------')
    # print(numberLongShort.to_latex())
    # print('---------****************************-------------\n')
    pdb.set_trace()
