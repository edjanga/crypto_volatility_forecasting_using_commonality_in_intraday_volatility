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
WIDTH = 1_500
HEIGHT = 800


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
        rv_hat = rv_hat.reshape(rv_hat.shape[0], 1)
        GARCH.loc[date.strftime('%Y-%m-%d'):date.strftime('%Y-%m-%d')] = rv_hat
        print(f'[End of Training]: GARCH model^{L} training on {date} has been completed.')

    def filtering_rule(self, signals: pd.DataFrame, rv: pd.DataFrame, threshold: float = .1, **kwargs) -> pd.DataFrame:
        num = signals.apply(lambda x, y: x.subtract(y), y=rv['ETHUSDT'].loc[signals.index].shift())
        tmp = kwargs['rv_mean'].reindex_like(signals)
        denom = tmp.apply(lambda x, y: y[x.name[-1]], y=kwargs['rv_mean'])
        denom = denom.shift().reindex(num.index).ffill()
        filter = num.div(denom).gt(threshold) | num.div(denom).lt(-threshold)
        return filter


class OptionTrader(Trader):

    def __init__(self, fees: float):
        Trader.__init__(self, fees)

    def process_mkt_data(self, data: pd.DataFrame, option_type: str = 'call') -> pd.DataFrame:
        mkt_data = data.loc[data.index.get_level_values(1) == f'{option_type}', :]
        return mkt_data

    def pivot(self, mkt_data: pd.DataFrame, h: str) -> Tuple[pd.DataFrame]:
        ls = list()
        index = pd.to_datetime(pd.date_range(start='2021-01-01', end='2023-07-01', freq=h, inclusive='left'),
                               utc=True)
        for col in ['strike_price', 'mark_price', 'underlying_price', 'expiration']:
            try:
                tmp = pd.pivot_table(mkt_data.reset_index(), index='timestamp', columns='option_tag', values=col)
                tmp = tmp.reindex(index)
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
    parser.add_argument('--title_figure', default=0, type=int, help='Title figures.')
    parser.add_argument('--liquidity', default=.75, type=float, help='Threshold of liquid to keep for backtesting.')
    parser.add_argument('--min_expiration', type=str, help='Minimum expiration to filter option chain data.',
                        default='7D')
    parser.add_argument('--max_expiration', type=str, help='Maximum expiration to filter option chain data.',
                        default='180D')
    parser.add_argument('--h', type=str, help='Time interval.', default='30T')
    parser.add_argument('--save', type=int, help='Save or show figures.', default=0)
    parser.add_argument('--signal_strength', type=float, help='Threshold used to assess how certain signals are.',
                        default=.05)
    parser.add_argument('--best_performing', type=int, help='Amount of models to backtest.',
                        default=1)
    args = parser.parse_args()
    print(args)
    title_figure = bool(args.title_figure)
    index = pd.to_datetime(pd.date_range(start='2021-01-01', end='2023-07-01', freq=args.h, inclusive='left'),
                           utc=True)
    db_obj = DBQuery()
    qlike = db_obj.query_data(db_obj.best_model_for_all_windows_query(), table='y')
    if args.best_performing == 1:
        qlike = pd.DataFrame(qlike.loc[qlike['values'].idxmin(), :]).transpose()
    else:
        qlike = qlike.drop(axis=0, index=qlike['values'].idxmin())
    suffix_name = \
        {'trading_session': {True: 'eq', False: 'eq_vixm'}, 'top_book': {True: 'top_book', False: 'full_book'}}
    PnL_per_strat = dict()
    trades_per_strat = dict()
    rv_models = dict()
    rv_mean = dict()
    qlike_models = dict()
    rv_GARCH = dict()
    qlike_GARCH = dict()
    L_train = {'1W': 7, '1M': 30, '6M': 180}
    trader_obj = OptionTrader(fees=.0003)
    option_chain_data_obj = OptionChainData()
    files = [file for file in glob.glob('../data_centre/tmp/datasets/*-*-*')]
    data = pd.concat([pd.read_parquet(option_chain) for option_chain in files])
    data = data.loc[:, ~data.columns.isin(['delta', 'gamma', 'vega', 'theta', 'rho'])]
    data = data.set_index('type', append=True).swaplevel(i=1, j=-1)
    data.loc[:, 'time2expiration'] = (data.expiration - data.index.get_level_values(-1)) / pd.to_timedelta('360D')
    data = data.query(f'time2expiration >= {pd.to_timedelta(args.min_expiration) / pd.to_timedelta("360D")} &'
                      f'time2expiration <= {pd.to_timedelta(args.max_expiration) / pd.to_timedelta("360D")}')
    calls, puts = trader_obj.process_mkt_data(data=data), trader_obj.process_mkt_data(data=data, option_type='put')
    calls_strike_price, calls_mark_price, calls_underlying_price, _ = trader_obj.pivot(calls, args.h)
    puts_strike_price, puts_mark_price, puts_underlying_price, _ = trader_obj.pivot(puts, args.h)
    straddle_ls = list(set(calls_mark_price.columns).intersection(set(puts_mark_price.columns)))
    eth = trader_obj.reader_obj.prices_read(symbol='ETHUSDT').resample(args.h).last().iloc[:, 0]
    rv = trader_obj.reader_obj.rv_read(symbol='ETHUSDT')
    returns = trader_obj.reader_obj.returns_read(symbol='ETHUSDT').replace(0, np.nan)
    returns = returns.fillna(returns.ewm(span=12, min_periods=1).mean())
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(lambda x: x, x=row) for _, row in qlike.iterrows()]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            L = row['L']
            model, regression, training_scheme, trading_session, top_book, h = row['model'], row['regression'], \
                row['training_scheme'], row['trading_session'], row['top_book'], row['h']
            y_hat = db_obj.forecast_query(L=L, model=model, regression=regression, trading_session=trading_session,
                                          top_book=top_book, training_scheme=training_scheme,
                                          symbol='ETHUSDT')[['y_hat']].drop_duplicates(keep='first')
            y_hat.columns = ['sig']
            rv_mean[L.lower()] = \
                rv.resample(f'{L_train[L]}D').mean().shift().reindex(y_hat.index).ffill().drop_duplicates(keep='first')
            y_hat.index = pd.to_datetime(y_hat.index)
            y_hat = y_hat.resample(h).sum()
            rv_models[(r'$\textit{M}$', L.lower())] = y_hat
            dates = list(set(np.unique(y_hat.index.date).tolist()))
            dates.sort()
            ############################################################################################################
            ### Fitting GARCH(1,1)
            ############################################################################################################
            GARCH = pd.DataFrame(index=pd.date_range(start=returns.index[0].date(),
                                                     end=returns.index[-1].date() + relativedelta(days=1),
                                                     freq=h, inclusive='left'), columns=['sig'], data=np.nan)
            GARCH.index = pd.to_datetime(GARCH.index, utc=True)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(trader_obj.fit_garch, date=date, returns=returns.loc[y_hat.index],
                                           L_train=L_train, L=L, GARCH=GARCH) for date in dates]
            rv_GARCH[('GARCH', L.lower())] = GARCH.loc[rv_models[(r'$\textit{M}$', L.lower())].index]
    rv_mean = pd.concat(rv_mean, axis=1).droplevel(axis=1, level=1).ffill().sort_index()
    rv_mean.index = pd.to_datetime(rv_mean.index, utc=True)
    ########################################################################################################
    ### Signals
    ########################################################################################################
    signals = \
        pd.concat(rv_models, axis=1).droplevel(axis=1, level=2).join(pd.concat(rv_GARCH, axis=1).droplevel(axis=1,
                                                                                                           level=2))
    filter = trader_obj.filtering_rule(signals=signals, rv_mean=rv_mean, rv=rv.resample(args.h).sum(),
                                       threshold=args.signal_strength)
    signals = np.sign(signals.apply(lambda x, y: x.subtract(y.loc[x.index].shift()), y=rv['ETHUSDT']))
    signals = signals.mul(filter)
    number_of_trades = signals.apply(lambda x: x.value_counts()).drop(0.0).fillna(0)
    number_of_trades.index = ['Short', 'Long']
    number_of_trades = pd.DataFrame(number_of_trades.unstack()).reset_index()
    number_of_trades.columns = ['Strategy', '$L_{train}$', 'Side', 'Trades']
    #########################################################################################################
    ### PnL
    #########################################################################################################
    calls_mark_price = calls_mark_price.filter(regex=f'{"|".join(straddle_ls)}')
    puts_mark_price = puts_mark_price.filter(regex=f'{"|".join(straddle_ls)}')
    straddle_mark_price = calls_mark_price.add(puts_mark_price)
    buy = \
        straddle_mark_price.apply(lambda x, y: x.subtract(y),
                                  y=eth.mul(trader_obj.fees).mul(2)).subtract(
            straddle_mark_price.shift().apply(
                lambda x, y: x.subtract(y),
                y=eth.shift().mul(trader_obj.fees).mul(2))).div(straddle_mark_price.shift()).mean(axis=1)
    short = \
        straddle_mark_price.shift().apply(lambda x, y: x.subtract(y),
                                          y=eth.shift().mul(trader_obj.fees).mul(2)).subtract(
            straddle_mark_price.apply(lambda x, y: x.subtract(y),
                                      y=eth.mul(trader_obj.fees).mul(2))).div(straddle_mark_price.shift()).mean(axis=1)
    PnL_per_strat = \
        (signals > 0).apply(lambda x, y: x.mul(y), y=buy).add((signals < 0).apply(lambda x, y: x.mul(y),
                                                                                  y=short)).resample('1D').sum()
    PnL_per_strat = PnL_per_strat.replace(0, np.nan)
    straddles = (~calls_mark_price.filter(regex=f'{"|".join(straddle_ls)}').isnull()) & \
                (~puts_mark_price.filter(regex=f'{"|".join(straddle_ls)}').isnull())
    first_valid_date_per_L = signals.apply(lambda x: x.first_valid_index()).droplevel(axis=0, level=0)
    first_valid_date_per_L = first_valid_date_per_L.apply(lambda x: x - relativedelta(days=1))
    ################################################################################################################
    ### Backtesting
    ################################################################################################################
    sharpeRatio = PnL_per_strat.mean().div(PnL_per_strat.std()).mul(np.sqrt(360)).reset_index().rename(
        columns={'level_0': 'Strategy', 'level_1': r'$L_{train}$', 0: 'Sharpe ratio'})
    sharpeRatio['$L_{train}$'] = sharpeRatio['$L_{train}$'].apply(lambda x: f'${x}$')
    sharpeRatio['Strategy'] = [model.upper().replace('_', ' ') if model in ['GARCH']
                               else model for model in sharpeRatio['Strategy']]
    print(f'Option universe size: {straddles.shape[1]}.\n')
    print('---------Sharpe ratio table-------------')
    print(sharpeRatio)
    PnL_per_strat = PnL_per_strat.fillna(0).cumsum()
    print('---------******************-------------\n')
    print('---------P&L per strategy table-------------')
    print(PnL_per_strat)
    print('---------**********************-------------\n')
    first_valid_date_per_L = first_valid_date_per_L.loc[first_valid_date_per_L.index.duplicated()]
    PnL_per_strat = PnL_per_strat.unstack().reset_index().groupby(
        'level_1').apply(lambda x, y: x.query(f'level_2 >= "{y[x.level_1.unique()[0]]}"'), y=first_valid_date_per_L)
    PnL_per_strat = PnL_per_strat.droplevel(axis=0, level=0).set_index('level_2')
    PnL_per_strat = \
        PnL_per_strat.reset_index().rename(columns={'level_0': 'Strategy', 'level_1': r'$L_{train}$',
                                                    'level_2': 'timestamp', 0: 'PnL'})
    for idx, L_train in enumerate(sharpeRatio['$L_{train}$'].unique().tolist()):
        tmp = sharpeRatio.loc[sharpeRatio['$L_{train}$'] == L_train, :]
        ghost = pd.concat([tmp.copy()]*10)
        ghost.loc[:, 'Sharpe ratio'] = 0
        ghost.loc[:, 'Strategy'] = [f'ghost{i}' for i in range(1, 21)]
        tmp = pd.concat([tmp, ghost])
        fig = px.bar(tmp, x='Sharpe ratio', y=r'$L_{train}$', color='Strategy', barmode='group',
                     title={True: f'Annualized Sharpe ratio - <i>{L_train.replace("$", "")}</i>',
                            False: ''}[title_figure],
                     text_auto='.2f', category_orders={'Strategy': ['GARCH', r'$\textit{M}$']},
                     orientation='h')
        fig.data = [plot.update({'showlegend': False}) if 'ghost' in plot.name else plot for plot in fig.data]
        fig.update_yaxes(title=dict(text=''))
        fig.update_traces(width=.3)
        fig.update_layout(yaxis=dict(domain=[.0, .25]))
        fig.data = list(map(lambda x: x.update({'y': ['']}), fig.data))
        fig.update_layout(width=1_000, height=600, font=dict(size=LABEL_AXIS_FONT_SIZE),
                          title=dict(font=dict(size=TITLE_FONT_SIZE)),
                          legend=dict(title=None, orientation='v', xanchor='right', yanchor='bottom', y=.1, x=1.1),
                          xaxis_title_font=dict(size=LABEL_AXIS_FONT_SIZE))
        fig.show()
    ####################################################################################################################
    ## Figure 2: PnL per strat
    ####################################################################################################################
    for idx, L_train in enumerate(PnL_per_strat['$L_{train}$'].unique().tolist()):
        tmp = PnL_per_strat.loc[PnL_per_strat['$L_{train}$'] == L_train, :]
        fig2 = px.line(tmp, x='timestamp', y='PnL', color='Strategy',
                       category_orders={'Strategy': ['GARCH', r'$\textit{M}$']})
        fig2.add_hline(0)
        fig2.update_xaxes(title_text='Date', tickangle=45)
        fig2.update_yaxes(title_text='returns')
        fig2.update_layout(title=dict(text={True: f'Cumulative P&L curves - <i>{L_train}</i>', False: ''}[title_figure],
                                      font=dict(size=TITLE_FONT_SIZE)),
                           width=1_000, height=600, font=dict(size=LABEL_AXIS_FONT_SIZE),
                           legend=dict(title=None, orientation='v', xanchor='right', yanchor='bottom', y=1, x=1))
        fig2.update_annotations(font=dict(size=LABEL_AXIS_FONT_SIZE), x=1)
        fig2.show()
    ####################################################################################################################
    ## Figure 3: Long/short trades
    ####################################################################################################################
    for idx, L_train in enumerate(number_of_trades['$L_{train}$'].unique().tolist()):
        tmp = number_of_trades.loc[number_of_trades['$L_{train}$'] == L_train, :]
        ghost = pd.concat([tmp.copy()] * 2)
        ghost.loc[:, 'Trades'] = 0
        ghost.loc[:, 'Strategy'] = [f'ghost{i}' for i in range(1, 5)]+[f'ghost{i}' for i in range(4, 0, -1)]
        tmp = pd.concat([tmp, ghost])
        fig3 = px.bar(tmp, x='Trades', y='Side', color='Strategy',
                      barmode='group',
                      title={True: f'Number of trades per side - <i>{L_train.replace("$", "")}</i>',
                             False: ''}[title_figure],
                      category_orders={'Strategy': ['ghost1', 'ghost2', 'GARCH', r'$\textit{M}$', 'ghost3', 'ghost4'],
                                       'Side': ['Long', 'Short']},
                      color_discrete_sequence=px.colors.qualitative.Plotly[2:4] +
                                              px.colors.qualitative.Plotly[:2] + px.colors.qualitative.Plotly[5:7])
        fig3.data = [plot.update({'showlegend': False}) if 'ghost' in plot.name else plot for plot in fig3.data]
        fig3.update_layout(yaxis=dict(domain=[.0, .25]))
        fig3.update_yaxes(title=dict(text=''))
        fig3.update_traces(width=.15)
        fig3.update_layout(width=1_000, height=600, font=dict(size=LABEL_AXIS_FONT_SIZE),
                           title=dict(font=dict(size=TITLE_FONT_SIZE)),
                           legend=dict(title=None, orientation='v', xanchor='right', yanchor='bottom', y=.1, x=1.1),
                           xaxis_title_font=dict(size=LABEL_AXIS_FONT_SIZE))
        fig3.show()