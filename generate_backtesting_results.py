import os.path
import pandas as pd
from data_centre.data import Reader
from scipy.stats import skew, kurtosis
import plotly.express as px
from trading.strategies import Trader
import argparse
import vectorbt as vbt
import pdb

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trading strategies: Backtesting.')
    parser.add_argument('--L', default='ALL', help='Lookback windows over which models are trained.',
                        type=str)
    args = parser.parse_args()
    reader_obj = Reader()
    returns = reader_obj.returns_read()
    """Selection of top 5 performing models using QLIKE"""
    performance = pd.read_csv('../results/performance.csv')
    performance = performance.loc[~performance.regression.isin(['ridge', 'risk_metrics']), :]
    performance = pd.pivot(data=performance,
                           columns=['metric'],
                           values='values',
                           index=['vol_regime', 'training_scheme', 'L', 'regression', 'model'])[['qlike']]
    if args.L != 'ALL':
        performance = performance.query(f'L == \"{args.L}\"')
    n_performers = 5
    top_performers = performance.sort_values(by='qlike').iloc[:n_performers].index
    bottom_performers = performance.sort_values(by='qlike').iloc[-n_performers:].index
    trader_obj = Trader(top_performers=top_performers, bottom_performers=bottom_performers, hp=1)
    trader_obj.PnL()
    cumPnL = pd.concat(trader_obj.PnL_dd, axis=1).dropna()
    vwap = trader_obj._prices.mul(trader_obj._volumes).div(trader_obj._volumes).dropna(axis=1, how='all')
    entries = pd.DataFrame(index=vwap.index, columns=vwap.columns.tolist(), data=0)
    entries.iloc[0, :] = 1
    pf_vwap = vbt.Portfolio.from_signals(vwap, entries=entries, init_cash=10000, size=1, freq=pd.to_timedelta('30T'))
    pf_vwap.stats()
    pdb.set_trace()
    vwap = pf_vwap.returns().sum(axis=1).add(1).cumprod()
    vwap.name = 'VWAP'
    cumPnL = pd.concat([cumPnL, vwap.loc[cumPnL.index]], axis=1)
    cumPnL = cumPnL.div(cumPnL.iloc[0, :])-1
    PnL_per_day = cumPnL.diff().resample('1D').mean().mul(10_000)
    cumPnL = pd.melt(cumPnL, ignore_index=False, var_name='model/strategy', value_name='values')
    fig = px.line(data_frame=cumPnL, y='values', color='model/strategy', title='Trading performance: Cumulative PnL')
    fig.add_hline(y=0, line_width=1, line_dash='dash', line_color='black')
    fig.update_xaxes(tickangle=45, title='Date')
    fig.update_yaxes(title='Cumulative PnL')
    fig.show()
    pdb.set_trace()
    fig.write_image(os.path.abspath(f'../figures/cum_PnL.pdf'))
    fig.show()