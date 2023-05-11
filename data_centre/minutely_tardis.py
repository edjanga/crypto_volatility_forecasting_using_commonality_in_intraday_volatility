import time
import pandas
import os
import glob
import pdb
import argparse
import concurrent.futures
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import subprocess
from helpers import coin_ls, exchange_ls
import re

missing_data_ls = list()
something_wrong_data_ls = list()
columns_names_ls = ['symbol', 'timeIndex', 'timeHMs', 'timeHMe', 'volBuyQty', 'volSellQty', 'volBuyNotional',
                    'volSellNotional', 'nrTrades', 'volBuyQty_lit', 'volBuyQty_hid', 'volSellQty_lit', 'volSellQty_hid',
                    'volBuyNotional_lit', 'volBuyNotional_hid', 'volSellNotional_lit', 'volSellNotional_hid',
                    'volBuyNrTrades_lit', 'volSellNrTrades_hid', 'volSellNrTrades_lit',
                    'volSellNrTrades_hid', 'pret_1m', 'bidPx', 'askPx', 'bidQty', 'askQty']


def generate_data_per_tag(tag: str) -> None:
    print(tag)
    year = tag.split('_')[-1].split('-')[0]
    data_type_ls = ['book_snapshot_5', 'trades']
    print(f'[PROCESS TO GENERATE MINUTELY TARDIS]: {tag} has started.')
    data_dd = \
        {data_type: pd.read_csv(filepath_or_buffer=
                                glob.glob(f'{args.location}/rawTARDIS/{year}/*/{data_type}/{tag}.csv.gz')[0],
                                index_col='timestamp',
                                date_parser=lambda x: datetime.fromtimestamp(int(x)/1_000_000))
       for _, data_type in enumerate(data_type_ls)}

    exchange_tag = data_dd['book_snapshot_5'].exchange.unique()[0]
    """
        Clean book_snapshot_5
    """
    book_snapshot_5 = data_dd['book_snapshot_5'].drop(['exchange', 'local_timestamp'], axis=1)
    book_snapshot_5 = book_snapshot_5.filter(regex=f'symbol|(bids|asks)\[[0]\]')
    book_snapshot_5 = book_snapshot_5.resample('T').last()

    def sidePxQty(df: pd.DataFrame, side: str) -> None:
        df[f'{side}Px'] = df[f'{side}s[0].price']
        df[f'{side}Qty'] = df[f'{side}s[0].amount']
        df.drop([f'{side}s[0].price', f'{side}s[0].amount'], axis=1, inplace=True)
    sidePxQty(book_snapshot_5, 'bid')
    sidePxQty(book_snapshot_5, 'ask')
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     sidePxQty_results = \
    #         {side: executor.submit(sidePxQty, book_snapshot_5, side) for _, side in enumerate(['bid', 'ask'])}
    mid_prices = book_snapshot_5.filter(regex='Px').mean(axis=1)
    book_snapshot_5.loc[:, 'pret_1m'] = np.log(mid_prices/mid_prices.shift())
    book_snapshot_5 = book_snapshot_5[list(set(columns_names_ls).intersection(set(book_snapshot_5.columns)))]
    timeHMs_ls = [''.join((h, m)) for h, m in zip(book_snapshot_5.index.strftime('%H'),
                                                  book_snapshot_5.index.strftime('%M'))]
    book_snapshot_5.loc[:, 'timeHMs'] = timeHMs_ls
    book_snapshot_5.loc[:, 'timeHMe'] = book_snapshot_5.timeHMs.shift(-1)
    timeHMe_ls = ['0000' if hhmm is None else hhmm for hhmm in book_snapshot_5.timeHMe]
    book_snapshot_5.loc[:, 'timeHMe'] = timeHMe_ls
    book_snapshot_5.loc[:, 'timeHMe'].replace('None', '0000')
    """
        Clean trades
    """
    # global trades
    # trades = data_dd['trades'].drop(['local_timestamp', 'id', 'exchange'], axis=1)
    # """
    #     Extra step to handle corner case (trades that took place a few milliseconds before 12).
    #     Idea: set all indexes prior/post to tag date to the very first valid index and / or the very last
    #     valid index.
    # """
    # date_tag = tag.split('_')[-1]
    # start = ' '.join((date_tag, '00:00:00'))
    # end = ' '.join((date_tag, '23:59:00'))
    # if not trades.loc[(trades.index < start) | (trades.index > end), :].empty:
    #     idx_s = pd.Series(trades.index)
    #     idx_s = idx_s.where((idx_s >= start) & (idx_s <= end), np.nan)
    #     idx_s.ffill(inplace=True)
    #     idx_s.bfill(inplace=True)
    #     trades.index = idx_s.values
    # trades.loc[:, 'notional'] = trades.loc[:, 'price']*trades.loc[:, 'amount']
    # side_action_dd = {'ask': 'buy', 'bid': 'sell'}
    # trades_dd = dict()
    # trades_dd['ask'] = trades.query(f'side == \"{side_action_dd["ask"]}\"')
    # trades_dd['bid'] = trades.query(f'side == \"{side_action_dd["bid"]}\"')
    #
    # def treat_per_side(df: pd.DataFrame, side: str) -> None:
    #
    #     df = df.drop(['side', 'price'], axis=1)
    #     nrTrades = df.resample('T').count()['symbol']
    #     df = df.resample('T').agg({'symbol': 'last', 'amount': 'last', 'notional': 'sum'})
    #     df[f'vol{side_action_dd[side].title()}NrTrades_lit'] = nrTrades
    #     df['timeHMs'] = [''.join((h, m)) for h, m in zip(df.index.strftime('%H'), df.index.strftime('%M'))]
    #     df['timeHMe'] = df['timeHMs'].shift(-1)
    #     df['timeHMe'] = ['0000' if hhmm is None else hhmm for hhmm in df.timeHMe]
    #     df['timeHMe'].replace('None', '0000')
    #     df = df.rename(columns={'amount': f'vol{side_action_dd[side].title()}Qty_lit',
    #                             'notional': f'vol{side_action_dd[side].title()}Notional_lit'})
    #     df[f'vol{side_action_dd[side].title()}Qty_lit'].replace(np.nan, 0, inplace=True)
    #     df.fillna(method='ffill', inplace=True)
    #     trades_dd[side] = df
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     treat_per_side_results = \
    #         {side: executor.submit(treat_per_side, trades_dd[side], side) for side in list(side_action_dd.keys())}
    # idx_ls = pd.Series(list(set(trades_dd['ask'].index.tolist()+trades_dd['bid'].index.tolist())))
    # idx_ls.sort_values(inplace=True)
    # trades = trades_dd['ask'].join(trades_dd['bid'], rsuffix='_', how='outer')
    # trades.index = \
    #     trades_dd['ask'].index if trades_dd['ask'].shape[0] > trades_dd['bid'].shape[0] else trades_dd['bid'].index
    # trades['nrTrades'] = \
    #     trades[['volBuyNrTrades_lit', 'volSellNrTrades_lit']].sum(axis=1)
    #
    #
    # def volQtyNotional(df: pd.DataFrame, action: str) -> None:
    #     df[f'vol{action.title()}Qty'] = df.filter(regex=f'vol{action.title()}Qty').sum(axis=1)
    #     df[f'vol{action.title()}Notional'] = df.filter(regex=f'vol{action.title()}Notional').sum(axis=1)
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     volQtyNotional_results = \
    #         {action.title(): executor.submit(volQtyNotional, trades, action)
    #          for _, action in enumerate(['buy', 'sell'])}
    # """
    #     minutelyTARDIS table
    # """
    # common_columns_ls = list(set(book_snapshot_5.columns).intersection(set(trades.columns)))
    # trades.drop(common_columns_ls, axis=1, inplace=True)
    trades = pd.DataFrame()
    minutelyTARDIS = book_snapshot_5.join(trades, how='outer', rsuffix='_')
    minutelyTARDIS.drop(minutelyTARDIS.columns[minutelyTARDIS.columns.str.contains('_')], axis=1, inplace=True)
    minutelyTARDIS.index = list(range(0, minutelyTARDIS.shape[0]))
    minutelyTARDIS.replace(np.nan, 0, inplace=True)
    minutelyTARDIS['timeIndex'] = 1
    minutelyTARDIS['timeIndex'] = minutelyTARDIS['timeIndex'].cumsum()
    """
        Drop minutelyTARDIS table at right location
    """
    filename = '/'.join((args.location, 'minutelyTARDIS', year, exchange_tag, '.'.join((tag, 'csv.gz'))))
    minutelyTARDIS.to_csv(filename, compression='gzip')
    print(f'minutelyTARDIS table has been generated for {tag} and saved.')


def generate_data() -> None:
    """
        args.destination/*/year/exchange/data_type/file.csv.gz
    """
    for _, year in enumerate([2022]):
        target_dir = f'{"/".join((args.location, "rawTARDIS", str(year), "*", "*", "*.csv.gz"))}'
        files_ls = glob.glob(target_dir)
        file_tags_ls = list(set([f.split('/')[-1].split('.')[0] for f in files_ls]))
        for _, tag in enumerate(file_tags_ls):
            generate_data_per_tag(tag)
        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     results_generate_data_dd = \
        #         {tag: executor.submit(generate_data_per_tag, tag) for _, tag in enumerate(file_tags_ls)}


# def consolidate_data_per_date(exchange: str, date_tag: str) -> None:
#     year = date_tag.split('-')[0]
#     command = f'{args.location}/minutelyTARDIS/{year}/{exchange}/*{date_tag}.csv.gz'.replace(' ', '\\ ')
#     subprocess.run(f'gzip -d {command}', shell=True)
#     command = f'{args.location}/minutelyTARDIS/{year}/{exchange}/*_{date_tag}.csv'.replace(' ', '\\ ')
#     command2 = \
#         f'{args.location}/minutelyTARDIS/{year}/{exchange}/{"".join(date_tag.split("-"))}.csv'.replace(' ', '\\ ')
#     subprocess.run(f'cat {command} >> {command2}', shell=True)
#     command = \
#         f'{args.location}/minutelyTARDIS/{year}/{exchange}/{"".join(date_tag.split("-"))}.csv'.replace(' ', '\\ ')
#     subprocess.run(f'gzip {command}', shell=True)
#     print(f'{"".join(date_tag.split("-"))}.csv.gz has been generated.')
#     command = f'{args.location}/minutelyTARDIS/{year}/{exchange}/*.csv'.replace(' ', '\\ ')
#     subprocess.run(f'rm {command}', shell=True)


def consolidate_data(exchange: str, location: str, year: int) -> None:
    target_dir = '/'.join((location, 'minutelyTARDIS', str(year), exchange, '*.csv.gz'))
    files_ls = glob.glob(target_dir)
    master_file_ls = list()
    def read_file(file: str):
        pdb.set_trace()
        tmp = pd.read_csv(file, compression='gzip')
        master_file_ls.append(tmp)

    read_file(files_ls[0])
    with concurrent.futures.ThreadPoolExecutor() as executor:
        read_file_results = {file: executor.submit(read_file) for _, file in enumerate(files_ls)}
    pdb.set_trace()
    files_ls = [file.split('/')[-2:] for _, file in enumerate(files_ls)]
    # files_ls = [(file[0], file[-1].split('.')[0].split('_')[-1]) for _, file in enumerate(files_ls)]
    # files_ls = list(set(files_ls))
    # files_ls = [{'exchange': file[0], 'date_tag': file[-1]} for _, file in enumerate(files_ls)]
    # #consolidate_data_per_date(exchange=exchange, date_tag=files_ls[0]['date_tag'])
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     results_consolidate_data_per_date = \
    #         {(tag['exchange'], tag['date_tag']):
    #              executor.submit(consolidate_data_per_date, exchange=tag['exchange'], date_tag=tag['date_tag'])
    #          for _, tag in enumerate(files_ls)}


def consolidate_latest_data(location: str) -> None:
    last_date_tag = datetime.now()-timedelta(days=1)
    last_date_tag = last_date_tag.strftime('%Y-%m-%d')
    target_dir = '/'.join((location, f'*_{last_date_tag}.csv.gz'))
    files_ls = glob.glob(target_dir)
    files_ls = [file.split('/')[-2:] for _, file in enumerate(files_ls)]
    files_ls = [(file[0], file[-1].split('.')[0].split('_')[-1]) for _, file in enumerate(files_ls)]
    files_ls = list(set(files_ls))
    files_ls = [{'exchange': file[0], 'date_tag': file[-1]} for _, file in enumerate(files_ls)]
    for _, tag in enumerate(files_ls):
        consolidate_data_per_date(date_tag=tag['date_tag'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Minutely Tardis data.')
    parser.add_argument('--daily_update', default=0, help="Part of the script to run only for daily updates.")
    parser.add_argument('--location', help="Directory that contains subdirectories with all data needed.")
    args = parser.parse_args()
    if not bool(args.daily_update):
        for year in range(2022, 2023):
            consolidate_data('binance', args.location, year=year)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                consolidate_data_results = \
                    {exchange: executor.submit(consolidate_data, exchange, args.location)
                     for _, exchange in enumerate(exchange_ls[:1])}
    else:
        today = datetime.now()-timedelta(days=1)
        today = today.strftime('%Y-%m-%d')
        year = today.split('-')[0]
        files_ls = glob.glob(f'{args.location}/rawTARDIS/{year}/*/*/*_{today}.csv.gz')
        files_ls = list(set([file.split('/')[-1].split('.')[0] for _, file in enumerate(files_ls)]))
        for _, tag in enumerate(files_ls):
            generate_data_per_tag(tag)
        consolidate_latest_data(args.location)