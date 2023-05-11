import glob
import pdb
import concurrent.futures
import ast
from decouple import config
from helpers import *
from tardis_dev import datasets, get_exchange_details
import os
import pandas as pd
from datetime import datetime, timedelta
import aiohttp
import urllib
import argparse
import numpy as np
import pytz


API_KEY = config('API_KEY')


def clean_dd(info_dd: dict) -> dict:

    del info_dd['type']
    del info_dd['dataTypes']
    del info_dd['stats']
    return {info_dd['id']: [info_dd[date] for date in ['availableSince', 'availableTo']]}


columns_names_ls = ['symbol', 'timeIndex', 'timeHMs', 'timeHMe', 'volBuyQty', 'volSellQty', 'volBuyNotional',
                    'volSellNotional', 'nrTrades', 'volBuyQty_lit', 'volBuyQty_hid', 'volSellQty_lit', 'volSellQty_hid',
                    'volBuyNotional_lit', 'volBuyNotional_hid', 'volSellNotional_lit', 'volSellNotional_hid',
                    'volBuyNrTrades_lit', 'volSellNrTrades_hid', 'volSellNrTrades_lit',
                    'volSellNrTrades_hid', 'pret_1m', 'bidPx', 'askPx', 'bidQty', 'askQty']


def file_name_nested(exchange, data_type, date, symbol, format):
    return f"{exchange}/{data_type}/{symbol.replace('-','')}_{date.strftime('%Y-%m-%d')}.{format}.gz"


def delete_file(file) -> None:
    os.remove(os.path.abspath(file))
    print(f'{file} has been deleted.')


def generate_data_per_tag(tag: str, destination: str) -> None:
    if 'USDCUSDT_' in tag:
        return None
    year = tag.split('_')[-1].split('-')[0]
    data_type_ls = ['book_snapshot_5', 'trades']
    try:
        data_dd = \
            {data_type: pd.read_csv(filepath_or_buffer=
                                    glob.glob(f'{destination}/rawTARDIS/{year}/binance/{data_type}/{tag}.csv.gz')[0],
                                    index_col='timestamp',
                                    date_parser=lambda x: datetime.fromtimestamp(int(x)/1_000_000,
                                                                                 tz=pytz.timezone('Etc/UTC')))
           for _, data_type in enumerate(data_type_ls)}

    except IndexError:
        print(f'Either book_snapshot_5 or trades data is missing for {tag}.')
        return None
    else:
        exchange_tag = data_dd['book_snapshot_5'].exchange.unique()[0]
        """
            Clean book_snapshot_5
        """
        book_snapshot_5 = data_dd['book_snapshot_5'].drop(['exchange', 'local_timestamp'], axis=1)
        book_snapshot_5 = book_snapshot_5.resample('T').last()
        for side in ['bid', 'ask']:
            book_snapshot_5[f'{side}Px'] = book_snapshot_5[f'{side}s[0].price']
            book_snapshot_5[f'{side}Qty'] = book_snapshot_5[f'{side}s[0].amount']
        book_snapshot_5.ffill(inplace=True)
        mid_prices = book_snapshot_5.filter(regex='Px').mean(axis=1)
        book_snapshot_5['pret_1m'] = np.log(mid_prices/mid_prices.shift())
        book_snapshot_5 = book_snapshot_5[list(set(columns_names_ls).intersection(set(book_snapshot_5.columns)))]
        book_snapshot_5['timeHMs'] = \
            [''.join((h, m)) for h, m in zip(book_snapshot_5.index.strftime('%H'), book_snapshot_5.index.strftime('%M'))]
        book_snapshot_5['timeHMe'] = book_snapshot_5['timeHMs'].shift(-1)
        book_snapshot_5['timeHMe'] = ['0000' if hhmm is None else hhmm for hhmm in book_snapshot_5.timeHMe]
        book_snapshot_5['timeHMe'].replace('None', '0000')
        """
            Clean trades
        """
        global trades
        trades = data_dd['trades'].drop(['local_timestamp', 'id', 'exchange'], axis=1)
        trades = trades.loc[' '.join((tag.split('_')[-1], '00:00:00')):, :]
        trades = trades.assign(notional=trades['price']*trades['amount'])
        side_action_dd = {'ask': 'buy', 'bid': 'sell'}
        trades_dd = dict()
        trades_dd['ask'] = trades.query(f'side == \"{side_action_dd["ask"]}\"')
        trades_dd['bid'] = trades.query(f'side == \"{side_action_dd["bid"]}\"')

        def treat_per_side(df: pd.DataFrame, side: str) -> None:

            df = df.drop(['side', 'price'], axis=1)
            nrTrades = df.resample('T').count()['symbol']
            df = df.resample('T').agg({'symbol': 'last', 'amount': 'last', 'notional': 'sum'})
            df[f'vol{side_action_dd[side].title()}NrTrades_lit'] = nrTrades
            df['timeHMs'] = [''.join((h, m)) for h, m in zip(df.index.strftime('%H'), df.index.strftime('%M'))]
            df['timeHMe'] = df['timeHMs'].shift(-1)
            df['timeHMe'] = ['0000' if hhmm is None else hhmm for hhmm in df.timeHMe]
            df['timeHMe'].replace('None', '0000')
            df['timeHMe'] = np.where(df['timeHMs'] == '2358', '2359', df['timeHMe'])
            df = df.rename(columns={'amount': f'vol{side_action_dd[side].title()}Qty_lit',
                                    'notional': f'vol{side_action_dd[side].title()}Notional_lit'})
            df[f'vol{side_action_dd[side].title()}Qty_lit'].replace(np.nan, 0, inplace=True)
            df.fillna(method='ffill', inplace=True)
            trades_dd[side] = df

        treat_per_side(trades_dd['ask'], 'ask')
        treat_per_side(trades_dd['bid'], 'bid')
        trades = pd.concat([trades_dd['ask'], trades_dd['bid'].drop(['timeHMs', 'timeHMe', 'symbol'], axis=1)], axis=1)
        trades[['volBuyNrTrades_hid', 'volSellNrTrades_hid', 'volBuyQty_hid', 'volSellQty_hid',
                'volBuyNotional_hid', 'volSellNotional_hid', ]] = np.nan
        trades['nrTrades'] = \
            trades[['volBuyNrTrades_lit', 'volSellNrTrades_lit',
                    'volBuyNrTrades_hid', 'volSellNrTrades_hid']].sum(axis=1)
        for action in ['buy', 'sell']:
            trades[f'vol{action.title()}Qty'] = trades.filter(regex=f'vol{action.title()}Qty').sum(axis=1)
            trades[f'vol{action.title()}Notional'] = trades.filter(regex=f'vol{action.title()}Notional').sum(axis=1)
        """
            minutelyTARDIS table
        """
        minutelyTARDIS = pd.concat([book_snapshot_5.drop(['symbol', 'timeHMs', 'timeHMe'], axis=1), trades], axis=1)
        minutelyTARDIS['timeIndex'] = 1
        minutelyTARDIS['timeIndex'] = minutelyTARDIS['timeIndex'].cumsum()
        minutelyTARDIS.ffill(inplace=True)
        assert not list(set(columns_names_ls).difference(set(minutelyTARDIS.columns)))
        assert (minutelyTARDIS.shape[0] == 1440)
        """
            Drop minutelyTARDIS table at right location
        """
        year_tag = tag.split('_')[-1].split('-')[0]
        file = '/'.join((f'{args.destination}', 'minutelyTARDIS', year_tag, exchange_tag, '.'.join((tag, 'csv.gz'))))

        minutelyTARDIS.to_csv(os.path.abspath(file), compression='gzip')
        print(f'{file} has been created.')
        book_snapshot_5_and_trades = \
            glob.glob(f'{destination}/rawTARDIS/{year_tag}/{exchange_tag}/*/{".".join((tag, "csv.gz"))}')
        with concurrent.futures.ThreadPoolExecutor() as executor:
            delete_book_snapshot_5_and_trades_results = \
                {file: executor.submit(delete_file, file) for _, file in enumerate(book_snapshot_5_and_trades)}


def download_data(exchange: str, symbol: str, year: int) -> None:

        today = datetime.now()
        availableSince = datetime(year, 1, 1).strftime('%Y-%m-%d')
        availableTo = datetime(year+1, 1, 1) if datetime(year+1, 1, 1) < today else today
        availableTo = availableTo.strftime('%Y-%m-%d')
        print(f"[START FETCHING]: {symbol} from {exchange} between {availableSince} and {availableTo} has started.")

        try:
            datasets.download(exchange=exchange, data_types=data_type_ls, from_date=availableSince, to_date=availableTo,
                              symbols=[symbol],
                              api_key=API_KEY, get_filename=file_name_nested,
                              download_dir=f'{args.destination}/rawTARDIS/{year}/')

        except aiohttp.client_exceptions.ClientOSError:
            print(f"[SKIP FETCHING]: {symbol} from {exchange} between {availableSince} and {availableTo} is skipped.")

        except urllib.error.HTTPError as e:
            # String representation of dictionary to actual dictionary
            availableSince = ast.literal_eval(e.msg)['datasetInfo']['availableSince']
            availableSince = availableSince.replace('Z', '').replace('T', ' ').split(' ')[0]
            availableTo_ls = availableSince.split('-')
            availableTo = datetime(int(availableTo_ls[0])+1, 1, 1).strftime('%Y-%m-%d').split(' ')[0]
            print(f"[NEW ATTEMPT FETCHING]: "
                  f"{symbol} from {exchange} between {availableSince} and {availableTo} has started.")
            try:
                datasets.download(exchange=exchange, data_types=data_type_ls,
                                  from_date=availableSince, to_date=availableTo, symbols=[symbol],
                                  api_key=API_KEY, get_filename=file_name_nested,
                                  download_dir=f'{args.destination}/"rawTARDIS"/{year}/')
            except urllib.error.HTTPError:
                print(f'[END FETCHING]: Fetching data for {symbol} between {availableSince} and '
                      f'{availableTo} has stopped as data is not available.')
                return None
        except KeyError:
            return None
        else:
            print(f"[END FETCHING]: {symbol} from {exchange} between {availableSince} and {availableTo} has now ended.")


def generate_data(destination: str, exchange: str, year: int) -> None:

    tag_ls = glob.glob(f'{destination}/rawTARDIS/{year}/{exchange}/*/*.csv.gz')
    tag_ls = list(set([name.split('/')[-1].split('.')[0] for _, name in enumerate(tag_ls)]))
    with concurrent.futures.ThreadPoolExecutor() as executor:
        generate_data_per_tag_results_dd = \
            {tag: executor.submit(generate_data_per_tag,
                                  tag=tag, destination=destination) for _, tag in enumerate(tag_ls)}


def aggregate_data(destination: str, exchange: str, year: int) -> None:
    symbol_dd = dict([(''.join((sym, 'usdt')).upper(), None) for _, sym in enumerate(coin_ls)])
    tag_ls = glob.glob(f'{destination}/minutelyTARDIS/{year}/{exchange}/*.csv.gz')

    def append_data_per_symbol(symbol: str, year: int) -> None:
        symbol_ls = \
            [tag_sym for _, tag_sym in enumerate(tag_ls)
             if (''.join((symbol, 'usdt')).upper() in tag_sym) & (str(year) in tag_sym)]
        df = pd.concat([pd.read_csv(sym_file, compression='gzip') for _, sym_file in enumerate(symbol_ls)])
        try:
            df.drop('Unnamed: 0', axis=1, inplace=True)
        except KeyError:
            pass
        df.sort_values('timestamp', inplace=True)
        symbol_dd[''.join((symbol, 'usdt')).upper()] = df


    with concurrent.futures.ThreadPoolExecutor() as executor:
        append_data_per_symbol_results_dd = \
            {symbol: executor.submit(append_data_per_symbol, symbol=symbol, year=year)
             for _, symbol in enumerate(coin_ls)}
    aggregate_data_df = pd.concat(symbol_dd)
    null_per_field_s = aggregate_data_df.isnull().sum()
    columns_to_be_dropped_s = null_per_field_s[null_per_field_s == aggregate_data_df.shape[0]].index
    aggregate_data_df.drop(columns_to_be_dropped_s, axis=1, inplace=True)
    symbol_group = aggregate_data_df.groupby('symbol')
    number_of_minutes_in_year = pd.date_range(start=datetime(year, 1, 1), end=datetime(year+1, 1, 1),
                                              inclusive='left', freq='1T').shape[0]
    aggregate_data_df.sort_values(by=['symbol', 'timestamp'], inplace=True)
    aggregate_data_df.index = list(range(0, aggregate_data_df.shape[0]))
    symbol_size_s = symbol_group.size()
    valid_symbol_s = symbol_size_s[symbol_size_s == number_of_minutes_in_year].index.tolist()
    aggregate_data_df = aggregate_data_df.loc[aggregate_data_df.symbol.isin(valid_symbol_s), :]
    aggregate_data_df.to_parquet(os.path.abspath(f'./data_centre/tmp/aggregate{year}'))
    print(f'[Aggregate Data]: Aggregate data for {exchange}{str(year)}.')


def make_year_subdir(year: int) -> None:
    year = str(year)
    if year not in os.listdir(os.path.abspath(f'{"/".join((args.destination, "rawTARDIS"))}')):
        os.mkdir(os.path.abspath(f'{"/".join((args.destination, "rawTARDIS", year))}'))
        if args.exchange not in os.listdir(os.path.abspath(f'{"/".join((args.destination, "rawTARDIS", year))}')):
            os.mkdir(os.path.abspath(f'{"/".join((args.destination, "rawTARDIS", year, args.exchange))}'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tardis historical data scraper.')
    parser.add_argument('--exchange', default="binance", help='Exchange to query data for.')
    parser.add_argument('--destination', help="Destination directory to store data.")
    args = parser.parse_args()
    download_data(exchange=args.exchange, symbol='BTCUSDT', year=2022)
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     make_year_subdir_results = {year: executor.submit(make_year_subdir, year) for year in list(range(2019, 2024))}
    #
    # for year in range(2022, 2023):
    #     with concurrent.futures.ThreadPoolExecutor() as executor:
    #         download_data_results = \
    #             {year: executor.submit(download_data(exchange='binance',
    #                                                  symbol=''.join((coin, 'usdt')).upper(),
    #                                                  year=year)) for _, coin in enumerate(coin_ls)}

    # for year in range(2022, 2024):
    #     if year == 2023:
    #         break
    # generate_data(args.destination, args.exchange, year=2022)
    # aggregate_data(args.destination, args.exchange, 2022)
    #
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     generate_data_results = \
    #         {year: executor.submit(generate_data, args.exchange, year) for year in list(range(2021, 2022))}
