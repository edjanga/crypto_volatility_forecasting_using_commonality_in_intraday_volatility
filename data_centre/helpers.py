"""Largest coins - To be queried against USDT"""
coin_ls = ['btc', 'eth', 'bnb', 'xrp', 'ada', 'ltc', 'etc', 'matic', 'ftm', 'doge', 'shib',
           'link', 'avax', 'atom', 'trx']#, 'eos', 'xlm', 'zec', 'neo', 'dash', 'grt', 'snx', 'dydx',
           #'dot', 'fil', 'cfx', 'apt', 'sol']
"""Intersection of the top 10 largest spot exchanges listed by CoinMarketCap
   (https://coinmarketcap.com/rankings/exchanges/) and exchanges available through Tardis API."""
exchange_ls = ['binance', 'okex', 'huobi', 'bitstamp', 'bybit'] #'kraken'

"""Data type queried"""
data_type_ls = ['book_snapshot_5', 'trades']
