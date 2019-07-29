import time
import requests
import datetime as dt
import pandas as pd

api_key = "59af9654fde15836d6274c389825375fb3dabd7c33552e0f6dd60d2523fdbe55"

tf_mapper = {'MIN':'T','HOUR':'H','DAY':'D','WEEK':'W','MONTH':'M'}

def resample_data(df, integer, htf):
    """Resample dataframe to a higher timeframe.
    :param df: An HLOCV pandas dataframe with a datetime index
    :type df: pandas.DataFrame
    :param integer: The frequency of the higher timeframe
    :type integer: int
    :param htf: The period of the higher timeframe (e.g. 'MIN', 'HOUR', 'DAY', 'WEEK', "MONTH")
    :type htf: str
    :return: A resampled pandas dataframe
    :rtype: pandas.DataFrame
    """
    htf = str(integer)+tf_mapper[htf]
    df['low']    = df.low.resample(htf).min()
    df['high']   = df.high.resample(htf).max()
    df['open']   = df.open.resample(htf).first()
    df['close']  = df.close.resample(htf).last()
    df['volume'] = df.volume.resample(htf).sum()
    return df.dropna()

def available_units():
    """Get available periods for resampling to higher timeframes.
    :return: A list of available time frames
    :rtype: list
    """
    return tf_mapper.keys()

def tf_to_secs(freq, unit):
    """Convert a timeframe into its equivalent in seconds.
    :param freq: The frequency of the timeframe
    :type freq: int
    :param unit: The period of the timeframe (e.g. 'MIN', 'HOUR', 'DAY', 'WEEK', "MONTH")
    :type unit: str
    :return: A timeframe represented as seconds
    :rtype: int
    """
    multiplier = {'MIN'  : 60,
                  'HOUR' : 3600,
                  'DAY'  : 86400,
                  'WEEK' : 604800,
                  'MONTH': 18144000}
    return freq*multiplier[unit]

def cc_available_pairs(exchange, show=False):
    """Print available trading pairs for Cypto Compare.
    :param exchange: The exchange to get data for
    :type exchange: str
    :param show: Pretty print data
    :type show: bool
    :return: A list of trading pairs
    :rtype: list
    """
    url = "https://min-api.cryptocompare.com/data/v2/all/exchanges"
    response = requests.get(url)
    data = response.json()
    pairs = []
    if data['Response'] == "Success":
        for i in data['Data']:
            if i == exchange:
                exchange_pairs = data['Data'][i]['pairs']
                for ticker, bases in exchange_pairs.items():
                    pairs += ["{0}_{1}".format(ticker, b) for b in bases]
                break
    else:
        raise ValueError(data["Message"])    

    # Case-insensitive sort
    pairs = sorted(pairs, key=lambda s: s.lower())

    if show:
        print("Available Pairs for {0}...".format(exchange))
        ptable.tableize(pairs, cols = 5).show()
    else:
        return pairs

def get_hourly(symbol, market="USD"):
    params = {'fsym': symbol,
              'tsym': market,
              'limit': 2000,
              'api_key': api_key}
    response = requests.get('https://min-api.cryptocompare.com/data/histohour', params=params)
    data = response.json()
    if data['Response'] == "Success":
        df = pd.DataFrame(data['Data'])
        df['date'] = pd.to_datetime(df['time'], unit='s')
        df['volume'] = df['volumefrom']
        df = df.set_index(df.date, drop=True)
        df = df[['low', 'high', 'open', 'close', 'volume']]
        return(df)
    else:
        raise ValueError(data["Message"])

def get_minutely(symbol, market="USD"):
    params = {'fsym': symbol,
              'tsym': market,
              'limit': 2000,
              'api_key': api_key}
    response = requests.get('https://min-api.cryptocompare.com/data/histominute', params=params)
    data = response.json()
    if data['Response'] == "Success":
        df = pd.DataFrame(data['Data'])
        df['date'] = pd.to_datetime(df['time'], unit='s')
        df['volume'] = df['volumefrom']
        df = df.set_index(df.date, drop=True)
        df = df[['low', 'high', 'open', 'close', 'volume']]
        return(df)
    else:
        raise ValueError(data["Message"])

def get_daily(symbol, market="USD"):
    params = {'fsym': 'ETH',
              'tsym': market,
              'limit': 1000,
              'api_key': api_key}
    response = requests.get('https://min-api.cryptocompare.com/data/histoday', params=params)
    data = response.json()
    if data['Response'] == "Success":
        df = pd.DataFrame(data['Data'])
        df['date'] = pd.to_datetime(df['time'], unit='s')
        df['volume'] = df['volumefrom']
        df = df.set_index(df.date, drop=True)
        df = df[['low', 'high', 'open', 'close', 'volume']]
        return(df)
    else:
        raise ValueError(data["Message"])
