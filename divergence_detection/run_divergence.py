from divergence_signals import find_divergence
from get_data import get_hourly, get_weekly, resample_data
import json
from gemini  import data, engine, helpers
import datetime 

def get_df(token, duration):
    duration = duration.split()
    df = get_hourly(token)
    df = resample_data(df, duration[0],duration[1])
    df = df.reset_index(level=None, drop=False, inplace=False, col_level=0)

    #print(df.tail())
    df_weekly = get_weekly(token)
    df_weekly = resample_data(df_weekly, '1','WEEK')
    df_weekly = df_weekly.reset_index(level=None, drop=False, inplace=False, col_level=0)

    return df, df_weekly

def get_signals(token_list,duration):
    signals =[]
    for i in range(len(token_list)):
        
        df, df_weekly = get_df(token_list[i], duration)
        div_df = find_divergence(df, df_weekly, duration)
        dict_div = {}
        dict_div['token']=token_list[i]
        dict_div['duration']=duration
        dict_div['start']=div_df['start']
        dict_div['end']=div_df['end']
        dict_div['type']=div_df['type']
        dict_div['cosine']=div_df['cosine']
        dict_div['market_state']=div_df['market_state']
        dict_div['volatility']=div_df['volatility']
        json_string=json.dumps(dict_div, default=str)
        signals.append(json_string)
    return signals

if __name__=="__main__":
    token_list = ['BTC', 'ETH', 'XRP', 'LTC', 'BAB', 'BCH']
    signals = get_signals(token_list,'4 HOUR')
    print(signals)