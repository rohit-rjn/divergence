from divergence_signals import find_divergence
from get_data import get_hourly, get_daily, resample_data
import json
import datetime 
import warnings
def get_df(token):
    #duration = duration.split()
    df = get_hourly(token)
    df_daily = get_daily(token)
    df_weekly = resample_data(df_daily, '1','WEEK')
    df_weekly = df_weekly.reset_index(level=None, drop=False, inplace=False, col_level=0)

    return df, df_weekly

def get_signals(token_list,duration):
    signals =[]
    hourly_signal = [[] for _ in range(len(duration))]
    #hourly_signal= [hourly_signal]*len(duration)
    for i in range(len(token_list)):
        try:                                                                                                                                         
            df, df_weekly = get_df(token_list[i])
        except:
            message = 'DataFrame '+str(token_list[i])+' not found'
            warnings.warn(message)
            continue

        
        for j in range(len(duration)):
            curr_duration = duration[j]
            #print(curr_duration)
            
            curr_duration = curr_duration.split()
            df_new = resample_data(df, curr_duration[0],curr_duration[1])
            df_new = df_new.reset_index(level=None, drop=False, inplace=False, col_level=0)
            div_df = find_divergence(df_new, df_weekly, curr_duration[0])
            if (div_df is not None):
                dict_div = {}
                dict_div['token']=token_list[i]
                dict_div['duration']=duration[j]
                dict_div['start']=div_df['start']
                dict_div['end']=div_df['end']
                dict_div['type']=div_df['type']
                dict_div['cosine']=div_df['cosine']
                dict_div['market_state']=div_df['market_state']
                dict_div['volatility']=div_df['volatility']
                hourly_signal[j].append(dict_div)
    signals_small = {}
    #print(hourly_signal[0])
    for i in range(len(duration)):
        signals_small[duration[i]]=hourly_signal[i]
    json_string=json.dumps(signals_small, default=str)
    
    return json_string

if __name__=="__main__":
    token_list = ['BTC', 'ETH', 'XRP', 'LTC', 'BAB', 'BCH', 'XPP']
    duration_list = ['2 HOUR','4 HOUR','8 HOUR','12 HOUR']
    signals = get_signals(token_list,duration_list)
    print(signals)