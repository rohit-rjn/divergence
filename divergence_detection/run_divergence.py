from divergences import find_divergence
from candles import get_hourly, get_daily,get_minutely, resample_data
import json
import datetime 
import warnings
warnings.filterwarnings('ignore')

def get_df(token):
    df = get_hourly(token)
    df_daily = get_daily(token)
    df_weekly = resample_data(df_daily, '1','WEEK')
    df_weekly = df_weekly.reset_index(level=None, drop=False, inplace=False, col_level=0)
    return df, df_weekly

def get_signals(token_list,duration):
    signals =[]
    hourly_signal = [[] for _ in range(len(duration))]
    
    for i in range(len(token_list)):
        print(token_list[i])
        try:
            df, df_weekly = get_df(token_list[i])
            for j in range(len(duration)):
                try:
                    curr_duration = duration[j]                
                    curr_duration = curr_duration.split()
                    df_new = resample_data(df, curr_duration[0],curr_duration[1])
                    df_new = df_new.reset_index(level=None, drop=False, inplace=False, col_level=0)
                    div_df = find_divergence(df_new, df_weekly, curr_duration[0])
                    #print(div_df)
                    if (div_df is not None):
                        dict_div = {}
                        dict_div['token']        = token_list[i]
                        dict_div['duration']     = duration[j]
                        dict_div['start']        = div_df['start'].strftime("%Y-%m-%d %H:%M")
                        dict_div['end']          = div_df['end'].strftime("%Y-%m-%d %H:%M")
                        dict_div['type']         = div_df['type']
                        dict_div['cosine']       = div_df['cosine']
                        dict_div['stochrsi_div'] = div_df['stochrsi_div']
                        dict_div['mfi_div']      = div_df['mfi_div']
                        dict_div['rsi_div']      = div_df['rsi_div']
                        dict_div['obv_div']      = div_df['obv_div']
                        dict_div['market_state'] = div_df['market_state']
                        dict_div['volatility']   = div_df['volatility']
                        dict_div['macd_20_50_9'] = div_df['macd_20_50_9']
                        dict_div['macd_50_100_20']= div_df['macd_50_100_20']
                        dict_div['macd_50_100_9'] = div_df['macd_50_100_9']
                        hourly_signal[j].append(dict_div)
                except Exception as e:
                    print(e)
                    print('Skipping '+str(token_list[i])+' on '+str(duration[j]))
        except Exception as e:
            print(e)
            print('Skipping '+str(token_list[i]))

    signals_small = {}
    for i in range(len(duration)):
        signals_small[duration[i]]=hourly_signal[i]
    return(signals_small)

if __name__=="__main__":
    token_list = ['BTC','ETH', 'XRP', 'LTC', 'BAB', 'BCH', 'XPP']
    duration_list = ['1 HOUR','2 HOUR','4 HOUR','8 HOUR','12 HOUR']
    signals = get_signals(token_list,duration_list)
    print(signals)
