import candles
import divergences

# Global
import json
import warnings
warnings.filterwarnings('ignore')

# Divergences ---------------------------

def get_df(token,signal_type):
    if signal_type == 'hour':
        df = candles.get_hourly(token)
    elif signal_type == 'minute':
        df = candles.get_minutely(token)
    df_daily = candles.get_daily(token)
    df_weekly = candles.resample_data(df_daily, '1','WEEK')
    df_weekly = df_weekly.reset_index(level=None, drop=False, inplace=False, col_level=0)
    return df, df_weekly

def get_signals(tokens, durations, signal_type):
    signals =[]
    for token in tokens:
        print(token)
        try:
            df, df_weekly = get_df(token,signal_type)
            for duration in durations:
                try:
                    duration_split = duration.split()
                    df_new = candles.resample_data(df, duration_split[0], duration_split[1])
                    df_new = df_new.reset_index(level=None, drop=False, inplace=False, col_level=0)
                    div_df = divergences.find_divergence(df_new, df_weekly, duration_split[0])
                    if (div_df is not None):
                        dict_div = {}
                        dict_div['token']         = token
                        dict_div['duration']      = duration
                        dict_div['start']         = div_df['start'].strftime("%Y-%m-%d %H:%M")
                        dict_div['end']           = div_df['end'].strftime("%Y-%m-%d %H:%M")
                        dict_div['type']          = div_df['type']
                        dict_div['cosine']        = round(div_df['cosine'], 3)
                        dict_div['stochrsi_div']  = div_df['stochrsi_div']
                        dict_div['mfi_div']       = div_df['mfi_div']
                        dict_div['rsi_div']       = div_df['rsi_div']
                        dict_div['obv_div']       = div_df['obv_div']
                        dict_div['market_state']  = div_df['market_state']
                        dict_div['volatility']    = div_df['volatility']
                        dict_div['macd_20_50_9']  = div_df['macd_20_50_9']
                        dict_div['macd_50_100_20']= div_df['macd_50_100_20']
                        dict_div['macd_50_100_9'] = div_df['macd_50_100_9']
                        signals.append(dict_div)
                except Exception as e:
                    print(e)
                    print('Skipping '+str(token)+' on '+str(duration))
        except Exception as e:
            print(e)
            print('Skipping '+str(token))

    return(signals)

def post_signals():
    tokens = ['BTC', 'ETH', 'XRP', 'LTC','BCH', 'EOS','BSV', 'TRX', 'ETC' ]
    #tokens = tokens[:20] # Top 20
    durations = ['2 HOUR','4 HOUR','8 HOUR','12 HOUR']
    signals = get_signals(tokens, durations,'hour')

    #print(signals)

def post_signals_minutely():
    minutes_list = ['5 MIN','10 MIN','15 MIN','20 MIN', '30 MIN', '45 MIN']
    token_list = ['BTC']
    signals_minutely = get_signals(token_list,minutes_list,'minute')
    
    #print(signals_minutely)

# Execute for Hourly basis - for Top 7 tokens
#post_signals()

# Execute or minutely basis for BTC 
post_signals_minutely()