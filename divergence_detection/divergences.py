# Global Imports
import pandas as pd
import numpy as np
import talib
from scipy import stats, spatial
from pandas import Series
from pykalman import KalmanFilter
import json
from numpy import NaN, Inf, arange, isscalar, asarray, array
import warnings


# Function to Add the MACD to DataFrame 
def add_macd(df):
    output = talib.MACD(df['close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd'] = output[0]
    df['macdsignal'] = output[1]
    df['macdhist'] = output[2]
    return df


# Add Kalman Filter to the the DataFrame 
def add_kalman(df):
    # Construct a Kalman filter
    kf = KalmanFilter(transition_matrices = [1],
                  observation_matrices = [1],
                  initial_state_mean = 0,
                  initial_state_covariance = 1,
                  observation_covariance=1,
                  transition_covariance=.01)
    state_means, _ = kf.filter(df['close'].values)
    state_means = pd.Series(state_means.flatten(), index=df['close'].index)
    df['kf']= state_means
    return df



# Code to Merge consecutive Divergences 
def merge(divergence_list):
    divergence_merge = []
    n = len(divergence_list)
    i=0
    while i < n-1:
        start=  divergence_list[i][0]
        while i< n-1 and  divergence_list[i][1] == divergence_list[i+1][0] and divergence_list[i][2]==divergence_list[i+1][2]:
            i+=1
        end = divergence_list[i][1]
        div_type = divergence_list[i][2]
        divergence_merge.append([start, end, div_type])
        i+=1
            
    return divergence_merge

# This code is to find the Minima's in MACD given the price's Minima



def signal_strength_cosine(df, divergence_list):
    #Step 1 Convert signal to Vectors
    vector_macd = []
    vector_price = []
    start = []
    end = []
    signal_type = []
     
    for i in range(0,len(divergence_list)):
        y_macd = df['macdhist'].loc[divergence_list[i][1]] - df['macdhist'].loc[divergence_list[i][0]]
        x_macd = divergence_list[i][1] - divergence_list[i][0]
        vector_macd.append([x_macd,y_macd])
        x_price = divergence_list[i][1] - divergence_list[i][0]
        y_price = df['close'].loc[divergence_list[i][1]] - df['close'].loc[divergence_list[i][0]]
        vector_price.append([x_price,y_price])
    
        start.append(df['date'].loc[divergence_list[i][0]])
        end.append(df['date'].loc[divergence_list[i][1]])
        if(divergence_list[i][2]==1):
            signal_type.append('Bullish')
        elif divergence_list[i][2]==2:
            signal_type.append('Hidden Bullish')
        elif divergence_list[i][2]==3:
            signal_type.append('Bearish')
        elif divergence_list[i][2]==4:
            signal_type.append('Hidden Bearish')

    signal_cosine = []
    for i in range(0, len(vector_macd)):

        quality = spatial.distance.cosine(vector_macd[i],vector_price[i])
        
        signal_cosine.append(quality)


    df_signal = pd.DataFrame()
    df_signal['start']= start
    df_signal['end'] = end
    df_signal['type']= signal_type
    df_signal['cosine']=signal_cosine
    if len(df_signal)>0:
        return df_signal.iloc[-1]
    else:
        return None
    

def add_divergence_macd(df,smoothing_price_pct=0.02, smoothing_macd=0.001):
    mintab_c = []
    maxtab_c = []
    mintab_m = []
    maxtab_m = []
    maxtab_p = []
    mintab_p = []
    
    #v_m = df2['macdhist'].values
    v_m = df['macdhist'].values
    v_c = df['kf'].values
    v_p = df['close'].values
    delta_m = smoothing_macd
    delta_c = smoothing_price_pct #150
    mn_c, mx_c = Inf, -Inf
    mnpos_m, mxpos_m=NaN, NaN
    mn_m, mx_m = Inf, -Inf
    mnpos_c, mxpos_c=NaN, NaN
        
    mn_p, mx_p = Inf, -Inf
    mnpos_p, mxpos_p=NaN, NaN
    
    divergence_list=[]
    lookformax_m = True
    lookformax_c = True


    for i in range(10,len(v_m)):
    
        this_m = v_m[i]
        if this_m > mx_m:
            mx_m = this_m
            mxpos_m = i
        if this_m < mn_m:
            mn_m = this_m
            mnpos_m = i
        
        # For bear Div, and Hidden Bear Div
        if lookformax_m:
            if this_m < mx_m-(delta_m*mx_m) and this_m <0 :    
            
                if len(maxtab_m)>0:
                    prev_local_maxima= maxtab_m[-1]
                else:
                    prev_local_maxima = [0, -Inf]
                #if(mx_c )
                maxtab_m.append((mxpos_m, mx_m))
                new_local_maxima= [mxpos_m, mx_m]
                mn_m = this_m
                
                # For Bear Div - MACD making lower high and price making higher high
                if(prev_local_maxima[1] > new_local_maxima[1]):
                    if df['close'].iloc[int(prev_local_maxima[0])] < df['close'].iloc[int(new_local_maxima[0])]:
                        divergence_list.append([prev_local_maxima[0],new_local_maxima[0],3])
                # Hidden Bear - MACD making higher high, price making lower high
                if(prev_local_maxima[1] < new_local_maxima[1]):
                    if df['close'].iloc[int(prev_local_maxima[0])] > df['close'].iloc[int(new_local_maxima[0])]:
                        divergence_list.append([prev_local_maxima[0],new_local_maxima[0],4])
                mnpos_m = i #x[i]
                lookformax_m = False
        else:
            # Bull Divergence 
            if this_m > mn_m+(delta_m*mn_m) and this_m > 0:
                if len(mintab_m) > 0:
                    prev_local_minima = mintab_m[-1]
                else:
                    prev_local_minima= [0, Inf]
                mintab_m.append((mnpos_m, mn_m))
                new_local_minima= [mnpos_m, mn_m]
                mx_m = this_m
                # Bull Div- MACD higher low, Price lower Low
                if(prev_local_minima[1] < new_local_minima[1]):
                    if df['close'].iloc[int(prev_local_minima[0])] > df['close'].iloc[int(new_local_minima[0])]:
                        divergence_list.append([prev_local_minima[0],new_local_minima[0],1])
                # Hidden Bull Div- Macd lower low, price higher low
                elif prev_local_minima[1] > new_local_minima[1]:
                    if df['close'].iloc[int(prev_local_minima[0])] < df['close'].iloc[int(new_local_minima[0])]:
                        divergence_list.append([prev_local_minima[0],new_local_minima[0],2])
                mxpos_m = i #x[i]
                lookformax_m = True
    return df,divergence_list, mintab_m, maxtab_m


def find_complete(df, divergence_list):
    df['signal_complete_end']=0
    for i in range(len(divergence_list)):
        if divergence_list[i][2]==1 or divergence_list[i][2]==3:
            j = int(divergence_list[i][1])
            if (df['macdhist'].loc[j]>0):
                #print(divergence_list[i])
                while (df['macdhist'].loc[j] > 0 ):
                    j+=1
                    df['signal_complete'].loc[j]=divergence_list[i][2]
                
                df['signal_complete_end'].loc[j]=1
            elif(df['macdhist'].loc[j]<0):
                #print(divergence_list[i])
                while (df['macdhist'].loc[j] < 0 ):
                    j+=1
                    df['signal_complete'].loc[j]=divergence_list[i][2]
            
                df['signal_complete_end'].loc[j]=1
    return df


#Finding MACD and adding to df
def add_macd_weekly(df):
    output = talib.MACD(df['close'].values, fastperiod=50, slowperiod=100, signalperiod=20)
    df['macd'] = output[0]
    df['macdsignal'] = output[1]
    df['macdhist'] = output[2]
    return df

# 0-No trends, 1-up trend(bull), -1 Down Trend(bear)
def add_trends(df):
    df['trend']=[0]*len(df)
    for i in range(len(df)-2):
        if(df['macdhist'].iloc[i] < df['macdhist'].iloc[i+1] and df['macdhist'].iloc[i+1] < df['macdhist'].iloc[i+2]):
            df['trend'].iloc[i+2]=1
        
        elif(df['macdhist'].iloc[i] > df['macdhist'].iloc[i+1] and df['macdhist'].iloc[i+1] > df['macdhist'].iloc[i+2]):
            df['trend'].iloc[i+2]= -1
    return df


def add_state(df_weekly, div_df):
    
    df_weekly = add_macd_weekly(df_weekly)
    df_weekly = add_trends(df_weekly)

    if df_weekly['trend'].iloc[-1] == 1:
        div_df['market_state']='Bullish'
    elif df_weekly['trend'].iloc[-1] == -1:
        div_df['market_state']='Bearish'
    return div_df


# To add the rolling Volatility to the code
def rolling_volatility(df, duration):
    #duration = duration.split()
    df['rolling_volatility']=0
    rolling_volatility_period = int(144/int(duration))
    roller = pd.Series.rolling(df['close'].pct_change(),rolling_volatility_period)
    volList = roller.std(ddof=0)
    df['rolling_volatility'] = volList
    return df

# Adding Volatility metric to the code
def add_volatility(df,div_df):
    div_df['volatility']=None
    mean = df['rolling_volatility'].mean()
    std = df['rolling_volatility'].std()
    if df['rolling_volatility'].iloc[-1] < (mean-std):
        div_df['volatility'] = 'Low'
    elif df['rolling_volatility'].iloc[-1] > (mean+std):
        div_df['volatility'] = 'High'
    else :
        div_df['volatility']='Medium'

    return div_df

def find_divergence(df, df_weekly, duration):
    #duration= duration.split()
                                                                                                                                                           
    df = df.reset_index(level=None, drop=False, inplace=False, col_level=0)
    df = add_macd(df)
    df = add_kalman(df)
    df = rolling_volatility(df,duration)
    df = df.fillna(0)
    df,divergence, mintab_m, maxtab_m = add_divergence_macd(df,smoothing_price_pct=0.01, smoothing_macd=0.001)
    
    #print(divergence)
    div_df = {}
    div_df = signal_strength_cosine(df, divergence)
    if(div_df is None):
        return None
    else:
        div_df = add_state(df_weekly,div_df)
        div_df = add_volatility(df, div_df)
        return div_df
