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
# Function to Add the MACD to DataFrame 
def add_macd(df,fast,slow,signal):
    output = talib.MACD(df['close'].values, fastperiod=fast, slowperiod=slow, signalperiod=signal)
    macd = 'macd'+str(fast)+str(slow)+str(signal)
    macdsignal = macd+'signal'
    macdhist = macd+'hist'
    df[macd] = output[0]
    df[macdsignal] = output[1]
    df[macdhist] = output[2]
    return df

#Function to add MFI indicator
def add_mfi(df):
    mfi = talib.MFI( df['high'].values, df['low'].values,df['close'].values,df['volume'].values, timeperiod=14)
    df['mfi']=mfi
    return df

# Add RSI indicator 
def add_rsi(df):
    rsi = talib.RSI(df['close'].values, timeperiod=14)
    df['rsi']=rsi
    return df

def STOCH(h, l, c, fastk_period, slowk_period, slowd_period):
    #hh = pd.rolling_max(h, fastk_period, min_periods=fastk_period)
    hh = h.rolling(window=fastk_period).max()
    #ll = pd.rolling_min(l, fastk_period, min_periods=fastk_period)
    ll = l.rolling(window=fastk_period).min()
    fast_k = 100 * (c - ll) / (hh - ll)
    #slow_k = pd.ewma(fast_k, span=slowk_period, min_periods=slowk_period)
    slow_k = pd.Series.ewm(fast_k, span=slowk_period).mean()
    #slow_d = pd.ewma(slow_k, span=slowd_period, min_periods=slowd_period)
    slow_d = pd.Series.ewm(slow_k, span=slowd_period).mean()
    return slow_k, slow_d

#Add StochRSI indicator
def add_stochrsi(df):
    #print(len())
    fastk, fastd = STOCH(df['rsi'],df['rsi'],df['rsi'], 14, 3, 3)
    temp = [0]*17
    #k = np.append(temp,fastk)
    #d = np.append(temp,fastd)
    df['fastk']=fastk
    df['fastd']=fastd
    return df

#Add Obv 
def add_obv(df):
    obv = talib.OBV(df['close'].values, df['volume'].values)
    df['obv']=obv
    return df

# Obv Oscillator 
def add_obv_osc(df):
    #obv = df['obv']
    obv_osc = df['obv'] - df['obv'].ewm(span=20,min_periods=0,adjust=False,ignore_na=False).mean()
    df['obv_osc']=obv_osc
    return df

# Code for finding Extremas

def peakdet(v, delta, start, x = None,):

    maxtab = []
    mintab = []
    delay=[]
    if x is None:
        x = arange(len(v))
    
    v = asarray(v)
    
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    
    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    
    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN
    
    lookformax = True
    
    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        
        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos+start, mx,i-mxpos))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos+start, mn,i-mnpos))
                mx = this
                mxpos = x[i]
                lookformax = True

    return array(maxtab),array(mintab)

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

# Code to find the Nearest Minima 
def check_minima(prev_local_minima,new_local_minima, df, mintab, flag=0):
    start= prev_local_minima[0]
    end= new_local_minima[0]
    local_minima = [] #None
    local_minima_list = []
    current_minima =[] # None
    j=0
    k=0
    for i in range(1,len(mintab)):
        
        if mintab[i-1][0]<= start and mintab[i][0] >= start and flag==0:
            local_minima = mintab[i-1]
            flag = 1
            #print(local_minima_mfi)
        if(mintab[i][0] >= start and mintab[i][0]<= end) and flag ==0:
            local_minima = mintab[i]
            flag = 1
            #print(local_minima_mfi)
        if (mintab[i][0] >= start and mintab[i][0]<= end) and flag ==1:
            if (mintab[i][0] - start) < (start - mintab[i-1][0]):
                local_minima=mintab[i]
                
        if mintab[i-1][0]<= end and mintab[i][0] >= end:
            current_minima = mintab[i]
            #print(local_minima_mfi)
    return local_minima, current_minima

def check_maxima(prev_local_maxima,new_local_maxima, df, maxtab, flag=0):
    start= prev_local_maxima[0]
    end= new_local_maxima[0]
    local_maxima = []
    local_maxima_list = []
    current_maxima = []
    j=0
    k=0
    for i in range(1,len(maxtab)):
        
        if maxtab[i-1][0]<= start and maxtab[i][0] >= start and flag==0:
            local_maxima = maxtab[i-1]
            flag = 1
            #print(local_maxima_mfi)
        if(maxtab[i][0] >= start and maxtab[i][0]<= end) and flag ==0:
            local_maxima = maxtab[i]
            flag = 1
        if (maxtab[i][0] >= start and maxtab[i][0]<= end) and flag ==1:
            if (maxtab[i][0]-start) < (start-maxtab[i-1][0]):
                local_maxima=maxtab[i]
            #print(local_maxima_mfi)
        if maxtab[i-1][0]<= end and maxtab[i][0] >= end:
            current_maxima_mfi = maxtab[i]
            #print(local_maxima_mfi)
    return local_maxima, current_maxima


def add_divergence(df):
    macdhist = 'macd12269hist'
    start = 0
    end = len(df)
    delta_m = 0.0001*df['close'].iloc[1]
    maxtab_m1,mintab_m1 = peakdet(df[macdhist].values, delta_m,start, x = None)
    delta_m2= 0.00001*df['close'].iloc[1]
    maxtab_m2,mintab_m2 = peakdet(df[macdhist].values, delta_m2,start, x = None)
    delta_sr = 3
    maxtab_sr,mintab_sr = peakdet(df['fastk'].values, delta_sr,start, x = None)
    delta_r = 2
    maxtab_r,mintab_r = peakdet(df['rsi'].values, delta_sr,start, x = None)
    delta_i = 1
    maxtab_i,mintab_i = peakdet(df['mfi'].values, delta_i,start, x = None)
    delta_o = 0.01*df['volume'].iloc[1]
    maxtab_o,mintab_o = peakdet(df['obv_osc'].values, delta_o,start, x = None)
    
    div_m = []
    div_sr = []
    div_r = []
    div_i = []
    div_o = []

    if maxtab_m2[-1][0] > end-4:
        if maxtab_m1[-1][0]!=maxtab_m2[-1][0]:
            curr = maxtab_m2[-1]
            prev = maxtab_m1[-1]
        else:
            curr = maxtab_m2[-1]
            prev = maxtab_m1[-2]
        # Regular bear Div - MACD lower high and Price higher high
        #print(prev[1],curr[1],df['close'].iloc[int(prev[0])] ,df['close'].iloc[int(curr[0])])
        if prev[1] > curr[1]:
            #df_close_range_low = df['close'].iloc[int(prev[0])] + 0.01*df['close'].iloc[int(prev[0])]
            #df_close_range_high = df['close'].iloc[int(prev[0])] - 0.01*df['close'].iloc[int(prev[0])]
            #if  df_close_range_low < df['close'].iloc[int(curr[0])] < df_close_range_high:
            #    div_m = (prev[0],curr[0],5)
            
            if df['close'].iloc[int(prev[0])] < df['close'].iloc[int(curr[0])]:
                div_m = (prev[0],curr[0],3)

        
        prev_rsi, curr_rsi = check_maxima(prev,curr, df, maxtab_r, flag=0)
        if len(prev_rsi)>0 and len(curr_rsi)>0:
            if prev_rsi[1] > curr_rsi[1]:
                if df['close'].iloc[int(prev_rsi[0])] < df['close'].iloc[int(curr_rsi[0])]:
                    div_r = (prev_rsi[0],curr_rsi[0],3)
                    
        prev_srsi, curr_srsi = check_maxima(prev,curr, df, maxtab_sr, flag=0)
        if len(prev_srsi)>0 and len(curr_srsi)>0:
            if prev_srsi[1] > curr_srsi[1]:
                if df['close'].iloc[int(prev_srsi[0])] < df['close'].iloc[int(curr_srsi[0])]:
                    div_sr = (prev_srsi[0],curr_srsi[0],3)
                    #df['end'].iloc[end]=6
        prev_mfi, curr_mfi = check_maxima(prev,curr, df, maxtab_i, flag=0)
        if len(prev_mfi)>0 and len(curr_mfi)>0:
            if prev_mfi[1] > curr_mfi[1]:
                if df['close'].iloc[int(prev_mfi[0])] < df['close'].iloc[int(curr_mfi[0])]:
                    div_i = (prev_mfi[0],curr_mfi[0],3)
        
        prev_obv, curr_obv = check_maxima(prev,curr, df, maxtab_o, flag=0)
        if len(prev_obv)>0 and len(curr_obv)>0:
            if prev_obv[1] > curr_obv[1]:
                if df['close'].iloc[int(prev_obv[0])] < df['close'].iloc[int(curr_obv[0])]:
                    div_o = (prev_obv[0],curr_obv[0],3)
                    
        #Hidden Bear Div - MACD higher high, and Price Lower high
        if prev[1] < curr[1]:
            if df['close'].iloc[int(prev[0])] > df['close'].iloc[int(curr[0])]:
                div_m = (prev[0],curr[0],4)

        prev_rsi, curr_rsi = check_maxima(prev,curr, df, maxtab_r, flag=0)
        if len(prev_rsi)>0 and len(curr_rsi)>0:
            if prev_rsi[1] < curr_rsi[1]:
                if df['close'].iloc[int(prev_rsi[0])] > df['close'].iloc[int(curr_rsi[0])]:
                    div_r = (prev_rsi[0],curr_rsi[0],4)
        
        prev_srsi, curr_srsi = check_maxima(prev,curr, df, maxtab_sr, flag=0)
        if len(prev_srsi)>0 and len(curr_srsi)>0:
            if prev_srsi[1] < curr_srsi[1]:
                if df['close'].iloc[int(prev_srsi[0])] > df['close'].iloc[int(curr_srsi[0])]:
                    div_sr = (prev_srsi[0],curr_srsi[0],4)

        prev_mfi, curr_mfi = check_maxima(prev,curr, df, maxtab_i, flag=0)
        if len(prev_mfi)>0 and len(curr_mfi)>0:
            if prev_mfi[1] < curr_mfi[1]:
                if df['close'].iloc[int(prev_mfi[0])] > df['close'].iloc[int(curr_mfi[0])]:
                    div_i = (prev_mfi[0],curr_mfi[0],4)
        
        prev_obv, curr_obv = check_maxima(prev,curr, df, maxtab_o, flag=0)
        if len(prev_obv)>0 and len(curr_obv)>0:
            if prev_obv[1] < curr_obv[1]:
                if df['close'].iloc[int(prev_obv[0])] < df['close'].iloc[int(curr_obv[0])]:
                    div_o = (prev_obv[0],curr_obv[0],4)
    
    if mintab_m2[-1][0]> end-4:
        if mintab_m1[-1][0]!=mintab_m2[-1][0]:
            curr = mintab_m2[-1]
            prev = mintab_m1[-1]
        else:
            curr = mintab_m2[-1]
            prev = mintab_m1[-2]
        # Regular bull Div - MACD higher low, and price lower low
        if prev[1] < curr[1]:
            #df_close_range_low = df['close'].iloc[int(prev[0])] + 0.01*df['close'].iloc[int(prev[0])]
            #df_close_range_high = df['close'].iloc[int(prev[0])] - 0.01*df['close'].iloc[int(prev[0])]
            #if  df_close_range_low > df['close'].iloc[int(curr[0])] < df_close_range_high:
            #    div_m = (prev[0],curr[0],6)
            
            if df['close'].iloc[int(prev[0])] > df['close'].iloc[int(curr[0])]:
                div_m = (prev[0],curr[0],1)

                
        prev_rsi, curr_rsi = check_minima(prev,curr, df, mintab_r, flag=0)
        if len(prev_rsi)>0 and len(curr_rsi)>0:
            if prev_rsi[1] < curr_rsi[1]:
                if df['close'].iloc[int(prev_rsi[0])] > df['close'].iloc[int(curr_rsi[0])]:
                    div_r = (prev_rsi[0],curr_rsi[0],1) 
        prev_srsi, curr_srsi = check_minima(prev,curr, df, mintab_sr, flag=0)
        if len(prev_srsi)>0 and len(curr_srsi)>0:
            if prev_srsi[1] < curr_srsi[1]:
                if df['close'].iloc[int(prev_srsi[0])] > df['close'].iloc[int(curr_srsi[0])]:
                    div_sr = (prev_srsi[0],curr_srsi[0],1)

        prev_mfi, curr_mfi = check_minima(prev,curr, df, mintab_i, flag=0)
        if len(prev_mfi)>0 and len(curr_mfi)>0:
            if prev_mfi[1] < curr_mfi[1]:
                if df['close'].iloc[int(prev_mfi[0])] > df['close'].iloc[int(curr_mfi[0])]:
                    div_i = (prev_mfi[0],curr_mfi[0],1)
        prev_obv, curr_obv = check_minima(prev,curr, df, maxtab_o, flag=0)
        if len(prev_obv)>0 and len(curr_obv)>0:
            if prev_obv[1] < curr_obv[1]:
                if df['close'].iloc[int(prev_obv[0])] > df['close'].iloc[int(curr_obv[0])]:
                    div_o = (prev_obv[0],curr_obv[0],1)
        
        # Hidden Bull Div - MACD lower Low, price higher low
        if prev[1] > curr[1]:
            if df['close'].iloc[int(prev[0])] < df['close'].iloc[int(curr[0])]:
                div_m = (prev[0],curr[0],2)

        
        prev_rsi, curr_rsi = check_minima(prev,curr, df, mintab_r, flag=0)
        if len(prev_rsi)>0 and len(curr_rsi)>0:
            if prev_rsi[1] > curr_rsi[1]:
                if df['close'].iloc[int(prev_rsi[0])] < df['close'].iloc[int(curr_rsi[0])]:
                    div_r = (prev_rsi[0],curr_rsi[0],2)
                    
        prev_srsi, curr_srsi = check_minima(prev,curr, df, mintab_sr, flag=0)
        if len(prev_srsi)>0 and len(curr_srsi)>0:
            if prev_srsi[1] > curr_srsi[1]:
                if df['close'].iloc[int(prev_srsi[0])] < df['close'].iloc[int(curr_srsi[0])]:
                    div_sr = (prev_srsi[0],curr_srsi[0],2)

        prev_mfi, curr_mfi = check_minima(prev,curr, df, mintab_i, flag=0)
        if len(prev_mfi)>0 and len(curr_mfi)>0:
            if prev_mfi[1] > curr_mfi[1]:
                if df['close'].iloc[int(prev_mfi[0])] < df['close'].iloc[int(curr_mfi[0])]:
                    div_i = (prev_mfi[0],curr_mfi[0],2)
        prev_obv, curr_obv = check_minima(prev,curr, df, maxtab_o, flag=0)
        if len(prev_obv)>0 and len(curr_obv)>0:
            if prev_obv[1] > curr_obv[1]:
                if df['close'].iloc[int(prev_obv[0])] < df['close'].iloc[int(curr_obv[0])]:
                    div_o = (prev_obv[0],curr_obv[0],2)
    return div_m,div_r,div_sr,div_i, div_o

def find_divergence_second(df, macd_string):
    #df2 = df[start:end]
    start = 0
    end = len(df)
    delta_m = 0.001*df['close'].iloc[1]
    maxtab_m1,mintab_m1 = peakdet(df[macd_string].values, delta_m,start, x = None)
    delta_m2= 0.0001*df['close'].iloc[1]
    maxtab_m2,mintab_m2 = peakdet(df[macd_string].values, delta_m2,start, x = None)
    div_m = []
    if maxtab_m2[-1][0] > start: #(end-4):
        if maxtab_m1[-1][0]!=maxtab_m2[-1][0]:
            curr = maxtab_m2[-1]
            prev = maxtab_m1[-1]
        else:
            curr = maxtab_m2[-1]
            prev = maxtab_m1[-2]
        # Regular bear Div - MACD lower high and Price higher high
        #print(curr[1],prev)
        if prev[1] > curr[1]:
            if df['close'].iloc[int(prev[0])] < df['close'].iloc[int(curr[0])]:
                div_m = (prev[0],curr[0],3)
                #df['end'].iloc[end]=3
                
        
        #Hidden Bear Div - MACD higher high, and Price Lower high
        if prev[1] < curr[1]:
            if df['close'].iloc[int(prev[0])] > df['close'].iloc[int(curr[0])]:
                div_m = (prev[0],curr[0],4)
                #df['end'].iloc[end]=4
        
    if mintab_m2[-1][0] > start: #(end-4):
        if mintab_m1[-1][0]!=mintab_m2[-1][0]:
            curr = mintab_m2[-1]
            prev = mintab_m1[-1]
        else:
            curr = mintab_m2[-1]
            prev = mintab_m1[-2]
        # Regular bull Div - MACD higher low, and price lower low
        if prev[1] < curr[1]:
            if df['close'].iloc[int(prev[0])] > df['close'].iloc[int(curr[0])]:
                div_m = (prev[0],curr[0],1)
                #df['end'].iloc[end]=1
        
        # Hidden Bull Div - MACD lower Low, price higher low
        if prev[1] > curr[1]:
            if df['close'].iloc[int(prev[0])] < df['close'].iloc[int(curr[0])]:
                div_m = (prev[0],curr[0],2)
                #df['end'].iloc[end]=2
                
    return div_m

def signal_to_ts(df, div_m,div_r,div_sr,div_i,div_o):
    #Step 1 Convert signal to Vectors
    vector_macd = []
    vector_price = []
    start = []
    end = []
    signal_type = []
    stochRSI = []
    MFI = []
     
    for i in range(0,len(div_m)):
        y_macd = df['macdhist'].loc[div_m[i][1]] - df['macdhist'].loc[div_m[i][0]]
        x_macd = div_m[i][1] - div_m[i][0]
        vector_macd.append([x_macd,y_macd])
        x_price = div_m[i][1] - div_m[i][0]
        y_price = df['close'].loc[div_m[i][1]] - df['close'].loc[div_m[i][0]]
        vector_price.append([x_price,y_price])
    
        start.append(df['date'].loc[div_m[i][0]])
        end.append(df['date'].loc[div_m[i][1]])
        if div_m[i][2]==1:
            signal_type.append('Bullish')
        elif div_m[i][2]==2:
            signal_type.append('Hidden Bullish')
        elif div_m[i][2]==3:
            signal_type.append('Bearish')
        elif div_m[i][2]==4:
            signal_type.append('Hidden Bearish')
        elif div_m[i][2]==5:
            signal_type.append('Exagerated Bear')
        elif div_m[i][2]==6:
            signal_type.append('Exagerated Bull')  

        if div_r[i]:
            RSI.append(True)
        else:
            RSI.append(False)
        if div_i[i]:
            MFI.append(True)
        else:
            MFI.append(False)
        if div_sr[i]:
            stochRSI.append(True)
        else:
            stochRSI.append(False)

    signal_cosine = []
    for i in range(0, len(vector_macd)):

        quality = spatial.distance.cosine(vector_macd[i],vector_price[i])
        
        signal_cosine.append(quality)

    df_signal = pd.DataFrame()
    df_signal['start']= start
    df_signal['end'] = end
    df_signal['type']= signal_type
    df_signal['cosine']=signal_cosine
    df_signal['StochRSI']=StochRSI
    df_signal['MFI']=MFI
    if len(df_signal)>0:
        return df_signal.iloc[-1]
    else:
        return None

def divergence_ts(df,div_m,div_r,div_sr,div_i, div_o, div_macd):
    start = 0
    end = 0
    signal_type = 0
    RSI = 0
    stochRSI = 0
    MFI = 0
    obv_osc = 0
    macd_20_50_9 = 0
    macd_50_100_20 = 0
    macd_50_100_9 = 0
    df_signal = {} #pd.DataFrame()
    #print(div_m)
    #for i in range(len(div_m)):
    if div_m:
        start = df['date'].loc[div_m[0]]
        end = df['date'].loc[div_m[1]]
        if div_m[2]==1:
            signal_type = 'Bullish'
        elif div_m[2]==2:
            signal_type = 'Hidden Bullish'
        elif div_m[2]==3:
            signal_type = 'Bearish'
        elif div_m[2]==4:
            signal_type = 'Hidden Bearish'
        elif div_m[2]==5:
            signal_type = 'Exagerrated Bear'
        elif div_m[2]==6:
            signal_type = 'Exagerrated Bull'


        if div_r:
            RSI = True
        else:
            RSI = False
        if div_sr:
            stochRSI = True
        else:
            stochRSI = False
        if div_i:
            MFI = True
        else:
            MFI = False
        if div_o:
            obv_osc = True
        else:
            obv_osc = False
        

        df_signal['start']          = start
        df_signal['end']            = end
        df_signal['type']           = signal_type
        df_signal['stochrsi_div']   = stochRSI
        df_signal['mfi_div']        = MFI
        df_signal['rsi_div']        = RSI
        df_signal['obv_div']        = obv_osc
        df_signal['cosine']         = 0
        
        #df_signal['macd_20_50_9']   = {'start':df['date'].loc[div_macd[0][0]],'end':df['date'].loc[div_macd[0][1]]}
        #df_signal['macd_50_100_20'] = {'start':df['date'].loc[div_macd[1][0]],'end':df['date'].loc[div_macd[1][1]]}
        #df_signal['macd_50_100_9']  = {'start':df['date'].loc[div_macd[2][0]],'end':df['date'].loc[div_macd[2][1]]}
        if div_macd[0]:
            df_signal['macd_20_50_9'] = {'start':df['date'].loc[div_macd[0][0]],'end':df['date'].loc[div_macd[0][1]]}
        else:
            df_signal['macd_20_50_9'] = False
        if div_macd[1]:
            df_signal['macd_50_100_20'] = {'start':df['date'].loc[div_macd[2][0]],'end':df['date'].loc[div_macd[0][1]]}
        else:
            df_signal['macd_50_100_20'] = False
        if div_macd[2]:
            df_signal['macd_50_100_9'] = {'start':df['date'].loc[div_macd[2][0]],'end':df['date'].loc[div_macd[0][1]]}
        else:
            df_signal['macd_50_100_9'] = False

    #print(len(df_signal))
    if len(df_signal)>0:
        return df_signal
    else:
        return None


def find_divergence(df, df_weekly, duration):
    #duration= duration.split()
                                                                                                                                                           
    df = df.reset_index(level=None, drop=False, inplace=False, col_level=0)
    df = add_macd(df,12,26,9)
    #df = add_macd(df)
    #df = add_kalman(df)
    df = rolling_volatility(df,duration)
    df = add_rsi(df)
    df = add_stochrsi(df)
    df = add_mfi(df)
    df = add_obv(df)
    df = add_obv_osc(df)
    df = df.fillna(0)
    
    div_m,div_r,div_sr,div_i, div_o = add_divergence(df)
    macd_var = [[20,50,9],[50,100,20],[50,100,9]]
    div_macd = [] 
    for i in range(len(macd_var)):
        df_macd = add_macd(df,macd_var[i][0],macd_var[i][1],macd_var[i][2])
        macd_string = 'macd'+str(macd_var[i][0])+str(macd_var[i][1])+str(macd_var[i][2])
        try: 
            div_macd.append(find_divergence_second(df_macd, macd_string))
        except:
            div_macd.append([])
    
    div_df = {}
    div_df = divergence_ts(df,div_m,div_r,div_sr,div_i, div_o, div_macd)
    #print(div_df)
    if(div_df is None):
        return None
    else:
        div_df = add_state(df_weekly,div_df)
        div_df = add_volatility(df, div_df)
        return div_df


