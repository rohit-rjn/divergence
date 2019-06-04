## Divergence Detection

This is a code for divergence Detection <br />
3 Main files as of now: <br />

divergence_signals.py - Code for main logic <br />
get_data.py - to get the tokens data <br />
run_divergence.py - It merges the code of get_data and divergence_signal <br />
```
Input for run_divergence: it takes 2 parameters: (token_list, duration_list) <br />

duration_list - in format: ['1 HOUR', '1-DAY'] <br />
token_list= list of tokens: ([BTC, ETH, LTC, ETH.....]) <br />
```
and outputs signal in list of divergence signals in format:

```
signals = [{'token':'USD_BTC', 'timeframe':'120_min', 'start':now, 'end':now, 'divergence':'bullish'},
           {'token':'USD_ETH', 'timeframe':'120_min', 'start':now, 'end':now, 'divergence':'bearish'},
           {'token':'USD_LTC', 'timeframe':'120_min', 'start':now, 'end':now, 'divergence':'neutral'}]

```

### Example

```
from run_divergence import get_signals

token_list = ['BTC', 'ETH', 'XRP', 'LTC', 'BAB', 'BCH']
duration_list = ['2 HOUR','4 HOUR','8 HOUR','12 HOUR']
signals = get_signals(token_list,duration_list)
print(signals)
```
