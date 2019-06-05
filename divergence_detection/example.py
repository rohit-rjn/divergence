from run_divergence import get_signals

token_list = ['BTC', 'ETH', 'XRP', 'LTC', 'BAB', 'BCH', 'EOS', 
				'BAT', 'XLM', 'BNB', 'ADA', 'BSV', 'TRX', 'XTZ', 
				'ATOM', 'ETC', 'NEO', 'XEM', 'MKR', 'ONT', 'LINK', 
				'ZEC', 'VET', 'CRO', 'DOGE', 'QTUM', 'OMG', 'DCR', 
				'HOT', 'WAVES', 'BTT','TUSD', 'LSK', 'NANO', 'REP',
				'BCD', 'RVN', 'ZIL', 'ZRX', 'ICX','XVG', 'PAX', 
				'BTS', 'BCN', 'DGB', 'NPXS', 'HT', 'IOST', 'AE', 
				'KMD', 'SC', 'ENJ', 'STEEM', 'AOA', 'QBIT', 
				'BTM','MAID', 'THR', 'SOLVE', 'KCS','THETA', 
				'WTC', 'STRAT', 'SNT', 'CNX', 'DENT', 'GNT', 'MCO', 
				'ELF', 'DAI', 'ARDR', 'FCT', 'XIN', 'VEST', 'TRUE', 
				'ZEN', 'SAN', 'PAI', 'ARK','MONA', 'DGD', 'GXC', 'WAX',
				'CLAM', 'AION', 'LRC', 'MATIC', 'MANA', 'ELA', 'LOOM',
				'PPT', 'NET' ]                                               
duration_list = ['2 HOUR','4 HOUR','8 HOUR','12 HOUR']
signals = get_signals(token_list,duration_list)
print(signals)