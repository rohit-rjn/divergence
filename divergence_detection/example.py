from run_divergence import get_signals

token_list = ['BTC', 'ETH', 'XRP', 'LTC', 'BAB', 'BCH', 'EOS', 
				'BAT', 'XLM', 'BNB', 'ADA', 'BSV', 'TRX', 'XTZ', 
				'ATOM', 'ETC', 'NEO', 'XEM', 'MKR', 'ONT', 'LINK', 
				'ZEC', 'VET', 'CRO', 'DOGE', 'QTUM', 'OMG', 'DCR', 
				'HOT', 'WAVES', 'BTT','TUSD', 'LSK', 'NANO', 'REP',
				'BCD', 'RVN', 'ZIL', 'ZRX', 'ICX','XVG', 'PAX', 
				'BTS', 'BCN', 'DGB', 'NPXS', 'HT', 'IOST', 'AE', 
				'KMD', 'ABBD', 'SC', 'ENJ', 'STEEM', 'AOA', 'QBIT', 
				'BTM','MAID', 'THR', 'SOLVE', 'INB', 'KCS','THETA', 
				'WTC', 'STRAT', 'SNT', 'CNX', 'DENT', 'GNT', 'MCO', 
				'ELF', 'DAI', 'ARDR', 'FCT', 'XIN', 'VEST', 'TRUE', 
				'ZEN', 'SAN', 'PAI', 'ARK','MONA', 'DGD', 'GXC', 'WAX',
				'CLAM', 'AION', 'LRC', 'MATIC', 'MANA', 'ELA', 'LOOM',
				'PPT', 'NET' ]                                               
signals = get_signals(token_list,'4 HOUR')
print(signals)