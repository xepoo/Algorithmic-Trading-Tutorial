import tushare as ts
import pandas as pd
import datetime
import os
import time


pro = ts.pro_api('3313904c0168e7bee0811581a6c668a2236a261475987a71640bb140')

#
def stockPriceIntraday(ticker, folder):
	# Step 1. Get intraday data online
	#intraday = ts.get_hist_data(ticker, ktype='D', pause=1)
	intraday = pro.daily(ts_code=ticker)
	#intraday = ts.pro_bar(ticker, api=pro, freq='5min')
	# Step 2. If the history exists, append
	file = folder+'/'+ticker+'.csv'
	if os.path.exists(file):
		history = pd.read_csv(file, index_col=0)
		intraday.append(history)

	# Step 3. Inverse based on index
	intraday.sort_index(inplace=True)
	intraday.index.name = 'timestamp'

	# Step 4. Save
	intraday.to_csv(file)
	print ('Intraday for ['+ticker+'] got.')

# Step 1. Get and save tickers basic data online from Tushare
#tickersRawData = ts.get_stock_basics()
stock_basic = pro.stock_basic()
dateToday = datetime.datetime.today().strftime('%Y%m%d')
file = '../stock_data/00. TickerListCN/StockBasic_TickerList_'+dateToday+'.csv'
stock_basic.to_csv(file, encoding="utf_8_sig")

stock_company = pro.stock_company(exchange='SZSE')
stock_company = stock_company.append(pro.stock_company(exchange='SSE'))
dateToday = datetime.datetime.today().strftime('%Y%m%d')
file = '../stock_data/00. TickerListCN/StockCompany_TickerList_'+dateToday+'.csv'
stock_company.to_csv(file, encoding="utf_8_sig")

tickers = stock_basic['ts_code'].tolist()
print ('Tickers saved.')
time.sleep(13)
# Step 2. Get stock price min bar (intraday) for all
for i, ticker in enumerate(tickers):
	try:
		print ('Intraday', i, '/', len(tickers))
		stockPriceIntraday(ticker, folder='../stock_data/01. IntradayCN')
		time.sleep(13)
	except:
		pass
print ('Intraday for all stocks got.')



