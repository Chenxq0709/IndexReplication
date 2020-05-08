import bs4 as bs
import os
import pandas_datareader.data as web
import pickle
import requests
import pandas as pd

def get_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        ticker = ticker.replace(".", "-")
        tickers.append(ticker.replace("\n", ""))
    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)
    print(tickers)
    return tickers

def get_nasdap_tickers():
    table = open('nasdaptickers.txt')
    tickers = []
    for row in table.readlines()[1:]:
        ticker = row.split('\t')[0]
        tickers.append(ticker)
    with open("nasdaptickers.pickle", "wb") as f:
        pickle.dump(tickers, f)
    print(tickers)
    return tickers

# Loop over tickers, get time series data
def get_data_from_yahoo(start, end, reload_list=False, tickerfile=""):
    if reload_list:
        if tickerfile=="nasdaptickers.pickle":
            tickers = get_nasdap_tickers()
        elif tickerfile=="sp500tickers.pickle":
            tickers = get_sp500_tickers()

    else:
        with open(tickerfile, "rb") as f:
            tickers = pickle.load(f)
    if not os.path.exists('stock_dfs'):
            os.makedirs('stock_dfs')
    notfetchable = []
    for ticker in tickers:
        # just in case your connection breaks, we'd like to save our progress!
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            try:
                df = web.get_data_yahoo(ticker, start=start, end=end)
                df.reset_index(inplace=True)
                df.set_index("Date", inplace=True)
    #            df = df.drop("Symbol", axis=1)
                df.to_csv('stock_dfs/{}.csv'.format(ticker))
            except:
                print('Could not fetch '+ ticker)
                notfetch = ticker
                notfetchable.append(notfetch)
        else:
            print('Already have {}'.format(ticker))

    if notfetchable:
        with open("notfetchable.pickle", "wb") as f:
            pickle.dump(tickers, f)


# tickers = get_nasdap_tickers()
#
# with open("nasdap100tickers.txt") as f:
#     tickers = f.read().split('\n')
#
# stockprice = pd.DataFrame()
# for ticker in tickers:
#     try:
#         if len(web.get_data_yahoo(ticker, start="2015-01-01", end="2015-01-04")['Close'])>0 :
#             stockprice[ticker] = web.get_data_yahoo(ticker, start="2015-01-01", end="2020-03-01")['Close']
#             print('load', ticker)
#     except:
#         print('pass', ticker)
#         pass
#
# # nasdap_returns = np.log(data / data.shift(1))
# with open('nasdaq100_stocks_price.pickle', 'wb') as file:
#     pickle.dump(stockprice, file)
# print(stockprice.shape)
# Index symbol
index_symbol = '^NDX'
indexprice = pd.DataFrame()
indexprice['Index'] = web.get_data_yahoo(index_symbol, start="2015-01-01", end="2020-03-01")['Close']
with open('nasdaq100_index.pickle', 'wb') as file:
    pickle.dump(indexprice, file)


print(indexprice.shape)

# Index symbol
index_symbol = '^IXIC'
indexprice = pd.DataFrame()
indexprice['Index'] = web.get_data_yahoo(index_symbol, start="2015-01-01", end="2020-03-01")['Close']
with open('nasdaq_composite_index.pickle', 'wb') as file:
    pickle.dump(indexprice, file)