# JustQuant_futures_hedging
The goal is getting 42 items of futures historical data from Quandl API then use PCA to generate a hedged portfolio automatically.  

The steps included getting transaction data from 2014 to 2018 and extracted the final price each day. Since there will be multiple values on the same day according to different futures contracts, we should be selecting the futures price with the largest trading volume. The price of each day is deicided by the futures contract that has the most significant trading volume, because of the more significant the amount of transaction, the added weight for that particular price. 

After extracted the price data, then we implement into PCA method to construct the hedge portfolio.
