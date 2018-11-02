# Principle Component Analysis on Hedging futures investing portfolio

## Goal
The goal is getting 42 items of futures historical data from Quandl API then use PCA to generate a hedged portfolio automatically.  

## Prerequisites
Bofore we start, these packages are required for our hedging:
```Python
import quandl
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
```
## Processing Steps
### Data collection:
The steps included getting transaction data from 2014 to 2018 and extracted the final price each day. Since there will be multiple values on the same day according to different futures contracts, we should be selecting the futures price with the largest trading volume. The price of each day is deicided by the futures contract that has the most significant trading volume, because of the more significant the amount of transaction, the added weight for that particular price. 

### Data manipulation and modeling
After extracted the price data, then I imputed missing value and standardized the data so that the model would converge faster. Finally, implement into PCA method to construct the hedge portfolio.
