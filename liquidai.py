import numpy as np
import pandas as pd
import quandl
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from matplotlib import pyplot as plt
import sklearn.preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import math

# Import the data

start = "2014-01-01"
end = "2018-12-31"

# from Wiki Continuous Futures, U.S. Treasury Bond Futures, Continuous Contract #1 (US1) (Front Month)
df1 = quandl.get("CHRIS/CME_US1", authtoken="XPy1DEwjeSK8sZYKn9nE", start_date=start, end_date=end)
# from the US Federal Reserve Data Releases, Federal Funds Effective Rate Daily
df2 = quandl.get("FED/RIFSPFF_N_D", authtoken="XPy1DEwjeSK8sZYKn9nE", start_date=start, end_date=end)

# Clean the data

## Combine the two data frames
data = pd.concat([df1, df2], axis=1)

## Drop the weekend values where we don't have price data
data.dropna(thresh=8,inplace=True)

## We've handled the obvious information to drop. Now we move on to the more nuanced imputations
## We start by reindexing the dataframe to get integer indices
data.reset_index(inplace=True)

## The Change column is the difference between a given day and the trading day's prior Settle price, so we fill in those
data.loc[(pd.isnull(data['Change'])), 'Change'] = data['Settle'].shift(1) - data['Settle']

## We'll set the Open price to the Settle price of the previous trading day
data.loc[(pd.isnull(data['Open'])), 'Open'] = data['Settle'].shift(1)

## We'll use interpolation to make a very rough guess as to what the last trade price would be on day's we don't have one
data['Last'].interpolate(method='polynomial', order=3, inplace=True)

## Reset the index again to the datetimes
data.set_index('Date', inplace=True)

## Change the order of the columns (for easier operations later)
data = data[['Open', 'High', 'Low', 'Last', 'Change', 'Volume', 'Previous Day Open Interest', 'Value', 'Settle']]

# Prepare the data

## We'll turn this problem into a supervised learning problem and look at a prediction window of 3 months (~63 trading days)
## Note: this is adopted from an earlier project from stackoverflow

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

## Here we make the data into a supervised learning problem and drop the unneeded columns for the current timestep
data_new = series_to_supervised(data.values, n_in=63, n_out=1)
data_new.drop(columns=data_new.columns[[i for i in range(len(data_new.columns)-9, len(data_new.columns)-1)]], inplace=True)

## Make the scaler and scale the data
scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data_new)

## Split the data into train and test sets
#trainX, testX, trainY, testY = train_test_split(data_scaled[:,:-1], data_scaled[:,-1], test_size = 0.3, random_state = 0)
split = data.shape[0] - 400
train = data_scaled[:split,:]
test = data_scaled[split:,:]

## Split the train and test sets into inputs and outputs
trainX, trainY = train[:,:-1], train[:,-1]
testX, testY = test[:,:-1], test[:,-1]

## Reshape the outputs for the LSTM
trainX = trainX.reshape(trainX.shape[0], 1, trainX.shape[1])
testX = testX.reshape(testX.shape[0], 1, testX.shape[1])

# Implementing the model

## Design the network
model = Sequential()
model.add(LSTM(50, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

## Fit the network
history = model.fit(trainX, trainY, epochs=50, batch_size=72, validation_data=(testX, testY), verbose=2, shuffle=False)

# Predict price action
predY = model.predict(testX)
testX = testX.reshape((testX.shape[0], testX.shape[2]))

## Invert scaling for forecast
inv_yhat = np.concatenate((predY, testX[:,:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

## Invert scaling for actual
testY = testY.reshape((len(testY), 1))
inv_y = np.concatenate((testY, testX[:,:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

## Calculate RMSE
rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

## Simulate trading

capital = [10000]
returns = [0]
maxb = 10000
minb = 10000
maxdrawdown = 0
for i in range(1, len(inv_yhat)):
    if inv_yhat[i] > inv_y[i-1]: # predict long
        capital.append(((inv_y[i] - inv_y[i-1]) / inv_y[i-1]) * capital[i-1] + capital[i-1])
        returns.append((inv_y[i] - inv_y[i-1]) / inv_y[i-1])
    else: # predict short
        capital.append(((inv_y[i-1] - inv_y[i]) / inv_y[i-1]) * capital[i-1] + capital[i-1])
        returns.append((inv_y[i-1] - inv_y[i]) / inv_y[i-1])
    if capital[i] > maxb:
        maxb = capital[i]
    elif capital[i] < minb:
        minb = capital[i]
    if (1 - minb/maxb) > maxdrawdown:
        maxdrawdown = 1 - minb/maxb
        
## Analyze performance
        
analysis = pd.DataFrame({'Capital': capital, 'Pct. Change': returns})
analysis['StdDev'] = analysis['Pct. Change'].std()

# Sortino Ratio calculations

MAR = .03
downside_stddev = math.sqrt(((analysis[analysis['StdDev'] < MAR]['StdDev'] ** 2).sum()))

annret = [1+x for x in returns]
annualized_return = 1
for i in annret:
    annualized_return = annualized_return * i
annualized_return -= 1

sortino = float((annualized_return - MAR) / downside_stddev)
print(f'Sortino Ratio: {sortino}')
print(f'Max Drawdown: {maxdrawdown}')
print(f'Annualized Return: {annualized_return}')

## Visualization

plt.subplot(2, 1, 1)
plt.plot(capital, label='AUM')
plt.title('AUM')

plt.subplot(2, 1, 2)
plt.plot(inv_yhat, label='Prediction')
plt.plot(inv_y, label='Actual')
plt.legend(loc='best')
plt.show()