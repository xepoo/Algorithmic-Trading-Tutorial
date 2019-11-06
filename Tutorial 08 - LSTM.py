# 导入 keras 等相关包
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10

CONST_TICKER = '000001.SZ'
TRAIN_TEST_RATIO = 0.9
PREDICT_DAYS = 1

# 选取 date 和 close 两列
df = pd.read_csv('../stock_data/01. IntradayCN/' + CONST_TICKER + '.csv')

data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['trade_date', 'close'])
for i in range(0,len(data)):
    new_data['trade_date'][i] = data['trade_date'][i]
    new_data['close'][i] = data['close'][i]

# setting index
new_data.index = new_data['trade_date']

# #plot
# plt.figure(figsize=(16,8))
# plt.plot(df['close'], label='Close Price history')
# plt.show()

new_data.drop('trade_date', axis=1, inplace=True)

# 分成 train and test
dataset = new_data.values

part_num = int(len(df)*TRAIN_TEST_RATIO)
train = dataset[0:part_num,:]
test = dataset[part_num:,:]

# 构造 x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(PREDICT_DAYS,len(train)):
    x_train.append(scaled_data[i-PREDICT_DAYS:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# 建立 LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

inputs = new_data[len(new_data) - len(test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(PREDICT_DAYS,inputs.shape[0]):
    X_test.append(inputs[i-PREDICT_DAYS:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

#for plotting
train = new_data[:part_num]
test = new_data[part_num:]
test['Predictions'] = closing_price
plt.plot(train['close'])
plt.plot(test[['close','Predictions']])
plt.show()
plt.show()