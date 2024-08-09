import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 스타일 설정
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

# 데이터 다운로드
end = datetime.now()
start = datetime(end.year - 10, end.month, end.day)
df = yf.download('AAPL', start=start, end=end)

# 데이터 전처리
data = df.filter(['Close'])
dataset = data.values
training_data_len = int(np.ceil(len(dataset) * .95))

# 데이터 스케일링
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# 학습 데이터 생성
train_data = scaled_data[0:int(training_data_len), :]
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# LSTM 모델 생성
model = Sequential()
model.add(LSTM(256, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(128, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# 모델 컴파일 및 학습
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=1, epochs=4)

# 테스트 데이터 생성
test_data = scaled_data[training_data_len - 60:, :]
x_test = []
y_test = dataset[training_data_len:, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# 예측 수행
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# 데이터 시각화
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

plt.figure(figsize=(16, 6))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

# 미래 예측
future_days = 30
last_60_days = data[-60:].values
last_60_days_scaled = scaler.transform(last_60_days)

X_future = []
X_future.append(last_60_days_scaled)
X_future = np.array(X_future)
X_future = np.reshape(X_future, (X_future.shape[0], X_future.shape[1], 1))

future_predictions = []

for _ in range(future_days):
    prediction = model.predict(X_future)
    future_predictions.append(prediction[0])
    new_input = np.append(X_future[0][1:], prediction, axis=0)
    X_future = np.array([new_input])
    X_future = np.reshape(X_future, (X_future.shape[0], X_future.shape[1], 1))

future_predictions = scaler.inverse_transform(future_predictions)

last_date = data.index[-1]
future_dates = pd.date_range(last_date, periods=future_days + 1, inclusive='right')
future_df = pd.DataFrame(future_predictions, index=future_dates, columns=['Prediction'])

plt.figure(figsize=(16, 6))
plt.title('Future Predictions')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD', fontsize=18)
plt.plot(data['Close'], label='Close Price History')
plt.plot(valid['Close'], label='Validation Data')
plt.plot(valid['Predictions'], label='Predictions')
plt.plot(future_df['Prediction'], label='Future Predictions')
plt.legend(['Close Price History', 'Validation Data', 'Predictions', 'Future Predictions'], loc='lower right')
plt.show()
