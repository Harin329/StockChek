import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from statsmodels.tools.eval_measures import rmse
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('Exports/MRNA_Export.csv')
print(df)

df.date = pd.to_datetime(df.date)
df = df.set_index("date")

train = df[:-12]
test = df[-12:]
train = train.drop("1. open", axis=1)
train = train.drop("2. high", axis=1)
train = train.drop("3. low", axis=1)
test = test.drop("1. open", axis=1)
test = test.drop("2. high", axis=1)
test = test.drop("3. low", axis=1)

scaler = MinMaxScaler()
print(train)
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)


n_input = 12
n_features = 2
generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(train, train, length=n_input, batch_size=6)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(200, activation='relu', input_shape=(n_input, n_features)))
model.add(tf.keras.layers.Reshape((2, 100)))
model.add(tf.keras.layers.Dropout(0.15))
model.add(tf.keras.layers.Dense(1))
model.compile(optimizer='adam', loss='mse')


model.fit_generator(generator,epochs=90)


pred_list = []

batch = train[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    pred_list.append(np.array([model.predict(batch)[0][0][0], model.predict(batch)[0][1][0]])) 
    batch = np.append(batch[:,1:,:],[[pred_list[i]]],axis=1)
    
   
df_predict = pd.DataFrame(scaler.inverse_transform(pred_list),
                          index=df[-n_input:].index, columns=['Prediction', 'Prediction2'])

df_test = pd.concat([df,df_predict], axis=1)


plt.figure(figsize=(20, 5))
plt.plot(df_test.index, df_test['4. close'])
plt.plot(df_test.index, df_test['Prediction'], color='r')
plt.xticks(fontsize=18)
plt.yticks(fontsize=16)
plt.show()
plt.plot(df_test.index, df_test['5. volume'])
plt.plot(df_test.index, df_test['Prediction2'], color='y')
plt.show()



pred_actual_rmse = rmse(df_test.iloc[-n_input:, [0]], df_test.iloc[-n_input:, [1]])
print("rmse: ", pred_actual_rmse)



train = df
train = train.drop("1. open", axis=1)
train = train.drop("2. high", axis=1)
train = train.drop("3. low", axis=1)

scaler.fit(train)
train = scaler.transform(train)



n_input = 12
n_features = 2
generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(train, train, length=n_input, batch_size=6)

model.fit_generator(generator,epochs=90)


pred_list = []

batch = train[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):   
    pred_list.append(np.array([model.predict(batch)[0][0][0], model.predict(batch)[0][1][0]])) 
    batch = np.append(batch[:,1:,:],[[pred_list[i]]],axis=1)


from pandas.tseries.offsets import DateOffset
add_dates = [df.index[-1] + DateOffset(days=x) for x in range(0,13) ]
future_dates = pd.DataFrame(index=add_dates[1:],columns=df.columns)


df_predict = pd.DataFrame(scaler.inverse_transform(pred_list),
                          index=future_dates[-12:].index, columns=['Prediction', 'Prediction2'])
df_proj = pd.concat([df,df_predict], axis=1)

plt.figure(figsize=(20, 5))
plt.plot(df_proj.index, df_proj['4. close'])
plt.plot(df_proj.index, df_proj['Prediction'], color='r')
plt.xticks(fontsize=18)
plt.yticks(fontsize=16)
plt.show()
plt.plot(df_proj.index, df_proj['5. volume'])
plt.plot(df_proj.index, df_proj['Prediction2'], color='y')
plt.show()


