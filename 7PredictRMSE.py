#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from statsmodels.tools.eval_measures import rmse
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

tf.random.set_seed(1)

symbolList = ['MU']

for sym in symbolList:

  df = pd.read_csv('Exports/' + sym + '_Export.csv')
  print(df)


  # In[9]:


  df.date = pd.to_datetime(df.date)
  df = df.set_index("date")


  # In[10]:


  train = df[:-100]
  test = df[-100:]
  train = train.drop("1. open", axis=1)
  train = train.drop("2. high", axis=1)
  train = train.drop("3. low", axis=1)
  train = train.drop("5. volume", axis=1)
  test = test.drop("1. open", axis=1)
  test = test.drop("2. high", axis=1)
  test = test.drop("3. low", axis=1)
  test = test.drop("5. volume", axis=1)


  # In[11]:


  scaler = MinMaxScaler()
  print(train)
  scaler.fit(train)
  train = scaler.transform(train)
  test = scaler.transform(test)


  # In[12]:


  n_input = 5
  n_features = 1
  generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(train, train, length=n_input, batch_size=64)
  generator_test = tf.keras.preprocessing.sequence.TimeseriesGenerator(test, test, length=n_input, batch_size=5)


  # In[13]:


  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.LSTM(30, activation='relu', input_shape=(n_input, n_features)))
  model.add(tf.keras.layers.Dropout(0.15))
  model.add(tf.keras.layers.Dense(1))
  model.compile(optimizer='adam', loss='mse')


  # In[15]:


  single_step_history = model.fit(generator,epochs=60,validation_data=generator_test,validation_steps=5)


  # In[16]:


  def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    plt.show()


  # In[17]:


  plot_train_history(single_step_history,
                    'Single Step Training and validation loss')


  # In[18]:


  pred_list = []

  batch = test[-n_input:].reshape((1, n_input, n_features))

  for i in range(n_input):   
      pred_list.append(model.predict(batch)[0]) 
      batch = np.append(batch[:,1:,:],[[pred_list[i]]],axis=1)
      


  # In[19]:


  df_predict = pd.DataFrame(scaler.inverse_transform(pred_list),
                            index=df[-n_input:].index, columns=['Prediction'])

  df_test = pd.concat([df,df_predict], axis=1)


  # In[20]:


  plt.figure(figsize=(20, 5))
  plt.plot(df_test.index, df_test['4. close'])
  plt.plot(df_test.index, df_test['Prediction'], color='r')
  plt.xticks(fontsize=18)
  plt.yticks(fontsize=16)
  plt.show()


  # In[21]:


  pred_actual_rmse = rmse(df_test.iloc[-n_input:, [0]], df_test.iloc[-n_input:, [1]])
  print("rmse: ", pred_actual_rmse)


  # In[22]:


  train = df
  train = train.drop("1. open", axis=1)
  train = train.drop("2. high", axis=1)
  train = train.drop("3. low", axis=1)
  train = train.drop("5. volume", axis=1)


  # In[23]:


  scaler.fit(train)
  train = scaler.transform(train)


  # In[24]:


  n_input = 5
  n_features = 1
  generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(train, train, length=n_input, batch_size=32)


  # In[ ]:


  model.fit(generator,epochs=60)


  # In[113]:


  pred_list = []

  batch = train[-n_input:].reshape((1, n_input, n_features))

  for i in range(n_input):   
      pred_list.append(model.predict(batch)[0]) 
      batch = np.append(batch[:,1:,:],[[pred_list[i]]],axis=1)


  # In[114]:


  from pandas.tseries.offsets import DateOffset
  add_dates = [df.index[-1] + DateOffset(days=x) for x in range(0,n_input+1) ]
  future_dates = pd.DataFrame(index=add_dates[1:],columns=df.columns)


  # In[115]:


  df_predict = pd.DataFrame(scaler.inverse_transform(pred_list),
                            index=future_dates[-n_input:].index, columns=['Prediction'])
  df_proj = pd.concat([df,df_predict], axis=1)


  # In[116]:


  plt.figure(figsize=(20, 5))
  plt.plot(df_proj.index, df_proj['4. close'])
  plt.plot(df_proj.index, df_proj['Prediction'], color='r')
  plt.xticks(fontsize=18)
  plt.yticks(fontsize=16)
  plt.show()
