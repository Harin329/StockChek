# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

symbolList = ['MU']
lookback = 100

# Try different window lengths (timesteps fed into network)
# Try adding dense layers, or multiple LSTM layers, or fewer LTSM nodes
# Try different optimizers, with various learning rates
# Look for additional datapoints to feed into the network
# How much data do you have? You may need more to get a good prediction
# Try different offsets for the Y variable, how many timesteps do you need to be able to predict out for your specific problem?

for s in symbolList:
    df = pd.read_csv('Exports/' + s + '_Export.csv')
    close_prices = df.loc[:,'4. close']

    train_data = close_prices[:5000].values.reshape(-1, 1)
    test_data = close_prices[5000:].reset_index()

    # Feature Scaling
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(train_data)

    # Creating a data structure with lookback timesteps and 1 output
    X_train = []
    y_train = []
    for i in range(lookback, len(train_data)):
        X_train.append(training_set_scaled[i-lookback:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshaping
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Part 2 - Building the RNN

    # Importing the Keras libraries and packages
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import LSTM
    from tensorflow.keras.layers import Dropout

    # Initialising the RNN
    regressor = Sequential()

    # Adding the first LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 200, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))

    # Adding a second LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 200, return_sequences = True))
    regressor.add(Dropout(0.2))


    # # Adding a third LSTM layer and some Dropout regularisation
    # regressor.add(LSTM(units = 200, return_sequences = True))
    # regressor.add(Dropout(0.2))

    # Adding a fourth LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 200))
    regressor.add(Dropout(0.2))

    # Adding the output layer
    regressor.add(Dense(units = 1))

    # Compiling the RNN
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

    # Fitting the RNN to the Training set
    regressor.fit(X_train, y_train, epochs = 100, batch_size = 128)

    # Part 3 - Making the predictions and visualising the results

    # Getting the real stock price of 2017
    #dataset_test = pd.read_csv('Exports/' + s + '_Export.csv')
    #real_stock_price = df.loc[:,'4. close'].values.reshape(-1, 1)

    # Getting the predicted stock price of 2017
    #dataset_total = pd.concat((df['4. close'], dataset_test['4. close']), axis = 0)
    dataset_total = df['4. close']
    #inputs = dataset_total[len(dataset_total) - len(dataset_test) - lookback:].values
    inputs = dataset_total[len(dataset_total) - lookback - len(test_data):].values
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)
    print(inputs.shape)
    X_test = []
    for i in range(lookback, lookback+len(test_data)):
        X_test.append(inputs[i-lookback:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_stock_price = regressor.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)


    # Visualising the results
    #plt.plot(real_stock_price, color = 'red', label = 'Real Stock Price')
    plt.plot(test_data.loc[:, '4. close'], color = 'red', label = 'Real Stock Price')
    plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()