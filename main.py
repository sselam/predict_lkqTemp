#rev 00
#read lkq temperatures (around 100 days) with rpm, group data by day (day starts at 7am),
#clean data - consider max temp if 1. RPM is at or above the acceptable minimum value (min_rpm set at 300), and if delta of temp value (max-min) is more than 20 only)),
#and classify into training/test (85%/15%))
#predict next 15% (the same number as test days) days

import configparser
import json
import math
import os
from functools import reduce

import numpy as np
import onetimepad
import requests
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta, time
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from pandas import DataFrame
from statsmodels.tsa.statespace.sarimax import SARIMAX#
# import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller


groupValuesBy = 'D'
deltaValue = 20
testPointsFactor = 0.15
forcast_days = 10
min_rpm = 300
min_amps = 50
tag_names = ["MtrAField1Temp"]#, "MtrAField2Temp", "MtrAInterPole1Temp", "MtrBField1Temp", "MtrBField2Temp", "MtrBInterPole1Temp"]
WINDOW_SIZE = 5
#tag_name = "MtrAInterPole1Temp"
#tag_name = "MtrBField2Temp"
#tag_name = "MtrAField1Temp"
#tag_name = "MtrAField2Temp"
#tag_name = "MtrBField1Temp"
#tag_name = "MtrBInterPole1Temp"
dataFromDate = '11/01/2022'
dataToDate = datetime.now().strftime("%m/%d/%Y")
#dataToDate = '01/31/2023'


# get spectare authentication token
def get_token():
    # get credentials
    config = configparser.ConfigParser()
    config.read('config.ini')
    random = "fw$45LNC81K$wk1V5#A7&k0&t2UCaqOGm^hoJn769&Vy2h3qsx"
    u = onetimepad.decrypt(config["DEFAULT"]["u"], random)
    p = onetimepad.decrypt(config["DEFAULT"]["p"], random)

    # get jwt_token
    headers = {
        # Already added when you pass json= but not when you pass data=
        'Content-Type': 'application/json',
        'Accept': 'application/json',
    }
    json_data = {'username': u, 'password': p}
    url = 'http://ss1.spectare-iss.com:8080/api/auth/login'
    response = requests.post(url, headers=headers, json=json_data).json()
    token = response['token']
    return token


# date to timestamp
def get_time_stamp(given_date):
    return datetime.strptime(given_date, '%m/%d/%Y').timestamp() * 1000


# get spectare temp data (as dayly max aggregate) and return dataframe of day, value(max-temp)
def get_lkq_temp(JWT_TOKEN, tag_name):
    headers = {
        'content-type': 'application/json',
        'X-Authorization': "Bearer " + JWT_TOKEN
    }
    # fromTime = 1660237200000    #Thursday, August 11, 2022 5:00:00 PM ::: Your time zone: Thursday, August 11, 2022 12:00:00 PM GMT-05:00 DST
    # toTime=1669053600000        #GMT: Monday, November 21, 2022 6:00:00 PM
    # toTime = 1669874400000
    fromTime = get_time_stamp(dataFromDate)
    toTime = get_time_stamp(dataToDate)

    # test
    # fromTime = 1674147604000  # Thursday, August 11, 2022 5:00:00 PM ::: Your time zone: Thursday, August 11, 2022 12:00:00 PM GMT-05:00 DST
    # toTime = 1674149584000  # GMT: Monday, November 21, 2022 6:00:00 PM

    response = requests.get('http://ss1.spectare-iss.com:8080/api/plugins/telemetry/DEVICE/41d8ce40-9e2c-11ec-bb60-a5b33c819c5a/values/timeseries?keys={tag_name},MtrRpm,MtrAmps&startTs={fromTime}&endTs={toTime}&limit=1000000'.format(
            tag_name=tag_name, fromTime=int(fromTime), toTime=int(toTime)), headers=headers)
    # response = requests.get(
    #     'http://ss1.spectare-iss.com:8080/api/plugins/telemetry/DEVICE/41d8ce40-9e2c-11ec-bb60-a5b33c819c5a/values/timeseries?keys=MtrBField2Temp&startTs={fromTime}&endTs={toTime}&limit=1000000'.format(
    #         fromTime=int(fromTime), toTime=int(toTime)), headers=headers)

    # Serializing json
    json_object = json.dumps(response.json(), indent=4)

    # Writing to data.json
    with open(tag_name+"_data.json", "w") as datafile:
        datafile.write(json_object)
        print(len(json_object))


def json_to_df(tempData):
    return pd.DataFrame(list(tempData.values())[0])


def clean_data(df):

    # change datatype of values to float
    df['temp'] = df['temp'].astype(float)

    df.index = df.index

    # resample data to daily maximum and daily minimum temperatures
    max_df = df.resample('D', offset='0h')['temp'].max()
    min_df = df.resample('D', offset='0h')['temp'].min()

    # merge the dataframes on timestamp ('ts') to calculate delta (the difference between max and min) and keep max only if delta is greater than "deltaValue"
    df_min_max = reduce(lambda left, right: pd.merge(left, right, on=['ts'], how='outer'), [max_df, min_df])
    df_min_max.columns = ['max_temp', 'min_temp']
    df_min_max['delta'] = df_min_max['max_temp'] - df_min_max['min_temp']
    df_min_max = df_min_max[df_min_max['delta'] >= deltaValue]
    df_max_temp = df_min_max.drop(columns=['min_temp', 'delta'])
    print(df_max_temp)

    return df_max_temp


def splitData(dayMaxTempReadingdf):

    # calculate number of test data points
    test_points = (int) (testPointsFactor * len(dayMaxTempReadingdf))
    print(test_points)
    #print(str(dayMaxTempReadingdf.min())+" --- "+str(dayMaxTempReadingdf[['value']].idxmin()))

    return dayMaxTempReadingdf[:-test_points], dayMaxTempReadingdf[-test_points:]


def predict_plot(train, test, tag_name):
    sns.set()
    plt.ylabel('Temp ('+tag_name+')')
    plt.xlabel('Date')
    plt.xticks(rotation=45)
    plt.plot(train.index, train['max_temp'], color="black")
    plt.plot(test.index, test['max_temp'], color="red")
    # plt.title(tag_name)
    # plt.show()

    y = train['max_temp']
    #y.index = y.index.to_period('D')

    SARIMAXmodel = sm.tsa.statespace.SARIMAX(y, order=(0, 0, 1), seasonal_order=(0, 1, 1, 12),
                                             enforce_stationarity=False, enforce_invertibility=False)
    SARIMAXmodel = SARIMAXmodel.fit(disp=False)

    y_pred = SARIMAXmodel.get_forecast(len(test.index))
    y_pred_df = y_pred.conf_int(alpha=0.05)
    y_pred_df["Predictions"] = SARIMAXmodel.predict(start=y_pred_df.index[0], end=y_pred_df.index[-1])
    y_pred_df.index = test.index
    y_pred_out = y_pred_df["Predictions"]
    plt.plot(y_pred_out, color='Blue', label='SARIMA Predictions')
    plt.legend()
    plt.show()


#get data forcast for (forcast_days) number of days
def predict_future(df):

    days = pd.date_range(df.index[-1] + timedelta(1), df.index[-1] + timedelta(days=10), freq='D')

    SARIMAXmodel = sm.tsa.statespace.SARIMAX(df['max_temp'], order=(0, 0, 1), seasonal_order=(0, 1, 1, 12),
                                             enforce_stationarity=False, enforce_invertibility=False)
    SARIMAXmodel = SARIMAXmodel.fit(disp=False)

    y_pred = SARIMAXmodel.get_forecast(forcast_days)
    y_pred_df = y_pred.conf_int(alpha=0.05)
    y_pred_df["Predictions"] = SARIMAXmodel.predict(start=y_pred_df.index[0], end=y_pred_df.index[-1])
    y_pred_df.index = days
    y_pred_out = y_pred_df["Predictions"]
    prediction_data = pd.DataFrame(y_pred_out).reset_index()
    prediction_data.columns = ['ts', 'value']
    prediction_data = prediction_data.set_index('ts')
    prediction_data.index = prediction_data.index.astype(np.int64)

    plt.plot(y_pred_out, color='Blue', label='SARIMA Predictions')
    plt.legend()
    plt.show()
    return prediction_data


# send forcast data to spectare
def prediction_to_spectare(data_forcast, tag_name, forcast):
    headers = {
        'Content-Type': 'application/json',
    }
    data_forcast.reset_index(inplace=True)
    data_forcast.columns = ['ts', tag_name]

    #data = json.dumps({tag_name: data_forcast.to_dict('records')}, indent = 4)

    telemetry = []
    if(forcast):
        data_forcast['ts'] = data_forcast['ts']/1000/1000
        #defaul_type = datetime
    else:
        data_forcast['ts'] = data_forcast.ts.astype('int64') // 10 ** 9
        #data_forcast['ts'] = pd.Timestamp(data_forcast['ts']).timestamp()  #pd.to_datetime(data_forcast['ts']).astype('int64') / 10**9
        #defaul_type = int
        #print(data_forcast.head())
    #print(data_forcast)
    #for j in range(len(data_forcast)):
    for ind in data_forcast.index:
        telemetry.append({"ts": data_forcast['ts'][ind], "values": {tag_name:data_forcast[tag_name][ind]}})
    print(telemetry)

    # prepare data to upload to spectare
    # convert data list to json
    # ensure default=datetime to take care of numpy type int64 mismatch error
    data = json.dumps(telemetry, default=datetime)
    #for i in range(len(telemetry)):
    if(forcast):
        response = requests.post('http://ss1.spectare-iss.com:8080/api/v1/Hj4BS1jdQ1SoIZKT0J8r/telemetry', headers=headers, data=data)
    else:
        response = requests.post('http://ss1.spectare-iss.com:8080/api/v1/unne6wcSK7oLbA2cEWL3/telemetry',
                                 headers=headers, data=data)
    print(response)


def df_to_x_y(df, window_size=5):

    df_as_np = df.to_numpy()
    x=[]
    y=[]
    for i in range(len(df_as_np)-window_size):
        row = [[a] for a in df_as_np[i:i+5]]
        x.append(row)
        label = df_as_np[i+5]
        y.append(label)

    return np.array(x), np.array(y)


def model_lstm(x, y):

    x_train, y_train = x[:60000], y[:60000]
    x_val, y_val = x[60000:65000], y[60000:65000]
    X_test, y_test = x[65000:], y[65000:]

    model1 = Sequential()
    model1.add(InputLayer((5,1)))
    model1.add(LSTM(64))
    model1.add(Dense(8, 'relu'))
    model1.add(Dense(1, 'linear'))
    print(model1.summary())

    cp1 = ModelCheckpoint('model1/', save_best_only=True)
    model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

    model1.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, callbacks=[cp1])


def to_sequences(dataset, seq_size=1):
    x = []
    y = []

    for i in range(len(dataset)-seq_size-1):
        window = dataset[i:(i+seq_size), 0]
        x.append(window)
        y.append(dataset[i+seq_size, 0])

    return np.array(x), np.array(y)

def main():
    JWT_TOKEN = get_token()

    for tag_name in tag_names:

        #get spectare data as json (if data not in file already)
        datafile = "C:\\Users\\Selamawit.Woldeamlak\\PycharmProjects\\predict_lkqTemp\\{tag_name}_data.json".format(tag_name = tag_name)
        print(datafile)
        if not os.path.exists(datafile) or os.path.getsize(datafile) <= 0:
            get_lkq_temp(JWT_TOKEN, tag_name)
            print("file not found")

        #else:
        # Opening JSON file
        f = open(tag_name+'_data.json')

        # returns JSON object as a dictionary
        tempData = json.load(f)
        print(len(list(tempData.keys())))

        temp_df = pd.DataFrame(list(tempData.values())[0], columns=['ts', 'value'])
        temp_df.columns = ['ts', 'temp']
        rpm_df = pd.DataFrame(list(tempData.values())[1], columns=['ts', 'value'])
        rpm_df.columns = ['ts', 'rpm']
        amps_df = pd.DataFrame(list(tempData.values())[2], columns=['ts', 'value'])
        amps_df.columns = ['ts', 'amps']

        #compose a list of the dataframes to merge
        merge_frames = [temp_df, rpm_df, amps_df]

        #merge the dataframes on timestamp ('ts')
        df_merged = reduce(lambda  left,right: pd.merge(left, right, on=['ts'], how='outer'), merge_frames).fillna('void')

        #write the merged output to csv file
        pd.DataFrame.to_csv(df_merged, 'merged_out.csv', sep=',', na_rep='--', index=False)

        #change values to float and remove records with rpm<300 or amps<50 (system operational)
        #df_merged['temp'] = df_merged['temp'].astype(float)
        df_merged['rpm'] = df_merged['rpm'].astype(float)
        df_merged['amps'] = df_merged['amps'].astype(float)

        df_filtered = df_merged.loc[(df_merged['rpm'] > min_rpm) & (df_merged['amps'] > min_amps), ['ts', "temp"]]

        #and change temp dataframe to timeseries dataframe with datetime index
        df_filtered['ts'] = pd.to_datetime(df_filtered['ts'], unit='ms')
        df_filtered = df_filtered.set_index(pd.DatetimeIndex(df_filtered['ts']))

        #adjust time to local time by adding 7 hours to timestamp index
        df_filtered.index = df_filtered.index + pd.DateOffset(hours=7)
        df_filtered = df_filtered.drop(columns=['ts'])
        df_filtered['temp'] = df_filtered['temp'].astype(float)

        # df_filtered['temp'].astype(float).plot()
        # plt.show()

        # write the filtered output to csv file
        pd.DataFrame.to_csv(df_filtered, 'filtered_out.csv', sep=',', na_rep='--', index=False)

        # Convert pandas dataframe to numpy array
        dataset = df_filtered.values

        #Normalize the dataset
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)

        print(dataset[0:10])

        train_size = int(len(dataset) * 0.66)
        test_size = len(dataset) - train_size
        train, test = dataset[0:train_size, :], dataset[train_size:len(dataset),:]

        seq_size = 5
        trainX, trainY = to_sequences(train, seq_size)
        testX, testY = to_sequences(test, seq_size)

        print("building deep model ...")

        # create and fit dense model
        model = Sequential()
        model.add(Dense(64, input_dim=seq_size, activation='relu'))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc'])
        print(model.summary())

        model.fit(trainX, trainY, validation_data=(testX, testY), verbose=2, epochs=100)

        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)

        trainPredict = scaler.inverse_transform(trainPredict)
        trainY_inverse = scaler.inverse_transform([trainY])
        testPredict = scaler.inverse_transform(testPredict)
        testY_inverse = scaler.inverse_transform([testY])

        trainScore = math.sqrt(mean_squared_error(trainY_inverse[0], trainPredict[:,0]))
        print('Train Score: %.2f RMSE' %(trainScore))

        testScore = math.sqrt(mean_squared_error(testY_inverse[0], testPredict[:, 0]))
        print('Test Score: %.2f RMSE' % (testScore))

        trainPredictPlot = np.empty_like(dataset)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[seq_size:len(trainPredict)+seq_size, :] = trainPredict

        testPredictPlot = np.empty_like(dataset)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(trainPredict)+(seq_size*2)++1:len(dataset)-1, :] = testPredict

        plt.plot(scaler.inverse_transform(dataset))
        plt.plot(trainPredictPlot)
        plt.plot(testPredictPlot)
        plt.show()

        #check if the data is stationary (Dickey-fuller test)
        # from statsmodels.tsa.stattools import adfuller
        # adf, pvalue, usedlag_, nobs_, critical_values_, icbest_ = adfuller(df_filtered)
        # print("pvalue = ", pvalue, " if above 0.5, data is not stationary")

        ###x, y = df_to_x_y(df_filtered, WINDOW_SIZE)

        #print(x.shape, y.shape)

        #tensorflow prediction on original data

        ###model_lstm(x, y)

        #tensorflow prediction on cleaned data

        # #clean data
        # cleanDataDf = clean_data(df_filtered)
        # # cleanDataDf['max_temp'].astype(float).plot()
        # # plt.show()
        # #print(cleanDataDf)
        # #
        # # #print(df4.groupby(pd.Grouper(freq='D')).value.agg(['max', 'idxmax']))
        # #
        # #split data into training and test
        # #train, test = splitData(cleanDataDf)
        # #print(train.head(68))
        # # print(test)
        #
        # #plot train, test; predict; and plot forcast
        # #predict_plot(train, test, tag_name)
        #
        # #predict future values
        # data_forcast = predict_future(cleanDataDf)
        #
        # # and send to spectare
        # #prediction_to_spectare(data_forcast, tag_name, True)
        #
        # # and also send max_tem history to spectare
        # #prediction_to_spectare(cleanDataDf, tag_name, False)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
