#using RNN for forcasting rev 00
#read lkq temperatures (around 100 days) with rpm, group data by day (day starts at 7am),
#clean data - consider max temp if 1. RPM is at or above the acceptable minimum value (min_rpm set at 300)

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
from keras.layers import LSTM, Flatten
from keras.callbacks import ModelCheckpoint
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from keras.layers import ConvLSTM2D


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
dataFromDate = '12/10/2022'
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
        df_merged = reduce(lambda  left, right: pd.merge(left, right, on=['ts'], how='outer'), merge_frames).fillna('void')

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

        print(dataset[0:10])

        # Normalize the dataset
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)

        train_size = int(len(dataset) * 0.66)
        test_size = len(dataset) - train_size
        train, test = dataset[0:train_size, :], dataset[train_size:len(dataset),:]

        seq_size = 7
        trainX, trainY = to_sequences(train, seq_size)
        testX, testY = to_sequences(test, seq_size)

        print("Shape of trainig set: {}".format(trainX.shape))
        print("Shape of test set: {}".format(testX.shape))

        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        print("Single LSTM with hidden Dense...")

        # create and fit dense model
        model = Sequential()
        model.add(LSTM(64, input_shape=(None, seq_size)))
        model.add(Dense(32))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.summary()
        print("Train...")

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
        #plt.plot(trainPredictPlot)
        plt.plot(testPredictPlot)
        plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
