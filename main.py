#rev 00
#read lkq temperatures (around 100 days) with rpm, group data by day (day starts at 7am),
#clean data - consider max temp if 1. RPM is at or above the acceptable minimum value (min_rpm set at 300), and if delta of temp value (max-min) is more than 20 only)),
#and classify into training/test (85%/15%))
#predict next 15% (the same number as test days) days

import configparser
import json
import os
from functools import reduce

import numpy as np
import onetimepad
import requests
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

from pandas import DataFrame
from statsmodels.tsa.statespace.sarimax import SARIMAX#
# import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller


groupValuesBy = 'D'
deltaValue = 20
testPointsFactor = 0.15
min_rpm = 300
min_amps = 50
#tag_name = "MtrAInterPole1Temp"
#tag_name = "MtrBField2Temp"
tag_name = "MtrAField1Temp"
#tag_name = "MtrAField2Temp"
dataFromDate = '09/20/2022'
dataToDate = '12/31/2022'


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
def get_lkq_temp(JWT_TOKEN):
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

    max_df = df.resample('12h', offset='5h')['temp'].max()
    min_df = df.resample('12h', offset='5h')['temp'].min()

    # merge the dataframes on timestamp ('ts')
    df_min_max = reduce(lambda left, right: pd.merge(left, right, on=['ts'], how='outer'), [max_df, min_df])
    df_min_max.columns = ['max_temp', 'min_temp']
    df_min_max['delta'] = df_min_max['max_temp'] - df_min_max['min_temp']
    df_min_max = df_min_max[df_min_max['delta'] >= deltaValue]
    df_max_temp = df_min_max.drop(columns=['min_temp', 'delta'])
    print(df_max_temp)

    #groupby day and get max and ptp columns
    # df_min_max['max_temp'] = df_min_max['max_temp'].astype(float)
    # df_max_temp = df_min_max.resample("D")['max_temp'].max()
    return df_max_temp

    # # groupby date and get delta of max & min temp values
    # daily_delta_df1 = pd.DataFrame()
    # # daily_delta_df1['delta'] = df.resample(rule='D')['value'].agg(np.ptp)
    #
    # # group by 24 hours, and get delta (between max and min)
    # daily_delta_df1['delta'] = df.resample('24h', offset='19h', label='right')['temp'].agg(np.ptp)
    #
    # # remove values with delta of < deltaValue
    # df2 = daily_delta_df1[daily_delta_df1['delta'] >= deltaValue]
    # print(df2)
    #
    # # dataframe of daily maximum temperatures
    # daily_max_df = pd.DataFrame()
    # # daily_max_df['value'] = df.resample('D')['value'].max()
    #
    # # group by 24 hours
    # daily_max_df['temp'] = df.resample('24h', offset='7h', label='right')['temp'].max()
    #
    # print(daily_max_df)
    #
    # #alternative to daily maximum temperatures
    # # df = df.groupby(pd.Grouper(freq='D'))['value'].transform('max')
    # # print(len(df))
    #
    # # remove rows with NaN values
    # daily_max_df = daily_max_df.dropna()
    # print(daily_max_df)
    #
    # # print(daily_max_df.head(68))
    #
    # # match dates of df2 to filter only those dates of daily_max_df and return it
    # return daily_max_df[daily_max_df.index.isin(df2.index)]


def splitData(dayMaxTempReadingdf):

    # calculate number of test data points
    test_points = (int) (testPointsFactor * len(dayMaxTempReadingdf))
    print(test_points)
    #print(str(dayMaxTempReadingdf.min())+" --- "+str(dayMaxTempReadingdf[['value']].idxmin()))

    return dayMaxTempReadingdf[:-test_points], dayMaxTempReadingdf[-test_points:]


def predict_plot(train, test):
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

    SARIMAXmodel = sm.tsa.statespace.SARIMAX(y, order=(0, 0, 1), seasonal_order=(1, 1, 1, 12),
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


def main():
    JWT_TOKEN = get_token()

    #get spectare data as json (if data not in file already)
    datafile = "C:\\Users\\Selamawit.Woldeamlak\\PycharmProjects\\predict_lkqTemp\\{tag_name}_data.json".format(tag_name = tag_name)
    print(datafile)
    if not os.path.exists(datafile) or os.path.getsize(datafile) <= 0:
        get_lkq_temp(JWT_TOKEN)
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

    #df_time_temp.index + pd.DateOffset(hours=5)

    # write the filtered output to csv file
    pd.DataFrame.to_csv(df_filtered, 'filtered_out.csv', sep=',', na_rep='--', index=False)

    # temp_df['ts'] = temp_df['ts'] / 1000
    # # convert timestamp to datetime and in Eastern timezone
    # # df1['ts'] = pd.to_datetime(df1['ts'], unit='s').dt.tz_localize('utc').dt.tz_convert('US/Eastern')
    #
    # # convert timestamp to datetime
    # temp_df['ts'] = pd.to_datetime(temp_df['ts'], unit='s')
    # df1 = temp_df.set_index('ts')
    # rpm_df['ts'] = rpm_df['ts'] / 1000
    # # convert timestamp to datetime and in Eastern timezone
    # # df2['ts'] = pd.to_datetime(df2['ts'], unit='s').dt.tz_localize('utc').dt.tz_convert('US/Eastern')
    #
    # # convert timestamp to datetime and in Eastern timezone
    # rpm_df['ts'] = pd.to_datetime(rpm_df['ts'], unit='s')
    # df2 = rpm_df.set_index('ts')
    #
    # df = pd.merge(df1, df2, left_index=True, right_index=True)
    # df.index = pd.DatetimeIndex(df.index) #.to_period('D')
    # # remove data if rpm is < 300
    # df['rpm'] = df['rpm'].astype(float)
    # df3 = df[df['rpm'] > min_rpm]
    # df4 = df3.drop(columns=['rpm'])
    # df4.to_csv('test_out1.csv')

    # print(len(df1))
    # print(df2.tail())
    # print(df4.head())
    #
    #clean data
    cleanDataDf = clean_data(df_filtered)
    #print(cleanDataDf)
    #
    # #print(df4.groupby(pd.Grouper(freq='D')).value.agg(['max', 'idxmax']))
    #
    #split data into training and test
    train, test = splitData(cleanDataDf)
    #print(train.head(68))
    # print(test)

    #plot train, test; predict; and plot forcast
    predict_plot(train, test)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
