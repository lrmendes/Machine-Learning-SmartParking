import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as md
import numpy as np

"""
Column Names Reference:

SectorCode
 SectorName
 LotName
 BayNumber
 BayName
 Arrived
 Departed
Street
"""

"""
Street Names Reference:

Street               Number Of Parking Lots from Dataset
Bougainville St      82185
Franklin St          39702
Flinders Way         38177
Captain Cook Cres    32714
Multi Storey         29871
Furneaux St          13548
Palmerston Lane       1588

Get This Info with:
df3 = data['Street'].value_counts()
print(df3)
"""

# Input and Output of Dataset
inputCSV = "OutputDatasets/Smart_Parking_Stays_JanFebMar_2017.csv"
outputCSV = "OutputDatasets/Smart_Parking_Stays_JanFebMar_2017_AverageArrivedDepartedByDateRange.csv"
queryStreet = 'Bougainville St'
startDate = "13/03/2017"
endDate = "19/03/2017"

## Begin CODE
count_days_range = (pd.to_datetime(endDate, format='%d/%m/%Y') - pd.to_datetime(startDate, format='%d/%m/%Y'))
count_days_range = count_days_range.days + 1

print("Opening Dataset...")
data = pd.read_csv(inputCSV)

print("Setting the Pattern in DateTime Columns...")
data[' Arrived'] = pd.to_datetime(data[' Arrived'], errors='ignore')
data[' Arrived'] = pd.to_datetime(data[" Arrived"].dt.strftime('%d/%m/%Y %H:%M'))
data[' Departed'] = pd.to_datetime(data[' Departed'], errors='ignore')
data[' Departed'] = pd.to_datetime(data[" Departed"].dt.strftime('%d/%m/%Y %H:%M'))

print("Setting the StartDate && EndDate Filter in Dataset...")
data = data[(data[' Arrived'] >= pd.to_datetime(startDate, format='%d/%m/%Y')) & (data[' Arrived'] < pd.to_datetime(endDate, format='%d/%m/%Y') + pd.DateOffset(1))]

print("Checking if the dataset have data for all days in the input range...")
check_days_range = data[' Arrived'].dt.normalize().value_counts()

if len(check_days_range) - count_days_range != 0:
    print("The dataset does not have all the days in your input range (required for average)")
    exit()

print("Setting the StreetName Filter in Dataset...")
data = data[(data['Street'] == 'Bougainville St')]

print("Setting the Pattern in DateTime Columns...")
data[' Arrived'] = pd.to_datetime(data[' Arrived'], errors='ignore')
data[' Arrived'] = pd.to_datetime(data[" Arrived"].dt.strftime('%d/%m/%Y %H:%M'))
data[' Departed'] = pd.to_datetime(data[' Departed'], errors='ignore')
data[' Departed'] = pd.to_datetime(data[" Departed"].dt.strftime('%d/%m/%Y %H:%M'))

print("Removing Parkings with more than 24h...")
data = data[(data[' Departed'] - data[' Arrived']).astype('timedelta64[s]') < 86400]

print("Grouping by Hour (To Get The Average)...")
times_arrived = pd.DatetimeIndex(data[' Arrived'])
data_grouped_by_hour_arrived = data.groupby([times_arrived.hour]).size()

times_departed = pd.DatetimeIndex(data[' Departed'])
data_grouped_by_hour_departed = data.groupby([times_departed.hour]).size()

data_grouped_by_hour_arrived = data_grouped_by_hour_arrived.apply(lambda x: x/count_days_range)#.round(2)

data_grouped_by_hour_departed = data_grouped_by_hour_departed.apply(lambda x: x/count_days_range)#.round(2)

hours_range = ["00:00","01:00","02:00","03:00","04:00","05:00","06:00","07:00","08:00",
         "09:00","10:00","11:00","12:00","13:00","14:00","15:00","16:00","17:00",
         "18:00","19:00","20:00","21:00","22:00","23:00"]

newFrame = pd.DataFrame({'HourRange': hours_range,'ArrivedParkingLots': data_grouped_by_hour_arrived,'DepartedParkingLots': data_grouped_by_hour_departed})

newFrame.to_csv(outputCSV, encoding='utf-8-sig', index=False, header=True)