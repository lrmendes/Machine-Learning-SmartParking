import pandas as pd
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

# Input and Output of Dataset
inputCSV = "Smart_Parking_Stays_Formatted.csv"
outputCSV = "OutputDatasets/Smart_Parking_Stays_Formatted_NumberOfParkingsArrivedGroupedByHours.csv"

#inputCSV = "OutputDatasets/Smart_Parking_Stays_JanFebMar_2017.csv"
#outputCSV = "OutputDatasets/Smart_Parking_Stays_JanFebMar_2017_NumberOfParkingsArrivedGroupedByHours.csv"

print("Opening Dataset...")
data = pd.read_csv(inputCSV)

print("Setting the Pattern in DateTime Columns...")
data[' Arrived'] = pd.to_datetime(data[' Arrived'], errors='ignore')
data[' Arrived'] = pd.to_datetime(data[" Arrived"].dt.strftime('%d/%m/%Y %H:%M'))

print("Groupping Data By Same Hour Parking Lots")
times = pd.DatetimeIndex(data[' Arrived'])
data_grouped_by_hour_count_size = data.groupby([times.hour]).size()

hours_range = ["00:00","01:00","02:00","03:00","04:00","05:00","06:00","07:00","08:00",
         "09:00","10:00","11:00","12:00","13:00","14:00","15:00","16:00","17:00",
         "18:00","19:00","20:00","21:00","22:00","23:00"]

print("Defining the Total Number of Parking Lots By Hour Range...")
newFrame = pd.DataFrame({'HourRange': hours_range, 'TotalParkingLots': data_grouped_by_hour_count_size})

print("Couting the Number of Days in the dataset...")
times = pd.DatetimeIndex(data[' Arrived'])
data_grouped_by_days_count_size = data.groupby([times.day,times.month,times.year]).size()

count_days = len(data_grouped_by_days_count_size)
print("Number of Different Days in the Dataset: ", count_days)

print("Defining the Daily Average of Parking Lots By Hour Range...")
newFrame['DailyAverageParkingLotsByHour'] = newFrame['TotalParkingLots'].apply(lambda x: x / count_days)

newFrame['DailyAverageParkingLotsByHour'] = newFrame['DailyAverageParkingLotsByHour'].round(2)

total_daily = newFrame['DailyAverageParkingLotsByHour'].sum(skipna=True)

newFrame['DailyParkingPercentageByHours'] = newFrame['DailyAverageParkingLotsByHour'].apply(lambda x: (x / total_daily))
newFrame['DailyParkingPercentageByHours'] = newFrame['DailyParkingPercentageByHours'].round(4)

print("Saving Output File...")
newFrame.to_csv(outputCSV, encoding='utf-8-sig', index=False, header=True)

exit()
