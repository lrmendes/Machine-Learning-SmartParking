# Define a pattern for DATE column (D-M-Y H-M-S)
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
inputCSV = "Smart_Parking_Stays.csv"
outputCSV = "Smart_Parking_Stays_Formatted.csv"

print("Opening Dataset...")
data = pd.read_csv(inputCSV)

print("Setting Pattern to Arrived DateTime Column...")
data[' Arrived'] = pd.to_datetime(data[' Arrived'], errors='ignore')
data[' Arrived'] = pd.to_datetime(data[" Arrived"].dt.strftime('%d/%m/%Y %H:%M'))

print("Setting Pattern to Departed DateTime Column...")
data[' Departed'] = pd.to_datetime(data[' Departed'], errors='ignore')
data[' Departed'] = pd.to_datetime(data[" Departed"].dt.strftime('%d/%m/%Y %H:%M'))

print("Sorting Output Date For order of Arrival and Departure...")
data = data.sort_values([' Arrived',' Departed'])

print("Saving Output File...")
data.to_csv(outputCSV, encoding='utf-8-sig', date_format='%d/%m/%Y %H:%M', index=False, header=True)

print("Done!")

exit()

