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
outputCSV = "OutputDatasets/Smart_Parking_Stays_JanFebMar_2017.csv"

print("Opening Dataset...")
data = pd.read_csv(inputCSV)
print(" - Initial Dataset size: ",len(data[' Arrived'])," rows.")

print("Setting the Pattern in DateTime Columns...")
data[' Arrived'] = pd.to_datetime(data[' Arrived'], errors='ignore')
data[' Departed'] = pd.to_datetime(data[' Departed'], errors='ignore')

# Queries to Separate a Part of the dataset

# Define only a year
print("Setting Query One: Year = 2017...")
nov_mask = data[(data[' Arrived'].dt.year == 2017)]

# Define only 3 months
print("Setting Query Two: Month = Jan or Fev or Mar...")
nov_mask = nov_mask[(nov_mask[' Arrived'].dt.month == 1) | (nov_mask[' Arrived'].dt.month == 2) | (nov_mask[' Arrived'].dt.month == 3)]
print(" - Actual Dataset size: ",len(nov_mask[' Arrived'])," rows.")


# Queries for Remove Trash Values

# First Type: Parkings with more than 24 hours
print("Setting Query Three: Remove Parkings with more than 24h...")
nov_mask = nov_mask[(nov_mask[' Departed'] - nov_mask[' Arrived']).astype('timedelta64[s]') < 86400]

# Second Trash Type: Parkings started in one day and departed in another day ( like enter on 22h and exit on 01h )
print("Setting Query Four: Remove Parkings started in one day and departed in another day...")
nov_mask = nov_mask[(nov_mask[' Arrived'].dt.day == nov_mask[' Departed'].dt.day)]

# Third Type: Parkings with less than 10 minutes
print("Setting Query Five: Remove Parkings with less than 10 minutes...")
nov_mask = nov_mask[(nov_mask[' Departed'] - nov_mask[' Arrived']).astype('timedelta64[s]') >= 600]
print(" - Final Dataset size: ",len(nov_mask[' Arrived'])," rows.")

print("Saving Output File...")
nov_mask.to_csv(outputCSV, encoding='utf-8-sig', date_format='%d/%m/%Y %H:%M', index=False, header=True)

print("Done!")

exit()
