# Define a pattern for DATE column (D-M-Y H-M-S)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md

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

# Plot All Parkings for specific day
query_day = 5
query_month = 3

# Input and Output of Dataset
inputCSV = "OutputDatasets/Smart_Parking_Stays_JanFebMar_2017.csv"
title = "Parkings on "+str(query_day)+", "+str(query_month)+", 2017 (Sunday)"

plt.interactive(True)
plt.figure(figsize=(40, 40))

print("Opening Dataset...")
data = pd.read_csv(inputCSV)

print("Setting the Pattern in DateTime Columns...")
data[' Arrived'] = pd.to_datetime(data[' Arrived'], errors='ignore')
data[' Arrived'] = pd.to_datetime(data[" Arrived"].dt.strftime('%d/%m/%Y %H:%M'))
data[' Departed'] = pd.to_datetime(data[' Departed'], errors='ignore')
data[' Departed'] = pd.to_datetime(data[" Departed"].dt.strftime('%d/%m/%Y %H:%M'))

# Query For Specific Day
print("Setting Query for Specific Day and Month...")
data = data[(data[' Arrived'].dt.month == query_month) & (data[' Arrived'].dt.day == query_day)]

print("Removing Invalid Data...")
data = data.dropna()

print("Generating Graph...")
fig, ax = plt.subplots(figsize=(10,10))

plt.scatter(x=data[' Arrived'], y=data[' Departed'])
ax.set_xlim(data[' Arrived'].min()-pd.Timedelta(1,'h'),
            data[' Arrived'].max()+pd.Timedelta(1,'h'))

ax.set_ylim(data[' Departed'].min()-pd.Timedelta(1,'h'),
            data[' Departed'].max()+pd.Timedelta(1,'h'))


ax.xaxis.set_major_locator(md.HourLocator(interval = 1))
ax.xaxis.set_major_formatter(md.DateFormatter('%H:%M'))

ax.yaxis.set_major_locator(md.HourLocator(interval = 1))
ax.yaxis.set_major_formatter(md.DateFormatter('%H:%M'))

fig.autofmt_xdate()

plt.title(title+"\nTotal: "+str(len(data[' Departed'])), fontsize=30, pad=20)
plt.xlabel("Arrived Hour", fontsize=20, labelpad=20)
plt.ylabel("Departed Hour", fontsize=20, labelpad=20)

plt.grid()
plt.show()

exit()
