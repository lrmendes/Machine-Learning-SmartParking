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

# Query Variable
query_hour = 12

# Input and Output of Dataset
inputCSV = "OutputDatasets/Smart_Parking_Stays_JanFebMar_2017.csv"
outputCSV = "OutputDatasets/Smart_Parking_Stays_JanFebMar_2017_ParkingsArrivedBetween"+str(query_hour)+"-"+str(query_hour+1)+"hours.csv"

title = "Parkings Arrived Between "+str(query_hour)+"h00 ~ "+str(query_hour)+"h59"

plt.interactive(True)
plt.figure(figsize=(40, 40))

print("Opening Dataset...")
data = pd.read_csv(inputCSV)

print("Setting the Pattern in DateTime Columns...")
data[' Arrived'] = pd.to_datetime(data[' Arrived'], errors='ignore')
data[' Arrived'] = pd.to_datetime(data[" Arrived"].dt.strftime('%d/%m/%Y %H:%M'))
data[' Departed'] = pd.to_datetime(data[' Departed'], errors='ignore')
data[' Departed'] = pd.to_datetime(data[" Departed"].dt.strftime('%d/%m/%Y %H:%M'))

# Query For Specific Hour
print("Setting Query for Specific Hour Range...")
data = data[(data[' Arrived'].dt.hour == query_hour)]

# Set All Dates as Same ( Because only Hours are important for it )
data[' Arrived'] = pd.to_datetime(data[" Arrived"].dt.strftime('01/01/2017 %H:%M'))
data[' Departed'] = pd.to_datetime(data[" Departed"].dt.strftime('01/01/2017 %H:%M'))

# Remove NA Values
print("Removing Invalid Data...")
data = data.dropna()

print("Generating Graph...")
fig, ax = plt.subplots(figsize=(14,10))
plt.scatter(x=data[' Arrived'], y=data[' Departed'])

ax.set_xlim(data[' Arrived'].min()-pd.Timedelta(5,'m'),
            data[' Arrived'].max()+pd.Timedelta(6,'m'))

ax.set_ylim(data[' Departed'].min()-pd.Timedelta(1,'h'),
            data[' Departed'].max()+pd.Timedelta(1,'h'))

ax.xaxis.set_major_locator(md.MinuteLocator(interval = 5))
ax.xaxis.set_major_formatter(md.DateFormatter('%H:%M'))

ax.yaxis.set_major_locator(md.HourLocator(interval = 1))
ax.yaxis.set_major_formatter(md.DateFormatter('%H:%M'))

fig.autofmt_xdate()

plt.title(title+"\nTotal: "+str(len(data[' Departed'])), fontsize=30, pad=20)
plt.xlabel("Arrived Hour", fontsize=20, labelpad=20)
plt.ylabel("Departed Hour", fontsize=20, labelpad=20)

plt.grid()
plt.show()

fig.savefig("Smart_Parking_Stays_JanFebMar_2017_ParkingsArrivedBetween"+str(query_hour)+"-"+str(query_hour+1)+"hours.png")

print("Saving Output File...")
data.to_csv(outputCSV, encoding='utf-8-sig', date_format='%d/%m/%Y %H:%M', index=False, header=True)

exit()
