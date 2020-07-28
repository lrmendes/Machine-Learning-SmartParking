# https://www.kaggle.com/mypapit/klccparking
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as md

# Query Params
query_day = 3
query_month = 6
query_year = 2016
output_graph_png = "OutputGraphs/parking_occupation_by_day_query.png"

print("[Filter 2] Opening Dataset...")
header_list = ["park_name", "available_spots","date"]
data = pd.read_csv('parking-klcc-2016-2017.txt', names=header_list, sep=';', engine='python')

# Define Numbers for "Open" & "Full" Values (From Dataset)
data = data.replace({'available_spots': 'FULL'}, 0)
data = data.replace({'available_spots': 'OPEN'}, np.nan)

# Remove OPEN (NA Values)
data = data.dropna()

data['available_spots'] = pd.to_numeric(data['available_spots'])

data['date'] = pd.to_datetime(data['date'])
data = data[(data['date'].dt.year == query_year) & (data['date'].dt.day == query_day) & (data['date'].dt.month == query_month)]

data['date'] = data['date'].dt.floor('T')
data = data[(data['date'].dt.minute == 0) | (data['date'].dt.minute == 15) |
            (data['date'].dt.minute == 30) | (data['date'].dt.minute == 45) ]

df_final = data.copy()
df_final['date'] = pd.to_datetime(df_final['date'])

#plot monthly sales
fig, ax = plt.subplots(figsize=(15,6))
plt.rcParams["figure.figsize"] = (15,10)

ax.set_xlim(df_final['date'].min(),
            df_final['date'].max())

ax.xaxis.set_major_locator(md.MinuteLocator(interval = 15))
ax.xaxis.set_major_formatter(md.DateFormatter('%H:%M'))

plt.axhline(y=5500, color='r', linestyle='dashed', label="Park Spots Empty")
plt.axhline(y=0, color='r', linestyle='dashed', label="Parking Spots Full")

plt.plot(df_final['date'], df_final['available_spots'])
plt.xticks(rotation=90, )

plt.title('Daily Parking Lot Occupation '+str(query_day)+'/'+str(query_month)+'/'+str(query_year)
          +'\n(15min verification)', fontsize=20,pad=10)

plt.xlabel("Time Range from Sensor",  labelpad=20,  fontsize=12)
plt.ylabel("Parking Spots Available", labelpad=20, fontsize=12)

plt.locator_params(axis='y', nbins=21)

plt.grid()
plt.show()

ax.figure.savefig(output_graph_png)
print("Saving Output Graph...")